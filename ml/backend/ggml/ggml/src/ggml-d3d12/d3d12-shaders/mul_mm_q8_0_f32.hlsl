// Q8_0 weight × F32 activation tiled matmul.
//
// Output tile: 64x64 per WG. Dispatcher in d3d12-ops-mulmm.hpp uses
// gx=ceil(N/64), gy=ceil(M/64) for this shader (was 32 before).
//
// Layout: 256 threads/WG arranged 16x16, each accumulates a 4x4 sub-tile
// of outputs. TK=32 matches one full Q8_0 block, so each cooperative load
// pulls exactly one block per row (one f16 scale + 32 int8 quants).
//
// Bigger tile vs the 32x32 variant: each WG processes 4x more outputs,
// roughly halving the activation/weight re-reads across WGs for the same
// total output volume. Trade-off: 16 KB groupshared (vs 4 KB), 256
// threads/WG (vs 64), so a few-times fewer WGs in flight per SM.
//
// Loads per WG per k0 iter:
//   sa: 64 rows x 32 quants = 2048 quants. tid = row*4 + quarter;
//       each thread loads 8 quants of one row via 2-3 packed dword loads.
//   sb: 32 K x 64 cols = 2048 floats. tid = col*4 + quarter;
//       each thread loads 8 consecutive K-elements of one col via 2 Load4s.
//
// Compute per thread per k0 iter:
//   32 unrolled k-steps, each doing 4 sa-reads + 4 sb-reads + 16 MACs.
//   Total: 512 MACs/thread/iter.
//
// sa is padded as [TM][TK+1] to break the stride-32 bank conflict pattern
// when threads read 4 consecutive m-rows for the same k.

#define TM       64u
#define TN       64u
#define TK       32u    // == Q8_0 block size
#define TM_PER    4u
#define TN_PER    4u
#define TG_M     16u    // threads in M dim (TM = TM_PER * TG_M)
#define TG_N     16u    // threads in N dim (TN = TN_PER * TG_N)
#define TG       (TG_M * TG_N)   // 256

#define Q8_QK         32u
#define Q8_BLOCK_SIZE 34u   // 2 (f16 scale) + 32 (int8 quants)
#define Q8_QS_OFFSET   2u

RWByteAddressBuffer src0_buf : register(u0);
RWByteAddressBuffer src1_buf : register(u1);
RWByteAddressBuffer dst_buf  : register(u2);

cbuffer Params : register(b0) {
    uint M;
    uint N;
    uint K;
    uint batch_ne2;
    uint broadcast2;
    uint broadcast3;
    uint src0_off;
    uint src1_off;
    uint dst_off;
    uint src0_nb1;
    uint src0_nb2;
    uint src0_nb3;
};

// Padded K dim to avoid bank conflicts when 4 consecutive m-rows read
// the same k (stride TK == 32 lands on the same bank without the pad).
groupshared float sa[TM][TK + 1u];
groupshared float sb[TK][TN];

float load_f16_aligned2(uint off) {
    const uint word = src0_buf.Load(off & ~3u);
    const uint half = (word >> ((off & 3u) * 8u)) & 0xFFFFu;
    return f16tof32(half);
}

[numthreads(TG, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint tid       = gtid.x;
    const uint thread_m  = tid / TG_N;             // 0..15
    const uint thread_n  = tid - thread_m * TG_N;  // 0..15

    const uint batch = gid.z;
    const uint b2    = batch % batch_ne2;
    const uint b3    = batch / batch_ne2;
    const uint s0b2  = (broadcast2 == 0u) ? b2 : (b2 / broadcast2);
    const uint s0b3  = (broadcast3 == 0u) ? b3 : (b3 / broadcast3);
    const uint src0_base = src0_off + s0b2 * src0_nb2 + s0b3 * src0_nb3;
    const uint src1_base = src1_off + batch * N * K * 4u;
    const uint dst_base  = dst_off  + batch * N * M * 4u;

    const uint tile_m_base = gid.y * TM;
    const uint tile_n_base = gid.x * TN;

    // sa load mapping: tid = row*4 + quarter. 64 rows × 4 threads = 256 threads.
    // Each thread loads 8 quants (2 packed dwords) covering 8 of the 32 quants
    // of one row. Quarter (0..3) selects which 8-quant chunk.
    const uint sa_row     = tid >> 2u;       // 0..63
    const uint sa_q       = tid & 3u;        // 0..3
    const uint sa_k_base  = sa_q * 8u;       // 0,8,16,24
    const uint sa_gm      = tile_m_base + sa_row;
    const uint sa_row_base = src0_base + sa_gm * src0_nb1;

    // sb load mapping: tid = col*4 + quarter. 64 cols × 4 threads = 256 threads.
    // Each thread loads 8 K-elements (2 Load4s) of one col.
    const uint sb_col     = tid >> 2u;       // 0..63
    const uint sb_q       = tid & 3u;        // 0..3
    const uint sb_k_base  = sb_q * 8u;
    const uint sb_gn      = tile_n_base + sb_col;
    const uint sb_col_base = src1_base + sb_gn * K * 4u;

    // Per-thread output tile: rows [m_base_local..+3], cols [n_base_local..+3].
    const uint m_base_local = thread_m * TM_PER;
    const uint n_base_local = thread_n * TN_PER;

    float acc[TM_PER][TN_PER];
    [unroll] for (uint ai = 0u; ai < TM_PER; ++ai)
        [unroll] for (uint aj = 0u; aj < TN_PER; ++aj)
            acc[ai][aj] = 0.0f;

    [loop]
    for (uint k0 = 0u; k0 < K; k0 += TK) {
        // ---------------- Load sa (Q8_0 -> float) ----------------
        if (sa_gm < M) {
            const uint block_off    = sa_row_base + (k0 / Q8_QK) * Q8_BLOCK_SIZE;
            const float scale       = load_f16_aligned2(block_off);
            const uint quants_base  = block_off + Q8_QS_OFFSET + sa_k_base;

            // Composite 2 packed dwords (8 int8 quants) from at most 3
            // raw 32-bit loads, regardless of alignment.
            const uint base   = quants_base & ~3u;
            const uint shift  = (quants_base & 3u) * 8u;
            const uint w0 = src0_buf.Load(base);
            const uint w1 = src0_buf.Load(base + 4u);
            const uint w2 = (shift == 0u) ? 0u : src0_buf.Load(base + 8u);
            const uint inv_shift = 32u - shift;
            const uint p0 = (shift == 0u) ? w0 : ((w0 >> shift) | (w1 << inv_shift));
            const uint p1 = (shift == 0u) ? w1 : ((w1 >> shift) | (w2 << inv_shift));

            sa[sa_row][sa_k_base + 0u] = scale * float((int)(p0 << 24u) >> 24);
            sa[sa_row][sa_k_base + 1u] = scale * float((int)(p0 << 16u) >> 24);
            sa[sa_row][sa_k_base + 2u] = scale * float((int)(p0 <<  8u) >> 24);
            sa[sa_row][sa_k_base + 3u] = scale * float((int) p0         >> 24);
            sa[sa_row][sa_k_base + 4u] = scale * float((int)(p1 << 24u) >> 24);
            sa[sa_row][sa_k_base + 5u] = scale * float((int)(p1 << 16u) >> 24);
            sa[sa_row][sa_k_base + 6u] = scale * float((int)(p1 <<  8u) >> 24);
            sa[sa_row][sa_k_base + 7u] = scale * float((int) p1         >> 24);
        } else {
            [unroll] for (uint i = 0u; i < 8u; ++i) {
                sa[sa_row][sa_k_base + i] = 0.0f;
            }
        }

        // ---------------- Load sb (F32 activations) ----------------
        if (sb_gn < N) {
            const uint k_start  = k0 + sb_k_base;
            const uint load_off = sb_col_base + k_start * 4u;
            const uint4 v0 = src1_buf.Load4(load_off);
            const uint4 v1 = src1_buf.Load4(load_off + 16u);
            sb[sb_k_base + 0u][sb_col] = asfloat(v0.x);
            sb[sb_k_base + 1u][sb_col] = asfloat(v0.y);
            sb[sb_k_base + 2u][sb_col] = asfloat(v0.z);
            sb[sb_k_base + 3u][sb_col] = asfloat(v0.w);
            sb[sb_k_base + 4u][sb_col] = asfloat(v1.x);
            sb[sb_k_base + 5u][sb_col] = asfloat(v1.y);
            sb[sb_k_base + 6u][sb_col] = asfloat(v1.z);
            sb[sb_k_base + 7u][sb_col] = asfloat(v1.w);
        } else {
            [unroll] for (uint i = 0u; i < 8u; ++i) {
                sb[sb_k_base + i][sb_col] = 0.0f;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // ---------------- 4x4 register-tiled inner product ----------------
        [unroll]
        for (uint k = 0u; k < TK; ++k) {
            float a0 = sa[m_base_local + 0u][k];
            float a1 = sa[m_base_local + 1u][k];
            float a2 = sa[m_base_local + 2u][k];
            float a3 = sa[m_base_local + 3u][k];
            float b0 = sb[k][n_base_local + 0u];
            float b1 = sb[k][n_base_local + 1u];
            float b2 = sb[k][n_base_local + 2u];
            float b3 = sb[k][n_base_local + 3u];

            acc[0][0] = mad(a0, b0, acc[0][0]);
            acc[0][1] = mad(a0, b1, acc[0][1]);
            acc[0][2] = mad(a0, b2, acc[0][2]);
            acc[0][3] = mad(a0, b3, acc[0][3]);
            acc[1][0] = mad(a1, b0, acc[1][0]);
            acc[1][1] = mad(a1, b1, acc[1][1]);
            acc[1][2] = mad(a1, b2, acc[1][2]);
            acc[1][3] = mad(a1, b3, acc[1][3]);
            acc[2][0] = mad(a2, b0, acc[2][0]);
            acc[2][1] = mad(a2, b1, acc[2][1]);
            acc[2][2] = mad(a2, b2, acc[2][2]);
            acc[2][3] = mad(a2, b3, acc[2][3]);
            acc[3][0] = mad(a3, b0, acc[3][0]);
            acc[3][1] = mad(a3, b1, acc[3][1]);
            acc[3][2] = mad(a3, b2, acc[3][2]);
            acc[3][3] = mad(a3, b3, acc[3][3]);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // ---------------- Store output 4x4 tile ----------------
    const uint out_m_base = tile_m_base + m_base_local;
    const uint out_n_base = tile_n_base + n_base_local;
    [unroll]
    for (uint mi = 0u; mi < TM_PER; ++mi) {
        [unroll]
        for (uint ni = 0u; ni < TN_PER; ++ni) {
            const uint gm = out_m_base + mi;
            const uint gn = out_n_base + ni;
            if (gm < M && gn < N) {
                dst_buf.Store(dst_base + gn * M * 4u + gm * 4u, asuint(acc[mi][ni]));
            }
        }
    }
}
