// Q8_0 weight × F32 activation tiled matmul.
//
// Output tile: 128x64 per WG. Dispatcher in d3d12-ops-mulmm.hpp uses
// gx=ceil(N/64), gy=ceil(M/128) for this shader.
//
// Layout: 512 threads/WG arranged 32x16, each accumulates a 4x4 sub-tile
// of outputs. TK=32 matches one full Q8_0 block, so each cooperative load
// pulls exactly one block per row (one f16 scale + 32 int8 quants).
//
// Wider M tile halves src1 (activation) re-reads across the M dimension
// (gy = ceil(M/128) vs ceil(M/64)). Per-thread compute, per-thread
// groupshared reads, and per-thread accumulators are unchanged (still
// 4x4 = 16 outputs/thread). Groupshared grows from ~16 KB to ~25 KB
// (sa = 128 * 33 * 4 = 16896 B; sb = 32 * 64 * 4 = 8192 B). Still
// well under Ampere's 99 KB per-block limit, but block-per-SM
// concurrency drops from 6 (256-thread) to 3 (512-thread) due to the
// 1536-thread-per-SM cap.
//
// Loads per WG per k0 iter:
//   sa: 128 rows x 32 quants = 4096 quants. tid = row*4 + quarter;
//       each thread loads 8 quants of one row via 2-3 packed dword loads.
//       512 threads cover 128 rows x 4 quarters exactly.
//   sb: 32 K x 64 cols = 2048 floats. With 512 threads each loads
//       1 Load4: tid = col*8 + chunk; sb_k_base = chunk*4 (0..28).
//
// Compute per thread per k0 iter:
//   32 unrolled k-steps, each doing 4 sa-reads + 4 sb-reads + 16 MACs.
//   Total: 512 MACs/thread/iter.
//
// sa is padded as [TM][TK+1] to break the stride-32 bank conflict pattern
// when threads read 4 consecutive m-rows for the same k.

#define TM      128u
#define TN       64u
#define TK       32u    // == Q8_0 block size
#define TM_PER    4u
#define TN_PER    4u
#define TG_M     32u    // threads in M dim (TM = TM_PER * TG_M)
#define TG_N     16u    // threads in N dim (TN = TN_PER * TG_N)
#define TG       (TG_M * TG_N)   // 512

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

    // sa load mapping: tid = row*4 + quarter. 128 rows × 4 threads = 512 threads.
    // Each thread loads 8 quants (2 packed dwords) covering 8 of the 32 quants
    // of one row. Quarter (0..3) selects which 8-quant chunk.
    const uint sa_row     = tid >> 2u;       // 0..127
    const uint sa_q       = tid & 3u;        // 0..3
    const uint sa_k_base  = sa_q * 8u;       // 0,8,16,24
    const uint sa_gm      = tile_m_base + sa_row;
    const uint sa_row_base = src0_base + sa_gm * src0_nb1;

    // sb load mapping: tid = col*8 + chunk. 64 cols × 8 threads = 512 threads.
    // Each thread loads 4 K-elements (1 Load4) of one col. Chunk (0..7)
    // selects which 4-K-element block.
    const uint sb_col     = tid >> 3u;       // 0..63
    const uint sb_q       = tid & 7u;        // 0..7
    const uint sb_k_base  = sb_q * 4u;       // 0,4,...,28
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
            sb[sb_k_base + 0u][sb_col] = asfloat(v0.x);
            sb[sb_k_base + 1u][sb_col] = asfloat(v0.y);
            sb[sb_k_base + 2u][sb_col] = asfloat(v0.z);
            sb[sb_k_base + 3u][sb_col] = asfloat(v0.w);
        } else {
            [unroll] for (uint i = 0u; i < 4u; ++i) {
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
