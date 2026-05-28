#define TG 64
#define QK 32u
#define BLOCK_SIZE 34u
#define QS_OFFSET 2u

RWByteAddressBuffer src0_buf : register(u0);
RWByteAddressBuffer src1_buf : register(u1);
RWByteAddressBuffer dst_buf  : register(u2);

cbuffer Params : register(b0) {
    uint K;
    uint M;
    uint batch;
    uint src0_row_stride;
    uint src1_row_stride;
    uint dst_row_stride;
    uint src0_off;
    uint src1_off;
    uint dst_off;
    uint ne2;
    uint src0_nb2;
    uint src0_nb3;
    uint src1_nb2;
    uint src1_nb3;
    uint dst_nb2;
    uint dst_nb3;
    uint broadcast2;
    uint broadcast3;
};

uint load_u8(RWByteAddressBuffer buf, uint off) {
    const uint word = buf.Load(off & ~3u);
    return (word >> ((off & 3u) * 8u)) & 0xFFu;
}

uint load_u16(RWByteAddressBuffer buf, uint off) {
    return load_u8(buf, off) | (load_u8(buf, off + 1u) << 8u);
}

float load_f16(RWByteAddressBuffer buf, uint off) {
    return f16tof32(load_u16(buf, off));
}

groupshared float partials[TG];

float wave_reduce_sum(float local, uint tid) {
    const uint lane = WaveGetLaneIndex();
    const uint wave_size = WaveGetLaneCount();
    const uint wave = tid / wave_size;
    const uint num_waves = (TG + wave_size - 1u) / wave_size;

    local = WaveActiveSum(local);
    if (lane == 0u) {
        partials[wave] = local;
    }
    GroupMemoryBarrierWithGroupSync();

    if (wave == 0u) {
        float total = 0.0f;
        for (uint i = lane; i < num_waves; i += wave_size) {
            total += partials[i];
        }
        total = WaveActiveSum(total);
        if (lane == 0u) {
            partials[0] = total;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return partials[0];
}

[numthreads(TG, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint m = gid.x;
    const uint b = gid.y;
    if (m >= M || b >= batch) {
        return;
    }

    const uint i2 = b % ne2;
    const uint i3 = b / ne2;
    const uint row_base = src0_off + (i2 / broadcast2) * src0_nb2 + (i3 / broadcast3) * src0_nb3 + m * src0_row_stride;
    const uint vec_base = src1_off + i2 * src1_nb2 + i3 * src1_nb3;

    float acc = 0.0f;

    // K_PER_ITER=4: each thread processes 4 consecutive quants per iteration.
    //
    // Memory access pattern across a warp of 32 threads (lane i handles
    // k = warp_base + i*4 .. i*4+3):
    //   - quant bytes: lanes 0..7 share one Q8_0 block (32 quant bytes,
    //     fully coalesced). Lanes 8..15 share the next block, etc.
    //     Driver coalesces the 4 byte loads per lane into one word load
    //     (offset is always multiple of 4 within the block).
    //   - activations: lanes read contiguous 4-float chunks → vec4-friendly.
    //   - scale: lanes 0..7 broadcast-read the same f16 (one memory txn).
    //
    // Net: ~4x fewer scale loads vs the per-quant variant, same coalesce,
    // FMAs grouped 4-wide so the compiler can schedule them in parallel.
    const uint stride = TG * 4u;
    const uint k_main = K & ~3u;
    uint k = gtid.x * 4u;
    [loop]
    while (k + 4u <= k_main) {
        const uint block     = k / QK;
        const uint elem      = k & (QK - 1u);
        const uint block_off = row_base + block * BLOCK_SIZE;
        const uint quants_off = block_off + QS_OFFSET + elem;

        const float scale = load_f16(src0_buf, block_off);

        // 4 contiguous quant bytes. quants_off is always 4-byte aligned
        // because elem is a multiple of 4 and (row_base + block*34 + 2)
        // alignment cancels: 34*block + 2 ≡ 2*(block+1) (mod 4), so
        // quants_off ≡ row_base + 2*(block+1) + 4*(elem/4) (mod 4). The
        // residue can be 0/2, so we may still be 2-aligned not 4-aligned —
        // fall back to load_u8 quartet which the driver coalesces.
        const uint b0 = load_u8(src0_buf, quants_off + 0u);
        const uint b1 = load_u8(src0_buf, quants_off + 1u);
        const uint b2 = load_u8(src0_buf, quants_off + 2u);
        const uint b3 = load_u8(src0_buf, quants_off + 3u);

        const int q0 = (int(b0) << 24) >> 24;
        const int q1 = (int(b1) << 24) >> 24;
        const int q2 = (int(b2) << 24) >> 24;
        const int q3 = (int(b3) << 24) >> 24;

        const float a0 = asfloat(src1_buf.Load(vec_base + (k + 0u) * 4u));
        const float a1 = asfloat(src1_buf.Load(vec_base + (k + 1u) * 4u));
        const float a2 = asfloat(src1_buf.Load(vec_base + (k + 2u) * 4u));
        const float a3 = asfloat(src1_buf.Load(vec_base + (k + 3u) * 4u));

        const float dp = mad(float(q0), a0,
                         mad(float(q1), a1,
                         mad(float(q2), a2,
                             float(q3) * a3)));
        acc = mad(scale, dp, acc);

        k += stride;
    }

    // Tail loop for the (unlikely) case where K is not a multiple of 4.
    while (k < K) {
        const uint block     = k / QK;
        const uint elem      = k & (QK - 1u);
        const uint block_off = row_base + block * BLOCK_SIZE;
        const int q = (int(load_u8(src0_buf, block_off + QS_OFFSET + elem)) << 24) >> 24;
        const float w = float(q) * load_f16(src0_buf, block_off);
        const float a = asfloat(src1_buf.Load(vec_base + k * 4u));
        acc = mad(w, a, acc);
        k += TG;
    }

    float total = wave_reduce_sum(acc, gtid.x);
    if (gtid.x == 0u) {
        dst_buf.Store(dst_off + i2 * dst_nb2 + i3 * dst_nb3 + m * 4u, asuint(total));
    }
}
