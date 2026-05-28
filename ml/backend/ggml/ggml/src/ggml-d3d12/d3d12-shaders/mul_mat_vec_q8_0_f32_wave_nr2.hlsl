// NUM_ROWS=2 wave-reduce Q8_0 GEMV. Each WG computes 2 consecutive output rows
// (m*2+0 and m*2+1) sharing one activation read per inner iteration. Activation
// bandwidth is ~halved vs the single-row wave shader on big-M shapes (FFN
// gate/up/down).
//
// Dispatcher dispatches ceil(M/2) WGs along x; we guard the second row when it
// would be out of bounds (odd M tail).

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

// Single-uint scale load. block_off is 2-byte aligned, so the half always
// fits inside one 4-byte word — saves one Load vs the byte-wise load_u16 path.
float load_f16_aligned2(RWByteAddressBuffer buf, uint off) {
    const uint word = buf.Load(off & ~3u);
    const uint half = (word >> ((off & 3u) * 8u)) & 0xFFFFu;
    return f16tof32(half);
}

float load_f16(RWByteAddressBuffer buf, uint off) {
    return f16tof32(load_u16(buf, off));
}

groupshared float partials_a[TG / 32u];
groupshared float partials_b[TG / 32u];

float wave_reduce_sum_2(float local, uint tid, uint slot) {
    const uint lane = WaveGetLaneIndex();
    const uint wave_size = WaveGetLaneCount();
    const uint wave = tid / wave_size;
    const uint num_waves = (TG + wave_size - 1u) / wave_size;

    local = WaveActiveSum(local);
    if (lane == 0u) {
        if (slot == 0u) partials_a[wave] = local;
        else            partials_b[wave] = local;
    }
    GroupMemoryBarrierWithGroupSync();

    float total = 0.0f;
    if (wave == 0u) {
        for (uint i = lane; i < num_waves; i += wave_size) {
            total += (slot == 0u) ? partials_a[i] : partials_b[i];
        }
        total = WaveActiveSum(total);
    }
    GroupMemoryBarrierWithGroupSync();
    return total;
}

[numthreads(TG, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint m_base = gid.x * 2u;
    const uint b = gid.y;
    if (m_base >= M || b >= batch) {
        return;
    }
    const bool have_m1 = (m_base + 1u) < M;

    const uint i2 = b % ne2;
    const uint i3 = b / ne2;
    const uint row_batch_off = src0_off + (i2 / broadcast2) * src0_nb2 + (i3 / broadcast3) * src0_nb3;
    const uint row0_base = row_batch_off + m_base * src0_row_stride;
    const uint row1_base = row_batch_off + (m_base + 1u) * src0_row_stride;
    const uint vec_base  = src1_off + i2 * src1_nb2 + i3 * src1_nb3;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const uint stride = TG * 4u;
    const uint k_main = K & ~3u;
    uint k = gtid.x * 4u;
    [loop]
    while (k + 4u <= k_main) {
        const uint block     = k / QK;
        const uint elem      = k & (QK - 1u);
        const uint block0_off = row0_base + block * BLOCK_SIZE;
        const uint block1_off = row1_base + block * BLOCK_SIZE;
        const uint quants0_off = block0_off + QS_OFFSET + elem;
        const uint quants1_off = block1_off + QS_OFFSET + elem;

        const float scale0 = load_f16_aligned2(src0_buf, block0_off);
        const float scale1 = have_m1 ? load_f16_aligned2(src0_buf, block1_off) : 0.0f;

        // Row 0 quants
        const uint base0_word = quants0_off & ~3u;
        const uint shift0     = (quants0_off & 3u) * 8u;
        uint packed0;
        if (shift0 == 0u) {
            packed0 = src0_buf.Load(base0_word);
        } else {
            const uint w0a = src0_buf.Load(base0_word);
            const uint w0b = src0_buf.Load(base0_word + 4u);
            packed0 = (w0a >> shift0) | (w0b << (32u - shift0));
        }
        const int q0_0 = (int)(packed0 << 24) >> 24;
        const int q0_1 = (int)(packed0 << 16) >> 24;
        const int q0_2 = (int)(packed0 <<  8) >> 24;
        const int q0_3 = (int) packed0         >> 24;

        // Row 1 quants (only when present)
        int q1_0 = 0, q1_1 = 0, q1_2 = 0, q1_3 = 0;
        if (have_m1) {
            const uint base1_word = quants1_off & ~3u;
            const uint shift1     = (quants1_off & 3u) * 8u;
            uint packed1;
            if (shift1 == 0u) {
                packed1 = src0_buf.Load(base1_word);
            } else {
                const uint w1a = src0_buf.Load(base1_word);
                const uint w1b = src0_buf.Load(base1_word + 4u);
                packed1 = (w1a >> shift1) | (w1b << (32u - shift1));
            }
            q1_0 = (int)(packed1 << 24) >> 24;
            q1_1 = (int)(packed1 << 16) >> 24;
            q1_2 = (int)(packed1 <<  8) >> 24;
            q1_3 = (int) packed1         >> 24;
        }

        // Shared activation load (the entire point of NUM_ROWS=2)
        const uint4 a_packed = src1_buf.Load4(vec_base + k * 4u);
        const float a0 = asfloat(a_packed.x);
        const float a1 = asfloat(a_packed.y);
        const float a2 = asfloat(a_packed.z);
        const float a3 = asfloat(a_packed.w);

        const float dp0 = mad(float(q0_0), a0,
                          mad(float(q0_1), a1,
                          mad(float(q0_2), a2,
                              float(q0_3) * a3)));
        acc0 = mad(scale0, dp0, acc0);

        if (have_m1) {
            const float dp1 = mad(float(q1_0), a0,
                              mad(float(q1_1), a1,
                              mad(float(q1_2), a2,
                                  float(q1_3) * a3)));
            acc1 = mad(scale1, dp1, acc1);
        }

        k += stride;
    }

    // Tail loop for the (unlikely) case where K is not a multiple of 4.
    while (k < K) {
        const uint block     = k / QK;
        const uint elem      = k & (QK - 1u);
        const uint block0_off = row0_base + block * BLOCK_SIZE;
        const int q0 = (int(load_u8(src0_buf, block0_off + QS_OFFSET + elem)) << 24) >> 24;
        const float w0 = float(q0) * load_f16(src0_buf, block0_off);
        const float a = asfloat(src1_buf.Load(vec_base + k * 4u));
        acc0 = mad(w0, a, acc0);
        if (have_m1) {
            const uint block1_off = row1_base + block * BLOCK_SIZE;
            const int q1 = (int(load_u8(src0_buf, block1_off + QS_OFFSET + elem)) << 24) >> 24;
            const float w1 = float(q1) * load_f16(src0_buf, block1_off);
            acc1 = mad(w1, a, acc1);
        }
        k += TG;
    }

    const float total0 = wave_reduce_sum_2(acc0, gtid.x, 0u);
    const float total1 = have_m1 ? wave_reduce_sum_2(acc1, gtid.x, 1u) : 0.0f;
    if (gtid.x == 0u) {
        const uint dst_batch_off = dst_off + i2 * dst_nb2 + i3 * dst_nb3;
        dst_buf.Store(dst_batch_off + m_base * 4u, asuint(total0));
        if (have_m1) {
            dst_buf.Store(dst_batch_off + (m_base + 1u) * 4u, asuint(total1));
        }
    }
}
