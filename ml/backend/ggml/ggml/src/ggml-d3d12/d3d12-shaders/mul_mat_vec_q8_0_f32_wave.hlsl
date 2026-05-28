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

// Single-uint scale load. block_off is guaranteed to be 2-byte aligned, so the
// 16-bit half-float scale always fits inside one 4-byte word — saves one Load
// vs the byte-wise load_u16 path.
float load_f16_aligned2(RWByteAddressBuffer buf, uint off) {
    const uint word = buf.Load(off & ~3u);
    const uint half = (word >> ((off & 3u) * 8u)) & 0xFFFFu;
    return f16tof32(half);
}

float load_f16(RWByteAddressBuffer buf, uint off) {
    return f16tof32(load_u16(buf, off));
}

groupshared float partials[TG];

// Reduce `local` across the WG. Only the return value at gtid.x == 0u is
// meaningful — callers that need the result on every thread should broadcast
// via groupshared or WaveReadLaneAt themselves. Single barrier (vs the
// previous two-barrier scheme) since the final read-and-sum is done only by
// thread 0.
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

    float total = 0.0f;
    if (tid == 0u) {
        [unroll(8)] for (uint i = 0u; i < num_waves; ++i) {
            total += partials[i];
        }
    }
    return total;
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
    //     The 4 byte loads per lane are issued as a single uint Load
    //     (and one extra Load for the 2-byte-misaligned case where the
    //     block straddles a 4-byte word boundary).
    //   - activations: lanes read contiguous 4-float chunks via Load4,
    //     fully coalesced into a single 16-byte transaction per lane.
    //   - scale: lanes 0..7 broadcast-read the same f16 (one memory txn
    //     coalesced by the driver).
    const uint stride = TG * 4u;
    const uint k_main = K & ~3u;
    uint k = gtid.x * 4u;

    // Manually unrolled by 4: each iteration issues all loads for four
    // groups of 4 quants before doing any MACs. The interleaved loads
    // give the hardware more memory parallelism to hide latency, which
    // matters especially for the large-K case (K=18944) where per-WG
    // activation footprint exceeds the L1 working set.
    const uint stride4 = stride * 4u;
    [loop]
    while (k + stride * 3u + 4u <= k_main) {
        // ----- group 0 @ k -----
        const uint block_0     = k / QK;
        const uint elem_0      = k & (QK - 1u);
        const uint block_off_0 = row_base + block_0 * BLOCK_SIZE;
        const uint quants_off_0 = block_off_0 + QS_OFFSET + elem_0;
        const uint base_word_0 = quants_off_0 & ~3u;
        const uint shift_0     = (quants_off_0 & 3u) * 8u;
        const uint w0_0 = src0_buf.Load(base_word_0);
        const uint w1_0 = src0_buf.Load(base_word_0 + 4u);
        const uint w1s_0 = (shift_0 == 0u) ? 0u : (w1_0 << (32u - shift_0));
        const uint packed_0 = (w0_0 >> shift_0) | w1s_0;
        const float scale_0 = load_f16_aligned2(src0_buf, block_off_0);
        const uint4 a_packed_0 = src1_buf.Load4(vec_base + k * 4u);

        // ----- group 1 @ k + stride -----
        const uint k1 = k + stride;
        const uint block_1     = k1 / QK;
        const uint elem_1      = k1 & (QK - 1u);
        const uint block_off_1 = row_base + block_1 * BLOCK_SIZE;
        const uint quants_off_1 = block_off_1 + QS_OFFSET + elem_1;
        const uint base_word_1 = quants_off_1 & ~3u;
        const uint shift_1     = (quants_off_1 & 3u) * 8u;
        const uint w0_1 = src0_buf.Load(base_word_1);
        const uint w1_1 = src0_buf.Load(base_word_1 + 4u);
        const uint w1s_1 = (shift_1 == 0u) ? 0u : (w1_1 << (32u - shift_1));
        const uint packed_1 = (w0_1 >> shift_1) | w1s_1;
        const float scale_1 = load_f16_aligned2(src0_buf, block_off_1);
        const uint4 a_packed_1 = src1_buf.Load4(vec_base + k1 * 4u);

        // ----- group 2 @ k + 2*stride -----
        const uint k2 = k + stride * 2u;
        const uint block_2     = k2 / QK;
        const uint elem_2      = k2 & (QK - 1u);
        const uint block_off_2 = row_base + block_2 * BLOCK_SIZE;
        const uint quants_off_2 = block_off_2 + QS_OFFSET + elem_2;
        const uint base_word_2 = quants_off_2 & ~3u;
        const uint shift_2     = (quants_off_2 & 3u) * 8u;
        const uint w0_2 = src0_buf.Load(base_word_2);
        const uint w1_2 = src0_buf.Load(base_word_2 + 4u);
        const uint w1s_2 = (shift_2 == 0u) ? 0u : (w1_2 << (32u - shift_2));
        const uint packed_2 = (w0_2 >> shift_2) | w1s_2;
        const float scale_2 = load_f16_aligned2(src0_buf, block_off_2);
        const uint4 a_packed_2 = src1_buf.Load4(vec_base + k2 * 4u);

        // ----- group 3 @ k + 3*stride -----
        const uint k3 = k + stride * 3u;
        const uint block_3     = k3 / QK;
        const uint elem_3      = k3 & (QK - 1u);
        const uint block_off_3 = row_base + block_3 * BLOCK_SIZE;
        const uint quants_off_3 = block_off_3 + QS_OFFSET + elem_3;
        const uint base_word_3 = quants_off_3 & ~3u;
        const uint shift_3     = (quants_off_3 & 3u) * 8u;
        const uint w0_3 = src0_buf.Load(base_word_3);
        const uint w1_3 = src0_buf.Load(base_word_3 + 4u);
        const uint w1s_3 = (shift_3 == 0u) ? 0u : (w1_3 << (32u - shift_3));
        const uint packed_3 = (w0_3 >> shift_3) | w1s_3;
        const float scale_3 = load_f16_aligned2(src0_buf, block_off_3);
        const uint4 a_packed_3 = src1_buf.Load4(vec_base + k3 * 4u);

        // ----- compute group 0 -----
        const float dp_0 = mad(float((int)(packed_0 << 24) >> 24), asfloat(a_packed_0.x),
                           mad(float((int)(packed_0 << 16) >> 24), asfloat(a_packed_0.y),
                           mad(float((int)(packed_0 <<  8) >> 24), asfloat(a_packed_0.z),
                               float((int) packed_0         >> 24) * asfloat(a_packed_0.w))));
        acc = mad(scale_0, dp_0, acc);

        // ----- compute group 1 -----
        const float dp_1 = mad(float((int)(packed_1 << 24) >> 24), asfloat(a_packed_1.x),
                           mad(float((int)(packed_1 << 16) >> 24), asfloat(a_packed_1.y),
                           mad(float((int)(packed_1 <<  8) >> 24), asfloat(a_packed_1.z),
                               float((int) packed_1         >> 24) * asfloat(a_packed_1.w))));
        acc = mad(scale_1, dp_1, acc);

        // ----- compute group 2 -----
        const float dp_2 = mad(float((int)(packed_2 << 24) >> 24), asfloat(a_packed_2.x),
                           mad(float((int)(packed_2 << 16) >> 24), asfloat(a_packed_2.y),
                           mad(float((int)(packed_2 <<  8) >> 24), asfloat(a_packed_2.z),
                               float((int) packed_2         >> 24) * asfloat(a_packed_2.w))));
        acc = mad(scale_2, dp_2, acc);

        // ----- compute group 3 -----
        const float dp_3 = mad(float((int)(packed_3 << 24) >> 24), asfloat(a_packed_3.x),
                           mad(float((int)(packed_3 << 16) >> 24), asfloat(a_packed_3.y),
                           mad(float((int)(packed_3 <<  8) >> 24), asfloat(a_packed_3.z),
                               float((int) packed_3         >> 24) * asfloat(a_packed_3.w))));
        acc = mad(scale_3, dp_3, acc);

        k += stride4;
    }

    // Tail handler for groups of 1 (up to 3 remaining unrolled iters).
    [loop]
    while (k + 4u <= k_main) {
        const uint block     = k / QK;
        const uint elem      = k & (QK - 1u);
        const uint block_off = row_base + block * BLOCK_SIZE;
        const uint quants_off = block_off + QS_OFFSET + elem;

        const float scale = load_f16_aligned2(src0_buf, block_off);

        const uint base_word = quants_off & ~3u;
        const uint shift     = (quants_off & 3u) * 8u;
        const uint w0 = src0_buf.Load(base_word);
        const uint w1 = src0_buf.Load(base_word + 4u);
        const uint w1_shifted = (shift == 0u) ? 0u : (w1 << (32u - shift));
        const uint packed = (w0 >> shift) | w1_shifted;

        const int q0 = (int)(packed << 24) >> 24;
        const int q1 = (int)(packed << 16) >> 24;
        const int q2 = (int)(packed <<  8) >> 24;
        const int q3 = (int) packed         >> 24;

        const uint4 a_packed = src1_buf.Load4(vec_base + k * 4u);
        const float a0 = asfloat(a_packed.x);
        const float a1 = asfloat(a_packed.y);
        const float a2 = asfloat(a_packed.z);
        const float a3 = asfloat(a_packed.w);

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
