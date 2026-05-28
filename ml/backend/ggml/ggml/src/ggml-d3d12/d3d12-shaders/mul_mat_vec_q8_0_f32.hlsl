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

float load_q8_0(uint row_base, uint k) {
    const uint block = k / QK;
    const uint elem = k & (QK - 1u);
    const uint block_off = row_base + block * BLOCK_SIZE;
    const uint byte_q = load_u8(src0_buf, block_off + QS_OFFSET + elem);
    const int q = (int(byte_q) << 24) >> 24;
    return float(q) * load_f16(src0_buf, block_off);
}

groupshared float partials[TG];

[numthreads(TG, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint m = gid.x;
    const uint b = gid.y;
    if (m >= M || b >= batch) {
        return;
    }

    const uint i2 = b % ne2;
    const uint i3 = b / ne2;
    const uint row_base = src0_off + i2 * src0_nb2 + i3 * src0_nb3 + m * src0_row_stride;
    const uint vec_base = src1_off + i2 * src1_nb2 + i3 * src1_nb3;

    float acc = 0.0f;
    for (uint k = gtid.x; k < K; k += TG) {
        const float w = load_q8_0(row_base, k);
        const float a = asfloat(src1_buf.Load(vec_base + k * 4u));
        acc += w * a;
    }

    partials[gtid.x] = acc;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = TG / 2u; stride > 0u; stride >>= 1u) {
        if (gtid.x < stride) {
            partials[gtid.x] += partials[gtid.x + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gtid.x == 0u) {
        dst_buf.Store(dst_off + i2 * dst_nb2 + i3 * dst_nb3 + m * 4u, asuint(partials[0]));
    }
}
