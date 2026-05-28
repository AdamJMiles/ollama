#define TG 64

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

uint16_t load_u16(RWByteAddressBuffer buf, uint off) {
    return (uint16_t)(load_u8(buf, off) | (load_u8(buf, off + 1u) << 8u));
}

half load_f16h(RWByteAddressBuffer buf, uint off) {
    return (half)f16tof32((uint)load_u16(buf, off));
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
    const uint row_base = src0_off + (i2 / broadcast2) * src0_nb2 + (i3 / broadcast3) * src0_nb3 + m * src0_row_stride;
    const uint vec_base = src1_off + i2 * src1_nb2 + i3 * src1_nb3;

    float acc = 0.0f;
    for (uint k = gtid.x; k < K; k += TG) {
        const half w = load_f16h(src0_buf, row_base + k * 2u);
        const float a = asfloat(src1_buf.Load(vec_base + k * 4u));
        acc += (float)w * a;
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
