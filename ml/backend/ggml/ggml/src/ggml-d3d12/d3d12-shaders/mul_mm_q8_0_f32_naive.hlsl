// Minimal correctness reference for Q8_0 mul_mat (slow). One thread per
// output dst element. Used to validate the dequant + index math is right
// before optimising with a tiled shared-memory kernel.

#define Q8_QK 32u
#define Q8_BLOCK_SIZE 34u
#define Q8_QS_OFFSET 2u

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
    const uint block     = k / Q8_QK;
    const uint elem      = k & (Q8_QK - 1u);
    const uint block_off = row_base + block * Q8_BLOCK_SIZE;
    const float scale    = load_f16(src0_buf, block_off);
    const uint  byte_q   = load_u8(src0_buf, block_off + Q8_QS_OFFSET + elem);
    const int   q        = (int(byte_q) << 24) >> 24;
    return float(q) * scale;
}

[numthreads(32, 32, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint n = gid.x * 32u + gtid.x;
    const uint m = gid.y * 32u + gtid.y;
    const uint batch = gid.z;
    if (m >= M || n >= N) return;

    const uint b2 = batch % batch_ne2;
    const uint b3 = batch / batch_ne2;
    const uint s0b2 = (broadcast2 == 0u) ? b2 : (b2 / broadcast2);
    const uint s0b3 = (broadcast3 == 0u) ? b3 : (b3 / broadcast3);

    const uint row_base = src0_off + s0b2 * src0_nb2 + s0b3 * src0_nb3 + m * src0_nb1;
    const uint src1_col = src1_off + batch * N * K * 4u + n * K * 4u;

    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        const float w = load_q8_0(row_base, k);
        const float a = asfloat(src1_buf.Load(src1_col + k * 4u));
        acc += w * a;
    }

    const uint dst_base = dst_off + batch * N * M * 4u;
    dst_buf.Store(dst_base + n * M * 4u + m * 4u, asuint(acc));
}
