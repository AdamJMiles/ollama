#define TM 32
#define TN 32
#define TK 16   // 32 Q8_0 elements span 1 block; with TK=16 each k0 iter
                // covers half a block per thread tile (q8_0 sub-block load)

#define Q8_QK 32u
#define Q8_BLOCK_SIZE 34u   // 2 bytes (f16 scale) + 32 bytes (int8)
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
    uint src0_nb1;   // Q8_0 row stride in bytes
    uint src0_nb2;
    uint src0_nb3;
};

groupshared float sa[TM][TK];
groupshared float sb[TK][TN];

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
    const uint block    = k / Q8_QK;
    const uint elem     = k & (Q8_QK - 1u);
    const uint block_off = row_base + block * Q8_BLOCK_SIZE;
    const float scale   = load_f16(src0_buf, block_off);
    const uint  byte_q  = load_u8(src0_buf, block_off + Q8_QS_OFFSET + elem);
    const int   q       = (byte_q < 128u) ? int(byte_q) : (int(byte_q) - 256);
    return float(q) * scale;
}

[numthreads(TN, TM, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint n = gid.x * TN + gtid.x;
    const uint m = gid.y * TM + gtid.y;
    const uint batch = gid.z;
    const uint b2 = batch % batch_ne2;
    const uint b3 = batch / batch_ne2;
    const uint s0b2 = (broadcast2 == 0u) ? b2 : (b2 / broadcast2);
    const uint s0b3 = (broadcast3 == 0u) ? b3 : (b3 / broadcast3);
    const uint src0_base = src0_off + s0b2 * src0_nb2 + s0b3 * src0_nb3;
    const uint src1_base = src1_off + batch * N * K * 4u;
    const uint dst_base = dst_off + batch * N * M * 4u;
    const uint tid = gtid.y * TN + gtid.x;

    float acc = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += TK) {
        for (uint i = tid; i < TM * TK; i += TM * TN) {
            const uint lm = i / TK;
            const uint lk = i - lm * TK;
            const uint gm = gid.y * TM + lm;
            const uint gk = k0 + lk;
            sa[lm][lk] = (gm < M && gk < K)
                ? load_q8_0(src0_base + gm * src0_nb1, gk)
                : 0.0f;
        }
        for (uint j = tid; j < TK * TN; j += TM * TN) {
            const uint lk = j / TN;
            const uint ln = j - lk * TN;
            const uint gk = k0 + lk;
            const uint gn = gid.x * TN + ln;
            sb[lk][ln] = (gk < K && gn < N) ? asfloat(src1_buf.Load(src1_base + gn * K * 4u + gk * 4u)) : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < TK; ++k) {
            acc += sa[gtid.y][k] * sb[k][gtid.x];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (m < M && n < N) {
        dst_buf.Store(dst_base + n * M * 4u + m * 4u, asuint(acc));
    }
}
