#define TM 32
#define TN 32
#define TK 16

RWByteAddressBuffer src0_buf : register(u0);
RWByteAddressBuffer src1_buf : register(u1);
RWByteAddressBuffer dst_buf  : register(u2);

cbuffer Params : register(b0) {
    uint M;
    uint N;
    uint K;
    uint batch_ne2;
    uint src0_off;
    uint src1_off;
    uint dst_off;
    uint src0_nb1;
    uint src0_nb2;
    uint src0_nb3;
};

groupshared float sa[TM][TK];
groupshared float sb[TK][TN];

[numthreads(TN, TM, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint n = gid.x * TN + gtid.x;
    const uint m = gid.y * TM + gtid.y;
    const uint batch = gid.z;
    const uint b2 = batch % batch_ne2;
    const uint b3 = batch / batch_ne2;
    const uint src0_base = src0_off + b2 * src0_nb2 + b3 * src0_nb3;
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
            sa[lm][lk] = (gm < M && gk < K) ? asfloat(src0_buf.Load(src0_base + gm * src0_nb1 + gk * 4u)) : 0.0f;
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
