RWByteAddressBuffer src0_buf : register(u0);
RWByteAddressBuffer src1_buf : register(u1);
RWByteAddressBuffer dst_buf  : register(u2);

cbuffer Params : register(b0) {
    uint count;
    uint src0_off;
    uint src1_off;
    uint dst_off;
    uint d_conv;
    uint ncs;
    uint d_inner;
    uint n_t;
};

[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    const uint idx = dtid.x;
    if (idx >= count) {
        return;
    }

    const uint c = idx % d_inner;
    const uint tmp = idx / d_inner;
    const uint t = tmp % n_t;
    const uint seq = tmp / n_t;

    const uint src_base = src0_off + ((seq * d_inner + c) * ncs + t) * 4u;
    const uint w_base = src1_off + (c * d_conv) * 4u;

    float sum = 0.0f;
    for (uint k = 0; k < d_conv; ++k) {
        sum += asfloat(src0_buf.Load(src_base + k * 4u)) * asfloat(src1_buf.Load(w_base + k * 4u));
    }

    dst_buf.Store(dst_off + idx * 4u, asuint(sum));
}
