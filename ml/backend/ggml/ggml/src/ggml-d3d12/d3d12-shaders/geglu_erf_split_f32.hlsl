// GEGLU_ERF split: src0 is x (activation input), src1 is g (gate)
// output: y_i = gelu_erf(x_i) * g_i
RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint count;
    uint src0_off;
    uint src1_off;
    uint dst_off;
};

float erff_approx(float x) {
    float sign = (x < 0.0) ? -1.0 : 1.0;
    float ax = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * ax);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t)
        + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    return sign * y;
}
float activate(float x) {
    return 0.5 * x * (1.0 + erff_approx(x * 0.70710678118654752440084436210484));
}

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint i = dtid.x;
    if (i >= count) {
        return;
    }

    float x = asfloat(src0.Load(src0_off + i * 4));
    float g = asfloat(src1.Load(src1_off + i * 4));
    float y = activate(x) * g;
    dst.Store(dst_off + i * 4, asuint(y));
}