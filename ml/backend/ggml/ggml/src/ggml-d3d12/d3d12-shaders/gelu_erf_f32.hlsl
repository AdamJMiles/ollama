RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint count;
    uint src_off;
    uint dst_off;
    uint param0;
};

float erff_approx(float x) {
    float sign = (x < 0.0) ? -1.0 : 1.0;
    float ax = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * ax);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t)
        + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    return sign * y;
}

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint i = dtid.x;
    if (i >= count) {
        return;
    }
    float x = asfloat(src.Load(src_off + i * 4));
    float y = 0.5 * x * (1.0 + erff_approx(x * 0.70710678118654752440084436210484));
    dst.Store(dst_off + i * 4, asuint(y));
}