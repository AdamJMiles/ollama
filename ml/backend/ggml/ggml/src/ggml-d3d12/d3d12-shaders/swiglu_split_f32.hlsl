// SWIGLU split: src0 is x (activation input), src1 is g (gate)
// output: y_i = silu(x_i) * g_i
RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint count;
    uint src0_off;
    uint src1_off;
    uint dst_off;
};


float activate(float x) {
    return x / (1.0 + exp(-x));
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