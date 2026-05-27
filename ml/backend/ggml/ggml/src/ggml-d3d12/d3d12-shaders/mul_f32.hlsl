// same-shape F32 elementwise multiply
RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint count;
    uint src0_off;
    uint src1_off;
    uint dst_off;
};

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint i = dtid.x;
    if (i >= count) {
        return;
    }

    float a = asfloat(src0.Load(src0_off + i * 4));
    float b = asfloat(src1.Load(src1_off + i * 4));
    float y = a * b;
    dst.Store(dst_off + i * 4, asuint(y));
}
