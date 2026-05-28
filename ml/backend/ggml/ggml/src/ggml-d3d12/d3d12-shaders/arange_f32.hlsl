RWByteAddressBuffer dst : register(u0);

cbuffer Params : register(b0) {
    uint count;
    uint dst_off;
    uint start_bits;
    uint step_bits;
};

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint i = dtid.x;
    if (i >= count) {
        return;
    }
    float y = asfloat(start_bits) + asfloat(step_bits) * (float)i;
    dst.Store(dst_off + i * 4, asuint(y));
}
