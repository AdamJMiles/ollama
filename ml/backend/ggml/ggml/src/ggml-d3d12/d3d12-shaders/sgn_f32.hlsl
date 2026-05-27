RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint count;
    uint src_off;
    uint dst_off;
    uint param0;
};

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint i = dtid.x;
    if (i >= count) {
        return;
    }
    float x = asfloat(src.Load(src_off + i * 4));
    float y = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
    dst.Store(dst_off + i * 4, asuint(y));
}