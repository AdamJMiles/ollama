RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint count;
    uint ne0; uint ne1; uint ne2; uint ne3;
    uint s1_ne0; uint s1_ne1; uint s1_ne2; uint s1_ne3;
    uint s1_nb0; uint s1_nb1; uint s1_nb2; uint s1_nb3;
    uint src0_off;
    uint src1_off;
    uint dst_off;
};

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx >= count) return;

    uint i0 = idx % ne0;
    uint t  = idx / ne0;
    uint i1 = t % ne1;
    uint t2 = t / ne1;
    uint i2 = t2 % ne2;
    uint i3 = t2 / ne2;

    uint b0 = i0 % s1_ne0;
    uint b1 = i1 % s1_ne1;
    uint b2 = i2 % s1_ne2;
    uint b3 = i3 % s1_ne3;
    uint b_addr = src1_off + b0 * s1_nb0 + b1 * s1_nb1 + b2 * s1_nb2 + b3 * s1_nb3;

    float a = asfloat(src0.Load(src0_off + idx * 4u));
    float b = asfloat(src1.Load(b_addr));
    dst.Store(dst_off + idx * 4u, asuint(a / b));
}
