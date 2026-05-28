RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint count;
    uint src_off;
    uint dst_off;
    uint dst_ne0;
    uint dst_ne1;
    uint dst_ne2;
    uint dst_ne3;
    uint src_ne0;
    uint src_ne1;
    uint src_ne2;
    uint src_ne3;
    uint src_nb0;
    uint src_nb1;
    uint src_nb2;
    uint src_nb3;
    uint lp0;
    uint lp1;
    uint lp2;
    uint lp3;
};

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx >= count) {
        return;
    }

    uint t = idx;
    uint i0 = t % dst_ne0;
    t /= dst_ne0;
    uint i1 = t % dst_ne1;
    t /= dst_ne1;
    uint i2 = t % dst_ne2;
    uint i3 = t / dst_ne2;

    uint value = 0u;
    if (i0 >= lp0 && i0 < lp0 + src_ne0 &&
        i1 >= lp1 && i1 < lp1 + src_ne1 &&
        i2 >= lp2 && i2 < lp2 + src_ne2 &&
        i3 >= lp3 && i3 < lp3 + src_ne3) {
        uint s0 = i0 - lp0;
        uint s1 = i1 - lp1;
        uint s2 = i2 - lp2;
        uint s3 = i3 - lp3;
        uint src_addr = src_off + s0 * src_nb0 + s1 * src_nb1 + s2 * src_nb2 + s3 * src_nb3;
        value = src.Load(src_addr);
    }

    dst.Store(dst_off + idx * 4u, value);
}
