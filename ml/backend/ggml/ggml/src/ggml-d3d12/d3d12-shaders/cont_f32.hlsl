RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint count;
    uint src_off;
    uint dst_off;
    uint ne0;
    uint ne1;
    uint ne2;
    uint ne3;
    uint src_nb0;
    uint src_nb1;
    uint src_nb2;
    uint src_nb3;
};

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx >= count) {
        return;
    }

    uint t = idx;
    uint i0 = t % ne0;
    t /= ne0;
    uint i1 = t % ne1;
    t /= ne1;
    uint i2 = t % ne2;
    uint i3 = t / ne2;

    uint src_addr = src_off + i0 * src_nb0 + i1 * src_nb1 + i2 * src_nb2 + i3 * src_nb3;
    uint v = src.Load(src_addr);
    dst.Store(dst_off + idx * 4u, v);
}
