RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer idx_buf : register(u1);
RWByteAddressBuffer dst : register(u2);

cbuffer Params : register(b0) {
    uint count;
    uint src_off;
    uint idx_off;
    uint dst_off;
    uint src_ne0;
    uint src_ne1;
    uint idx_ne0;
    uint idx_ne1;
    uint idx_ne2;
    uint src_nb0;
    uint src_nb1;
    uint src_nb2;
    uint src_nb3;
    uint idx_nb0;
    uint idx_nb1;
    uint idx_nb2;
};

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint out_idx = dtid.x;
    if (out_idx >= count) {
        return;
    }

    uint elem = out_idx % src_ne0;
    uint row_linear = out_idx / src_ne0;
    uint idx_i0 = row_linear % idx_ne0;
    row_linear /= idx_ne0;
    uint idx_i1 = row_linear % idx_ne1;
    uint idx_i2 = row_linear / idx_ne1;

    uint idx_addr = idx_off + idx_i0 * idx_nb0 + idx_i1 * idx_nb1 + idx_i2 * idx_nb2;
    int row = asint(idx_buf.Load(idx_addr));

    uint value = 0u;
    if (row >= 0 && (uint) row < src_ne1) {
        uint src_addr = src_off + elem * src_nb0 + (uint) row * src_nb1 + idx_i1 * src_nb2 + idx_i2 * src_nb3;
        value = src.Load(src_addr);
    }

    dst.Store(dst_off + out_idx * 4u, value);
}
