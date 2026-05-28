RWByteAddressBuffer src_buf : register(u0);
RWByteAddressBuffer dst_buf : register(u1);

cbuffer Params : register(b0) {
    uint total_elems;
    uint src_off;
    uint dst_off;
    uint flags;
    uint iw;
    uint ih;
    uint ic;
    uint kw;
    uint kh;
    uint ow;
    uint oh;
    uint k_total;
    int s0;
    int s1;
    int p0;
    int p1;
    int d0;
    int d1;
};

float im2col_value(uint idx) {
    const uint ik = idx % k_total;
    const uint oj = idx / k_total;
    const uint ox = oj % ow;
    const uint oy = (oj / ow) % oh;
    const uint n = oj / (ow * oh);

    const uint ikw = ik % kw;
    const uint ikh = (ik / kw) % kh;
    const uint iic = ik / (kw * kh);
    const int sx = int(ox) * s0 - p0 + int(ikw) * d0;
    const int sy = int(oy) * s1 - p1 + int(ikh) * d1;
    if (sx < 0 || sy < 0 || sx >= int(iw) || sy >= int(ih)) {
        return 0.0f;
    }

    const uint src_idx = ((n * ic + iic) * ih + uint(sy)) * iw + uint(sx);
    return asfloat(src_buf.Load(src_off + src_idx * 4u));
}

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    const bool dst_f16 = (flags & 1u) != 0u;
    const uint idx = dst_f16 ? dtid.x * 2u : dtid.x;
    if (idx >= total_elems) {
        return;
    }

    const float v0 = im2col_value(idx);
    if (!dst_f16) {
        dst_buf.Store(dst_off + idx * 4u, asuint(v0));
        return;
    }

    uint packed = f32tof16(v0) & 0xFFFFu;
    if (idx + 1u < total_elems) {
        packed |= (f32tof16(im2col_value(idx + 1u)) & 0xFFFFu) << 16;
    } else {
        packed |= dst_buf.Load(dst_off + (idx >> 1u) * 4u) & 0xFFFF0000u;
    }
    dst_buf.Store(dst_off + (idx >> 1u) * 4u, packed);
}
