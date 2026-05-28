RWByteAddressBuffer src_buf : register(u0);
RWByteAddressBuffer dst_buf : register(u1);

cbuffer Params : register(b0) {
    uint total_elems;
    uint src_off;
    uint dst_off;
    uint iw;
    uint ih;
    uint ow;
    uint oh;
    uint channels;
    uint k0;
    uint k1;
    int s0;
    int s1;
    int p0;
    int p1;
};

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    const uint idx = dtid.x;
    if (idx >= total_elems) {
        return;
    }

    const uint plane = ow * oh;
    const uint ox = idx % ow;
    const uint oy = (idx / ow) % oh;
    const uint c = (idx / plane) % channels;
    const uint n = idx / (plane * channels);

    const int ix0 = int(ox) * s0 - p0;
    const int iy0 = int(oy) * s1 - p1;
    float sum = 0.0f;

    for (uint ky = 0; ky < k1; ++ky) {
        const int sy = iy0 + int(ky);
        if (sy < 0 || sy >= int(ih)) {
            continue;
        }
        for (uint kx = 0; kx < k0; ++kx) {
            const int sx = ix0 + int(kx);
            if (sx < 0 || sx >= int(iw)) {
                continue;
            }
            const uint src_idx = ((n * channels + c) * ih + uint(sy)) * iw + uint(sx);
            sum += asfloat(src_buf.Load(src_off + src_idx * 4u));
        }
    }

    dst_buf.Store(dst_off + idx * 4u, asuint(sum / float(k0 * k1)));
}
