#define TG_SIZE 64

RWByteAddressBuffer src_buf  : register(u0);
RWByteAddressBuffer pos_buf  : register(u1);
RWByteAddressBuffer freq_buf : register(u2);
RWByteAddressBuffer dst_buf  : register(u3);

cbuffer Params : register(b0) {
    uint ne0;
    uint ne1;
    uint ne2;
    uint ne3;
    uint n_dims;
    uint src_off;
    uint dst_off;
    uint pos_off;
    uint freq_off;
    uint freq_base_bits;
    uint freq_scale_bits;
    uint flags;
};

[numthreads(TG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint tid = gtid.x;
    const uint head = gid.x;
    const uint token = gid.y;
    const uint batch = gid.z;
    if (head >= ne1 || token >= ne2 || batch >= ne3) {
        return;
    }

    const uint row = ((batch * ne2 + token) * ne1 + head) * ne0;
    const uint src_row = src_off + row * 4u;
    const uint dst_row = dst_off + row * 4u;
    const int pos = asint(pos_buf.Load(pos_off + token * 4u));
    const float freq_base = asfloat(freq_base_bits);
    const float freq_scale = asfloat(freq_scale_bits);
    const bool has_freq_factors = (flags & 1u) != 0u;
    const uint n_pairs = n_dims / 2u;

    for (uint j = tid; j < n_pairs; j += TG_SIZE) {
        const uint i0 = 2u * j;
        float theta = (float) pos * freq_scale * pow(freq_base, -2.0f * (float) j / (float) n_dims);
        if (has_freq_factors) {
            theta /= asfloat(freq_buf.Load(freq_off + j * 4u));
        }

        const float cos_th = cos(theta);
        const float sin_th = sin(theta);
        const float x0 = asfloat(src_buf.Load(src_row + (i0 + 0u) * 4u));
        const float x1 = asfloat(src_buf.Load(src_row + (i0 + 1u) * 4u));

        dst_buf.Store(dst_row + (i0 + 0u) * 4u, asuint(x0 * cos_th - x1 * sin_th));
        dst_buf.Store(dst_row + (i0 + 1u) * 4u, asuint(x0 * sin_th + x1 * cos_th));
    }

    for (uint i = n_dims + tid; i < ne0; i += TG_SIZE) {
        dst_buf.Store(dst_row + i * 4u, src_buf.Load(src_row + i * 4u));
    }
}
