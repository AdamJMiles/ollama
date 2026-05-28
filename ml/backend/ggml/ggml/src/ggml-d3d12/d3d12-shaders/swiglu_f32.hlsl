// SWIGLU non-split: input row [x_0..x_{n_out-1}, g_0..g_{n_out-1}] (swap if flags bit 0 set)
// output: y_i = silu(x_i) * g_i
RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint count_out;
    uint n_out;
    uint src_off;
    uint dst_off;
    uint flags;
};


float activate(float x) {
    return x / (1.0 + exp(-x));
}

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx >= count_out) {
        return;
    }

    uint row = idx / n_out;
    uint col = idx - row * n_out;
    uint n_in = n_out * 2;
    uint row_off = src_off + row * n_in * 4;

    uint x_off;
    uint g_off;
    if ((flags & 1u) != 0u) {
        g_off = row_off + col * 4;
        x_off = row_off + (col + n_out) * 4;
    } else {
        x_off = row_off + col * 4;
        g_off = row_off + (col + n_out) * 4;
    }

    float x = asfloat(src.Load(x_off));
    float g = asfloat(src.Load(g_off));
    float y = activate(x) * g;
    dst.Store(dst_off + idx * 4, asuint(y));
}