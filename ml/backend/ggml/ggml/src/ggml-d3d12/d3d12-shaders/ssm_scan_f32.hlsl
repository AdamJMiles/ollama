#define TG_SIZE 256

RWByteAddressBuffer s0_buf  : register(u0);
RWByteAddressBuffer x_buf   : register(u1);
RWByteAddressBuffer dt_buf  : register(u2);
RWByteAddressBuffer A_buf   : register(u3);
RWByteAddressBuffer B_buf   : register(u4);
RWByteAddressBuffer C_buf   : register(u5);
RWByteAddressBuffer ids_buf : register(u6);
RWByteAddressBuffer dst_buf : register(u7);

cbuffer Params : register(b0) {
    uint s0_off;
    uint x_off;
    uint dt_off;
    uint A_off;
    uint B_off;
    uint C_off;
    uint ids_off;
    uint dst_off;
    uint d_state;
    uint head_dim;
    uint n_head;
    uint n_group;
    uint n_tok;
    uint n_seq;
    uint state_out_off;
};

groupshared float partial[TG_SIZE];

float softplus_f32(float x) {
    return x > 20.0f ? x : log(1.0f + exp(x));
}

[numthreads(TG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint i1 = gid.x;
    const uint h = gid.y;
    const uint seq = gid.z;
    const uint m = gtid.x;

    if (i1 >= head_dim || h >= n_head || seq >= n_seq) {
        return;
    }

    const int seq_in_i = asint(ids_buf.Load(ids_off + seq * 4u));
    if (seq_in_i < 0) {
        return;
    }
    const uint seq_in = (uint) seq_in_i;

    const uint state_plane = d_state * head_dim * n_head;
    const uint state_head_base = h * head_dim * d_state + i1 * d_state;
    const uint s0_base = seq_in * state_plane + state_head_base;
    const uint sout_base = seq * state_plane + state_head_base;

    const uint x_stride_t = head_dim * n_head;
    const uint x_base = seq * n_tok * x_stride_t + h * head_dim + i1;
    const uint dt_base = seq * n_tok * n_head + h;
    const uint y_base = x_base;

    const uint heads_per_group = n_head / n_group;
    const uint g = h / heads_per_group;
    const uint bc_stride_t = d_state * n_group;
    const uint bc_base = seq * n_tok * bc_stride_t + g * d_state;

    float state = 0.0f;
    if (m < d_state) {
        state = asfloat(s0_buf.Load(s0_off + (s0_base + m) * 4u));
    }

    for (uint t = 0; t < n_tok; ++t) {
        const float dtv = asfloat(dt_buf.Load(dt_off + (dt_base + t * n_head) * 4u));
        const float dtsp = softplus_f32(dtv);
        const float dA = exp(dtsp * asfloat(A_buf.Load(A_off + h * 4u)));
        const float x_dt = asfloat(x_buf.Load(x_off + (x_base + t * x_stride_t) * 4u)) * dtsp;

        float local = 0.0f;
        if (m < d_state) {
            const uint bc_idx = bc_base + t * bc_stride_t + m;
            state = state * dA + asfloat(B_buf.Load(B_off + bc_idx * 4u)) * x_dt;
            local = state * asfloat(C_buf.Load(C_off + bc_idx * 4u));
        }

        partial[m] = local;
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint stride = TG_SIZE / 2u; stride > 0u; stride >>= 1u) {
            if (m < stride) {
                partial[m] += partial[m + stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (m == 0u) {
            dst_buf.Store(dst_off + (y_base + t * x_stride_t) * 4u, asuint(partial[0]));
        }
    }

    if (m < d_state) {
        dst_buf.Store(dst_off + state_out_off + (sout_base + m) * 4u, asuint(state));
    }
}
