#define TG_SIZE 64
#define MAX_KV 4096u
#define MAX_D 256u

RWByteAddressBuffer q_buf    : register(u0);
RWByteAddressBuffer k_buf    : register(u1);
RWByteAddressBuffer v_buf    : register(u2);
RWByteAddressBuffer mask_buf : register(u3);
RWByteAddressBuffer dst_buf  : register(u4);

cbuffer Params : register(b0) {
    uint D;
    uint nq;
    uint nkv;
    uint nh;
    uint nh_kv;
    uint q_off;
    uint k_off;
    uint v_off;
    uint mask_off;
    uint dst_off;
    uint q_nb1;
    uint q_nb2;
    uint k_nb1;
    uint k_nb2;
    uint v_nb1;
    uint v_nb2;
    uint dst_nb1;
    uint dst_nb2;
    uint mask_nb1;
    uint flags;
    uint scale_bits;
};

groupshared float s_buf[MAX_KV];
groupshared float q_cache[MAX_D];
groupshared float reduce_max[TG_SIZE];
groupshared float reduce_sum[TG_SIZE];

float q_load(uint d, uint i, uint h) {
    return asfloat(q_buf.Load(q_off + i * q_nb1 + h * q_nb2 + d * 4u));
}

float k_load(uint d, uint j, uint h_kv) {
    return asfloat(k_buf.Load(k_off + j * k_nb1 + h_kv * k_nb2 + d * 4u));
}

float v_load(uint d, uint j, uint h_kv) {
    return asfloat(v_buf.Load(v_off + j * v_nb1 + h_kv * v_nb2 + d * 4u));
}

uint load_u8_mask(RWByteAddressBuffer buf, uint off) {
    const uint word = buf.Load(off & ~3u);
    return (word >> ((off & 3u) * 8u)) & 0xFFu;
}

float mask_load(uint j, uint i) {
    const bool is_f16 = (flags & 2u) != 0u;
    if (is_f16) {
        const uint addr = mask_off + i * mask_nb1 + j * 2u;
        const uint half_bits = load_u8_mask(mask_buf, addr) | (load_u8_mask(mask_buf, addr + 1u) << 8u);
        return f16tof32(half_bits);
    }
    return asfloat(mask_buf.Load(mask_off + i * mask_nb1 + j * 4u));
}

void dst_store(uint d, uint h, uint i, float value) {
    dst_buf.Store(dst_off + h * dst_nb1 + i * dst_nb2 + d * 4u, asuint(value));
}

[numthreads(TG_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint h = gid.x;
    const uint i = gid.y;
    const uint tid = gtid.x;
    if (h >= nh || i >= nq) return;

    const uint h_kv = h * nh_kv / nh;
    const float scale = asfloat(scale_bits);
    const bool has_mask = (flags & 1u) != 0u;

    // Cache q[d, i, h] for d in [0, D) so the per-j inner loop reads q from
    // groupshared instead of UAV.
    for (uint d = tid; d < D; d += TG_SIZE) {
        q_cache[d] = q_load(d, i, h);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint j = tid; j < nkv; j += TG_SIZE) {
        float dot = 0.0f;
        for (uint d = 0; d < D; ++d) {
            dot += q_cache[d] * k_load(d, j, h_kv);
        }
        float s = dot * scale;
        if (has_mask) {
            s += mask_load(j, i);
        }
        s_buf[j] = s;
    }
    GroupMemoryBarrierWithGroupSync();

    float local_max = -3.402823466e+38f;
    for (uint j = tid; j < nkv; j += TG_SIZE) {
        local_max = max(local_max, s_buf[j]);
    }
    reduce_max[tid] = local_max;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = TG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            reduce_max[tid] = max(reduce_max[tid], reduce_max[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float row_max = reduce_max[0];

    float local_sum = 0.0f;
    for (uint j = tid; j < nkv; j += TG_SIZE) {
        const float e = exp(s_buf[j] - row_max);
        s_buf[j] = e;
        local_sum += e;
    }
    reduce_sum[tid] = local_sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride2 = TG_SIZE / 2u; stride2 > 0u; stride2 >>= 1u) {
        if (tid < stride2) {
            reduce_sum[tid] += reduce_sum[tid + stride2];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float inv_sum = reduce_sum[0] > 0.0f ? 1.0f / reduce_sum[0] : 0.0f;

    for (uint j = tid; j < nkv; j += TG_SIZE) {
        s_buf[j] *= inv_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint d = tid; d < D; d += TG_SIZE) {
        float acc = 0.0f;
        for (uint j = 0; j < nkv; ++j) {
            acc += s_buf[j] * v_load(d, j, h_kv);
        }
        dst_store(d, h, i, acc);
    }
}
