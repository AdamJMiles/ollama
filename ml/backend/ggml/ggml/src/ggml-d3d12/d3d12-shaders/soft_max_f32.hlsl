#define TG_SIZE 256

RWByteAddressBuffer src_buf  : register(u0);
RWByteAddressBuffer mask_buf : register(u1);
RWByteAddressBuffer dst_buf  : register(u2);

cbuffer Params : register(b0) {
    uint ne0;
    uint ne1;
    uint src_off;
    uint dst_off;
    uint mask_off;
    uint mask_stride_row;
    uint scale_bits;
    uint flags;
};

groupshared float reduce_max[TG_SIZE];
groupshared float reduce_sum[TG_SIZE];

[numthreads(TG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint row = gid.x;
    const uint tid = gtid.x;
    const float scale = asfloat(scale_bits);
    const bool has_mask = (flags & 1u) != 0u;

    const uint row_off_in = src_off + row * ne0 * 4u;
    const uint row_off_out = dst_off + row * ne0 * 4u;
    const uint row_off_mask = mask_off + row * mask_stride_row;

    float local_max = -3.402823466e+38f;
    for (uint i = tid; i < ne0; i += TG_SIZE) {
        float v = asfloat(src_buf.Load(row_off_in + i * 4u)) * scale;
        if (has_mask) {
            v += asfloat(mask_buf.Load(row_off_mask + i * 4u));
        }
        local_max = max(local_max, v);
    }

    reduce_max[tid] = local_max;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = TG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            reduce_max[tid] = max(reduce_max[tid], reduce_max[tid + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float row_max = reduce_max[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < ne0; i += TG_SIZE) {
        float v = asfloat(src_buf.Load(row_off_in + i * 4u)) * scale;
        if (has_mask) {
            v += asfloat(mask_buf.Load(row_off_mask + i * 4u));
        }
        const float e = exp(v - row_max);
        dst_buf.Store(row_off_out + i * 4u, asuint(e));
        local_sum += e;
    }

    reduce_sum[tid] = local_sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = TG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            reduce_sum[tid] += reduce_sum[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float inv_sum = 1.0f / reduce_sum[0];

    for (uint i = tid; i < ne0; i += TG_SIZE) {
        const float e = asfloat(dst_buf.Load(row_off_out + i * 4u));
        dst_buf.Store(row_off_out + i * 4u, asuint(e * inv_sum));
    }
}
