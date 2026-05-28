#define TG_SIZE 256

RWByteAddressBuffer src_buf  : register(u0);
RWByteAddressBuffer mask_buf : register(u1);
RWByteAddressBuffer dst_buf  : register(u2);

cbuffer Params : register(b0) {
    uint ne0;
    uint ne1;
    uint src_ne2;
    uint mask_ne2;
    uint mask_ne3;
    uint src_off;
    uint dst_off;
    uint mask_off;
    uint mask_stride_row;
    uint scale_bits;
    uint flags;
    uint _pad;
};

groupshared float reduce_data[TG_SIZE];

float wave_reduce_sum(float local, uint tid) {
    const uint lane = WaveGetLaneIndex();
    const uint wave_size = WaveGetLaneCount();
    const uint wave = tid / wave_size;
    const uint num_waves = (TG_SIZE + wave_size - 1u) / wave_size;

    local = WaveActiveSum(local);
    if (lane == 0u) {
        reduce_data[wave] = local;
    }
    GroupMemoryBarrierWithGroupSync();

    if (wave == 0u) {
        float total = 0.0f;
        for (uint i = lane; i < num_waves; i += wave_size) {
            total += reduce_data[i];
        }
        total = WaveActiveSum(total);
        if (lane == 0u) {
            reduce_data[0] = total;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return reduce_data[0];
}

float wave_reduce_max(float local, uint tid) {
    const uint lane = WaveGetLaneIndex();
    const uint wave_size = WaveGetLaneCount();
    const uint wave = tid / wave_size;
    const uint num_waves = (TG_SIZE + wave_size - 1u) / wave_size;

    local = WaveActiveMax(local);
    if (lane == 0u) {
        reduce_data[wave] = local;
    }
    GroupMemoryBarrierWithGroupSync();

    if (wave == 0u) {
        float total = -3.402823466e+38f;
        for (uint i = lane; i < num_waves; i += wave_size) {
            total = max(total, reduce_data[i]);
        }
        total = WaveActiveMax(total);
        if (lane == 0u) {
            reduce_data[0] = total;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return reduce_data[0];
}

[numthreads(TG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint row = gid.x;
    const uint tid = gtid.x;
    const float scale = asfloat(scale_bits);
    const bool has_mask = (flags & 1u) != 0u;

    const uint row_off_in = src_off + row * ne0 * 4u;
    const uint row_off_out = dst_off + row * ne0 * 4u;

    const uint i01 = (ne1 == 0u) ? 0u : (row % ne1);
    const uint t02 = (ne1 == 0u) ? 0u : (row / ne1);
    const uint i02 = (src_ne2 == 0u) ? 0u : (t02 % src_ne2);
    const uint i03 = (src_ne2 == 0u) ? 0u : (t02 / src_ne2);
    const uint mi02 = (mask_ne2 == 0u) ? 0u : (i02 % mask_ne2);
    const uint mi03 = (mask_ne3 == 0u) ? 0u : (i03 % mask_ne3);
    const uint mask_row = i01 + mi02 * ne1 + mi03 * ne1 * mask_ne2;
    const uint row_off_mask = mask_off + mask_row * mask_stride_row;

    float local_max = -3.402823466e+38f;
    for (uint i = tid; i < ne0; i += TG_SIZE) {
        float v = asfloat(src_buf.Load(row_off_in + i * 4u)) * scale;
        if (has_mask) {
            v += asfloat(mask_buf.Load(row_off_mask + i * 4u));
        }
        local_max = max(local_max, v);
    }
    const float row_max = wave_reduce_max(local_max, tid);

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
    const float inv_sum = 1.0f / wave_reduce_sum(local_sum, tid);

    for (uint i = tid; i < ne0; i += TG_SIZE) {
        const float e = asfloat(dst_buf.Load(row_off_out + i * 4u));
        dst_buf.Store(row_off_out + i * 4u, asuint(e * inv_sum));
    }
}
