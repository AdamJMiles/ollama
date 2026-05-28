#define TG_SIZE 256

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint group_size;
    uint src_off;
    uint dst_off;
    uint param0;
    uint sample_size;
};

groupshared float sdata[TG_SIZE];

float wave_reduce_sum(float local, uint tid) {
    const uint lane = WaveGetLaneIndex();
    const uint wave_size = WaveGetLaneCount();
    const uint wave = tid / wave_size;
    const uint num_waves = (TG_SIZE + wave_size - 1u) / wave_size;

    local = WaveActiveSum(local);
    if (lane == 0u) {
        sdata[wave] = local;
    }
    GroupMemoryBarrierWithGroupSync();

    if (wave == 0u) {
        float total = 0.0f;
        for (uint i = lane; i < num_waves; i += wave_size) {
            total += sdata[i];
        }
        total = WaveActiveSum(total);
        if (lane == 0u) {
            sdata[0] = total;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return sdata[0];
}

[numthreads(TG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint group = gid.x;
    uint sample = gid.y;
    uint tid = gtid.x;
    float eps = asfloat(param0);

    uint group_begin = group * group_size;
    if (group_begin >= sample_size) {
        return;
    }
    uint group_end = min(group_begin + group_size, sample_size);
    uint count = group_end - group_begin;
    uint base_elem = sample * sample_size + group_begin;
    uint base_in = src_off + base_elem * 4u;
    uint base_out = dst_off + base_elem * 4u;

    float local = 0.0f;
    for (uint i = tid; i < count; i += TG_SIZE) {
        local += asfloat(src.Load(base_in + i * 4u));
    }
    float mean = wave_reduce_sum(local, tid) / (float) count;

    local = 0.0f;
    for (uint i = tid; i < count; i += TG_SIZE) {
        float v = asfloat(src.Load(base_in + i * 4u)) - mean;
        local += v * v;
    }
    float variance = wave_reduce_sum(local, tid) / (float) count;
    float scale = rsqrt(variance + eps);

    for (uint i = tid; i < count; i += TG_SIZE) {
        float v = asfloat(src.Load(base_in + i * 4u));
        dst.Store(base_out + i * 4u, asuint((v - mean) * scale));
    }
}
