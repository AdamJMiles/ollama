#define TG_SIZE 256

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint ne0;
    uint src_off;
    uint dst_off;
    uint param0;
    uint param1;
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
    uint row = gid.x;
    uint tid = gtid.x;
    uint row_off_in = src_off + row * ne0 * 4u;
    uint row_off_out = dst_off + row * ne0 * 4u;
    float eps = asfloat(param0);

    float local = 0.0f;
    for (uint i = tid; i < ne0; i += TG_SIZE) {
        float v = asfloat(src.Load(row_off_in + i * 4u));
        local += v * v;
    }

    float mean2 = wave_reduce_sum(local, tid) / (float) ne0;
    float scale = rsqrt(mean2 + eps);

    for (uint i = tid; i < ne0; i += TG_SIZE) {
        float v = asfloat(src.Load(row_off_in + i * 4u));
        dst.Store(row_off_out + i * 4u, asuint(v * scale));
    }
}
