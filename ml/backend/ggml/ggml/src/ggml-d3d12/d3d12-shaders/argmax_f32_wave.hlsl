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

groupshared float svals[TG_SIZE];
groupshared uint sidx[TG_SIZE];

void wave_reduce_argmax(inout float best_val, inout uint best_idx, uint tid) {
    const uint lane = WaveGetLaneIndex();
    const uint wave_size = WaveGetLaneCount();
    const uint wave = tid / wave_size;
    const uint num_waves = (TG_SIZE + wave_size - 1u) / wave_size;

    float wave_val = WaveActiveMax(best_val);
    uint wave_idx = WaveActiveMax(best_val == wave_val ? best_idx : 0u);
    if (lane == 0u) {
        svals[wave] = wave_val;
        sidx[wave] = wave_idx;
    }
    GroupMemoryBarrierWithGroupSync();

    if (wave == 0u) {
        float local_val = asfloat(0xff800000u);
        uint local_idx = 0u;
        for (uint i = lane; i < num_waves; i += wave_size) {
            float other_val = svals[i];
            uint other_idx = sidx[i];
            if (other_val > local_val || (other_val == local_val && other_idx > local_idx)) {
                local_val = other_val;
                local_idx = other_idx;
            }
        }
        float final_val = WaveActiveMax(local_val);
        uint final_idx = WaveActiveMax(local_val == final_val ? local_idx : 0u);
        if (lane == 0u) {
            svals[0] = final_val;
            sidx[0] = final_idx;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    best_val = svals[0];
    best_idx = sidx[0];
}

[numthreads(TG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint tid = gtid.x;
    uint row_off_in = src_off + row * ne0 * 4u;

    float best_val = asfloat(0xff800000u);
    uint best_idx = 0u;
    for (uint i = tid; i < ne0; i += TG_SIZE) {
        float v = asfloat(src.Load(row_off_in + i * 4u));
        if (v > best_val || (v == best_val && i > best_idx)) {
            best_val = v;
            best_idx = i;
        }
    }

    wave_reduce_argmax(best_val, best_idx, tid);
    if (tid == 0u) {
        dst.Store(dst_off + row * 4u, best_idx);
    }
}
