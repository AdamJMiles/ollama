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

    svals[tid] = best_val;
    sidx[tid] = best_idx;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = TG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            float other_val = svals[tid + s];
            uint other_idx = sidx[tid + s];
            if (other_val > svals[tid] || (other_val == svals[tid] && other_idx > sidx[tid])) {
                svals[tid] = other_val;
                sidx[tid] = other_idx;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u) {
        dst.Store(dst_off + row * 4u, sidx[0]);
    }
}
