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

[numthreads(TG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint tid = gtid.x;
    uint row_off_in = src_off + row * ne0 * 4u;
    uint row_off_out = dst_off + row * ne0 * 4u;
    float eps = asfloat(param0);

    float local = 0.0f;
    for (uint i = tid; i < ne0; i += TG_SIZE) {
        local += asfloat(src.Load(row_off_in + i * 4u));
    }
    sdata[tid] = local;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = TG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float mean = sdata[0] / (float) ne0;
    GroupMemoryBarrierWithGroupSync();

    local = 0.0f;
    for (uint i = tid; i < ne0; i += TG_SIZE) {
        float v = asfloat(src.Load(row_off_in + i * 4u)) - mean;
        local += v * v;
    }
    sdata[tid] = local;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = TG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float variance = sdata[0] / (float) ne0;
    float scale = rsqrt(variance + eps);

    for (uint i = tid; i < ne0; i += TG_SIZE) {
        float v = asfloat(src.Load(row_off_in + i * 4u));
        dst.Store(row_off_out + i * 4u, asuint((v - mean) * scale));
    }
}
