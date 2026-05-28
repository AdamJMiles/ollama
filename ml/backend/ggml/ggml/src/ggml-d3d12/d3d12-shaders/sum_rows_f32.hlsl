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

    if (tid == 0u) {
        dst.Store(dst_off + row * 4u, asuint(sdata[0]));
    }
}
