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
void main(uint3 gtid : SV_GroupThreadID) {
    uint tid = gtid.x;

    // One workgroup sums the full tensor for now; split reductions can speed this up later.
    float local = 0.0f;
    for (uint i = tid; i < ne0; i += TG_SIZE) {
        local += asfloat(src.Load(src_off + i * 4u));
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
        dst.Store(dst_off, asuint(sdata[0]));
    }
}
