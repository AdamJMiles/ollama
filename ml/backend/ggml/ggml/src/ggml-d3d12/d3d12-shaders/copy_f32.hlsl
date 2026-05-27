// Simple byte-granular copy used as the Phase 4 smoke test.
// Root constants:
//   b0.x = byte_count       (total bytes to copy, must be multiple of 4)
//   b0.y = src_byte_offset  (offset in bytes, must be multiple of 4)
//   b0.z = dst_byte_offset  (offset in bytes, must be multiple of 4)
// UAVs:
//   u0 = src (byte-address buffer)
//   u1 = dst (byte-address buffer)

cbuffer Params : register(b0) {
    uint byte_count;
    uint src_byte_offset;
    uint dst_byte_offset;
    uint _pad;
};

RWByteAddressBuffer src_buf : register(u0);
RWByteAddressBuffer dst_buf : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    const uint i = dtid.x * 4u; // 4 bytes per thread
    if (i >= byte_count) {
        return;
    }
    uint v = src_buf.Load(src_byte_offset + i);
    dst_buf.Store(dst_byte_offset + i, v);
}
