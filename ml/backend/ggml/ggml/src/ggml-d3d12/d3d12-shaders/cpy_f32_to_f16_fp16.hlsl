// Convert a contiguous F32 buffer to a contiguous F16 buffer.
// Used for KV cache writes during decode where the activation arrives as F32
// but the cache stores F16.
//
// Root constants:
//   b0.x = elem_count       (total f32 elements to convert == total f16 dst elems)
//   b0.y = src_byte_offset  (offset in bytes, must be multiple of 4)
//   b0.z = dst_byte_offset  (offset in bytes, must be multiple of 4)
// UAVs:
//   u0 = src f32 (byte-address buffer)
//   u1 = dst f16 (byte-address buffer)
//
// Each thread converts two adjacent F32 elements into a packed uint of two
// F16 halves and emits one 32-bit store. For odd element counts the last
// thread does a read-modify-write on the trailing half-uint.

cbuffer Params : register(b0) {
    uint elem_count;
    uint src_byte_offset;
    uint dst_byte_offset;
    uint _pad;
};

RWByteAddressBuffer src_buf : register(u0);
RWByteAddressBuffer dst_buf : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    const uint p = dtid.x;
    const uint pairs = elem_count / 2u;
    if (p < pairs) {
        const uint i_lo = 2u * p;
        const uint i_hi = 2u * p + 1u;
        const float a = asfloat(src_buf.Load(src_byte_offset + i_lo * 4u));
        const float b = asfloat(src_buf.Load(src_byte_offset + i_hi * 4u));
        const uint pa = (uint)asuint16((float16_t)a);
        const uint pb = (uint)asuint16((float16_t)b);
        const uint packed = pa | (pb << 16);
        dst_buf.Store(dst_byte_offset + p * 4u, packed);
        return;
    }

    if ((elem_count & 1u) != 0u && p == pairs) {
        const uint i_last = elem_count - 1u;
        const float a = asfloat(src_buf.Load(src_byte_offset + i_last * 4u));
        const uint pa = (uint)asuint16((float16_t)a);
        const uint dst_addr = dst_byte_offset + pairs * 4u;
        const uint existing = dst_buf.Load(dst_addr);
        const uint packed = pa | (existing & 0xFFFF0000u);
        dst_buf.Store(dst_addr, packed);
    }
}
