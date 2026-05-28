// Make a permuted/non-contiguous F16 tensor contiguous in row-major (i0 fastest)
// element order. Output buffer is laid out densely with 2 F16 elements per uint.
//
// Each thread emits one 4-byte store covering two adjacent (in dst) F16
// elements. To avoid relying on byte-addressed 16-bit typed loads, we issue
// 4-byte ByteAddressBuffer loads at the 4-byte-aligned containing word and
// extract the appropriate half.
//
// Root constants (11 dwords):
//   b0.x  = count          (total elements == ggml_nelements)
//   b0.y  = src_off        (base byte offset in src)
//   b0.z  = dst_off        (base byte offset in dst, dense F16 layout)
//   b0.w  = src_ne0        (must be even)
//          src_ne1, src_ne2, src_ne3
//          src_nb0, src_nb1, src_nb2, src_nb3   (byte strides)
//
// UAVs:
//   u0 = src f16 buffer
//   u1 = dst f16 buffer (contiguous, dense)

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint count;
    uint src_off;
    uint dst_off;
    uint src_ne0;
    uint src_ne1;
    uint src_ne2;
    uint src_ne3;
    uint src_nb0;
    uint src_nb1;
    uint src_nb2;
    uint src_nb3;
};

uint load_half(uint addr) {
    const uint aligned = addr & ~3u;
    const uint w = src.Load(aligned);
    return ((addr & 2u) == 0u) ? (w & 0xFFFFu) : ((w >> 16) & 0xFFFFu);
}

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    const uint p = dtid.x;
    const uint pairs = count / 2u;
    if (p >= pairs) return;

    const uint half_ne0 = src_ne0 / 2u;
    uint t = p;
    uint pair_in_row = t % half_ne0;
    t /= half_ne0;
    uint i1 = (src_ne1 == 0u) ? 0u : (t % src_ne1);
    t = (src_ne1 == 0u) ? 0u : (t / src_ne1);
    uint i2 = (src_ne2 == 0u) ? 0u : (t % src_ne2);
    uint i3 = (src_ne2 == 0u) ? 0u : (t / src_ne2);

    const uint i0_lo = pair_in_row * 2u;
    const uint i0_hi = i0_lo + 1u;
    const uint base_addr = src_off + i1 * src_nb1 + i2 * src_nb2 + i3 * src_nb3;
    const uint addr_lo = base_addr + i0_lo * src_nb0;
    const uint addr_hi = base_addr + i0_hi * src_nb0;

    const uint v_lo = load_half(addr_lo);
    const uint v_hi = load_half(addr_hi);
    const uint packed = v_lo | (v_hi << 16);
    dst.Store(dst_off + p * 4u, packed);
}
