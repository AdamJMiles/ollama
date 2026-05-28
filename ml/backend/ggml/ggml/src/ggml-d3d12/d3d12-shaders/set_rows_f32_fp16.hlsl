// GGML_OP_SET_ROWS — write rows of an F32 source tensor into indexed rows of
// a destination tensor (F32 or F16). Indices are I32 or I64.
//
// Selected via flags bits:
//   bit 0 = dst is F16 (else F32)
//   bit 1 = idx is I64 (else I32)
//
// Semantics (matches ggml_compute_forward_set_rows):
//   for (i03 in [0, ne03)) for (i02 in [0, ne02)) for (i in [0, nr)):
//     i1 = idx[i, i02 % ne11, i03 % ne12]
//     dst[:, i1, i02, i03] = (cast<dst_type>) src[:, i, i02, i03]   // nc elements
//
// One thread group per (i, i02, i03) source row. 64 threads cooperate on the
// row's columns (strided by 64). For F16 dst, each thread handles a pair so it
// can do a single 32-bit store of two packed halves.
//
// Root constants:
//   b0.x  nc                 elements per row (== src->ne[0] == dst->ne[0])
//   b0.y  nr                 rows in src (== src->ne[1])
//   b0.z  ne02               src batch dim 0
//   b0.w  ne03               src batch dim 1
//   b1.x  ne11               idx batch dim 0
//   b1.y  ne12               idx batch dim 1
//   b1.z  src_off            byte offset of src data
//   b1.w  dst_off            byte offset of dst data
//   b2.x  idx_off            byte offset of idx data
//   b2.y  src_nb1            src row stride in bytes (column 1 stride)
//   b2.z  src_nb2            src batch stride 0 in bytes
//   b2.w  src_nb3            src batch stride 1 in bytes
//   b3.x  dst_nb1            dst row stride in bytes
//   b3.y  dst_nb2            dst batch stride 0 in bytes
//   b3.z  dst_nb3            dst batch stride 1 in bytes
//   b3.w  idx_nb0            idx element stride in bytes (4 for i32, 8 for i64)
//   b4.x  idx_nb1            idx batch stride 0 in bytes
//   b4.y  idx_nb2            idx batch stride 1 in bytes
//   b4.z  flags              bit0 = dst_is_f16, bit1 = idx_is_i64
//   b4.w  _pad
//
// UAVs:
//   u0 = src  (R32_TYPELESS ByteAddressBuffer of f32)
//   u1 = idx  (R32_TYPELESS ByteAddressBuffer of i32/i64)
//   u2 = dst  (R32_TYPELESS ByteAddressBuffer of f32/f16)

#define TG_SIZE 64

cbuffer Params : register(b0) {
    uint nc;
    uint nr;
    uint ne02;
    uint ne03;
    uint ne11;
    uint ne12;
    uint src_off;
    uint dst_off;
    uint idx_off;
    uint src_nb1;
    uint src_nb2;
    uint src_nb3;
    uint dst_nb1;
    uint dst_nb2;
    uint dst_nb3;
    uint idx_nb0;
    uint idx_nb1;
    uint idx_nb2;
    uint flags;
    uint _pad;
};

RWByteAddressBuffer src_buf : register(u0);
RWByteAddressBuffer idx_buf : register(u1);
RWByteAddressBuffer dst_buf : register(u2);

[numthreads(TG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint linear_row = gid.x;
    const uint i   = linear_row % nr;
    const uint tmp = linear_row / nr;
    const uint i02 = tmp % ne02;
    const uint i03 = tmp / ne02;

    if (i03 >= ne03) {
        return;
    }

    const bool dst_is_f16 = (flags & 1u) != 0u;
    const bool idx_is_i64 = (flags & 2u) != 0u;

    const uint i11 = i02 % ne11;
    const uint i12 = i03 % ne12;

    // idx address: idx_data + i*idx_nb0 + i11*idx_nb1 + i12*idx_nb2
    // i64 indices are stored little-endian so the low 32 bits are at the base
    // address. Valid KV cache indices fit in 32 bits, so we just read the low
    // word in both cases.
    const uint idx_addr = idx_off + i * idx_nb0 + i11 * idx_nb1 + i12 * idx_nb2;
    const int i1 = asint(idx_buf.Load(idx_addr));

    const uint src_row = src_off + i  * src_nb1 + i02 * src_nb2 + i03 * src_nb3;
    const uint dst_row = dst_off + (uint)i1 * dst_nb1 + i02 * dst_nb2 + i03 * dst_nb3;

    if (!dst_is_f16) {
        // F32 dst: one 4-byte store per element, strided by TG_SIZE.
        for (uint k = gtid.x; k < nc; k += TG_SIZE) {
            const uint v = src_buf.Load(src_row + k * 4u);
            dst_buf.Store(dst_row + k * 4u, v);
        }
        return;
    }

    // F16 dst: pack two consecutive halves into one 32-bit store. Each thread
    // owns pair p (covering columns 2p and 2p+1). For typical model head
    // dimensions (multiple of 2) every store is a clean 32-bit write.
    //
    // Use native `float16_t` casts (-enable-16bit-types, cs_6_2+) so the
    // conversion is guaranteed round-to-nearest-even, matching the CPU
    // reference's `_cvtss_sh(x, 0)`. The HLSL `f32tof16` intrinsic has
    // implementation-defined rounding and on some drivers truncates, which
    // pushes accumulated NMSE just above the test tolerance.
    const uint n_pairs = nc / 2u;
    for (uint p = gtid.x; p < n_pairs; p += TG_SIZE) {
        const float a = asfloat(src_buf.Load(src_row + (2u * p)      * 4u));
        const float b = asfloat(src_buf.Load(src_row + (2u * p + 1u) * 4u));
        const uint pa = (uint)asuint16((float16_t)a);
        const uint pb = (uint)asuint16((float16_t)b);
        const uint packed = pa | (pb << 16);
        dst_buf.Store(dst_row + p * 4u, packed);
    }

    // Trailing odd element (rare): read-modify-write the upper half so we
    // don't disturb the neighbour stored by a sibling row.
    if ((nc & 1u) != 0u && gtid.x == 0u) {
        const uint p_last = n_pairs;
        const float a = asfloat(src_buf.Load(src_row + (nc - 1u) * 4u));
        const uint pa = (uint)asuint16((float16_t)a);
        const uint existing = dst_buf.Load(dst_row + p_last * 4u);
        const uint packed   = pa | (existing & 0xFFFF0000u);
        dst_buf.Store(dst_row + p_last * 4u, packed);
    }
}
