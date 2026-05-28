#define TG_SIZE 256
#define QK 32u
#define BLOCK_SIZE 34u
#define QS_OFFSET 2u

RWByteAddressBuffer src_buf : register(u0);
RWByteAddressBuffer idx_buf : register(u1);
RWByteAddressBuffer dst_buf : register(u2);

cbuffer Params : register(b0) {
    uint total;
    uint ne0;
    uint idx_ne0;
    uint idx_ne1;
    uint src_ne1;
    uint src_off;
    uint idx_off;
    uint dst_off;
    uint flags;
};

uint load_u8(RWByteAddressBuffer buf, uint off) {
    const uint word = buf.Load(off & ~3u);
    return (word >> ((off & 3u) * 8u)) & 0xFFu;
}

uint load_u16(RWByteAddressBuffer buf, uint off) {
    return load_u8(buf, off) | (load_u8(buf, off + 1u) << 8u);
}

float load_f16(RWByteAddressBuffer buf, uint off) {
    return f16tof32(load_u16(buf, off));
}

[numthreads(TG_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    const uint out_idx = dtid.x;
    if (out_idx >= total) {
        return;
    }

    const uint col = out_idx % ne0;
    const uint idx_linear = out_idx / ne0;
    const uint i11 = (idx_linear / idx_ne0) % idx_ne1;
    const uint i12 = idx_linear / (idx_ne0 * idx_ne1);

    int row_i = 0;
    if ((flags & 1u) != 0u) {
        row_i = int(idx_linear % src_ne1);
    } else {
        row_i = asint(idx_buf.Load(idx_off + idx_linear * 4u));
    }
    if (row_i < 0 || uint(row_i) >= src_ne1) {
        dst_buf.Store(dst_off + out_idx * 4u, asuint(0.0f));
        return;
    }

    const uint blocks_per_row = ne0 / QK;
    const uint row_bytes = blocks_per_row * BLOCK_SIZE;
    const uint src_nb2 = row_bytes * src_ne1;
    const uint src_nb3 = src_nb2 * idx_ne1;
    const uint block_off = src_off + i12 * src_nb3 + i11 * src_nb2 + uint(row_i) * row_bytes + (col / QK) * BLOCK_SIZE;
    const uint byte_q = load_u8(src_buf, block_off + QS_OFFSET + (col & (QK - 1u)));
    const int q = (int(byte_q) << 24) >> 24;
    const float d = load_f16(src_buf, block_off);
    const float v = float(q) * d;
    dst_buf.Store(dst_off + out_idx * 4u, asuint(v));
}
