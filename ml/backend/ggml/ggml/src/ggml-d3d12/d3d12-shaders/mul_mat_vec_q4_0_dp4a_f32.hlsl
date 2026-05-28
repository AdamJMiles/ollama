#define TG 64
#define QK 32u
#define BLOCK_SIZE 18u
#define QS_OFFSET 2u

RWByteAddressBuffer src0_buf : register(u0);
RWByteAddressBuffer src1_buf : register(u1);
RWByteAddressBuffer dst_buf  : register(u2);

cbuffer Params : register(b0) {
    uint K;
    uint M;
    uint batch;
    uint src0_row_stride;
    uint src1_row_stride;
    uint dst_row_stride;
    uint src0_off;
    uint src1_off;
    uint dst_off;
    uint ne2;
    uint src0_nb2;
    uint src0_nb3;
    uint src1_nb2;
    uint src1_nb3;
    uint dst_nb2;
    uint dst_nb3;
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

float load_src1(uint vec_base, uint k) {
    return asfloat(src1_buf.Load(vec_base + k * 4u));
}

uint pack_i8_4(int q0, int q1, int q2, int q3) {
    return (uint(q0) & 0xFFu) |
           ((uint(q1) & 0xFFu) << 8u) |
           ((uint(q2) & 0xFFu) << 16u) |
           ((uint(q3) & 0xFFu) << 24u);
}

int quantize_i8(float x, float inv_scale) {
    return clamp(int(round(x * inv_scale)), -127, 127);
}

float block_max_abs(uint vec_base, uint block) {
    const uint k0 = block * QK;
    float max_abs = 0.0f;
    [unroll]
    for (uint i = 0u; i < QK; ++i) {
        max_abs = max(max_abs, abs(load_src1(vec_base, k0 + i)));
    }
    return max_abs;
}

uint pack_activations(uint vec_base, uint block, uint group, float inv_scale) {
    const uint k0 = block * QK + group * 4u;
    const int q0 = quantize_i8(load_src1(vec_base, k0 + 0u), inv_scale);
    const int q1 = quantize_i8(load_src1(vec_base, k0 + 1u), inv_scale);
    const int q2 = quantize_i8(load_src1(vec_base, k0 + 2u), inv_scale);
    const int q3 = quantize_i8(load_src1(vec_base, k0 + 3u), inv_scale);
    return pack_i8_4(q0, q1, q2, q3);
}

int q4_0_value(uint block_off, uint elem) {
    const uint packed = load_u8(src0_buf, block_off + QS_OFFSET + (elem & 15u));
    const uint q = (elem < 16u) ? (packed & 0x0Fu) : (packed >> 4u);
    return int(q) - 8;
}

uint pack_q4_0(uint block_off, uint group) {
    const uint elem = group * 4u;
    return pack_i8_4(q4_0_value(block_off, elem + 0u),
                     q4_0_value(block_off, elem + 1u),
                     q4_0_value(block_off, elem + 2u),
                     q4_0_value(block_off, elem + 3u));
}

groupshared uint q4_cache[TG * 8u];
groupshared float partials[TG];

[numthreads(TG, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint m = gid.x;
    const uint b = gid.y;
    if (m >= M || b >= batch) {
        return;
    }

    const uint i2 = b % ne2;
    const uint i3 = b / ne2;
    const uint row_base = src0_off + i2 * src0_nb2 + i3 * src0_nb3 + m * src0_row_stride;
    const uint vec_base = src1_off + i2 * src1_nb2 + i3 * src1_nb3;
    const uint num_blocks = K / QK;

    float acc = 0.0f;
    for (uint block_base = 0u; block_base < num_blocks; block_base += TG) {
        const uint block = block_base + gtid.x;
        const bool active = block < num_blocks;
        uint block_off = 0u;
        if (active) {
            block_off = row_base + block * BLOCK_SIZE;
            [unroll]
            for (uint group = 0u; group < 8u; ++group) {
                q4_cache[gtid.x * 8u + group] = pack_q4_0(block_off, group);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (active) {
            const float max_abs = block_max_abs(vec_base, block);
            const float inv_scale = (max_abs > 0.0f) ? (127.0f / max_abs) : 0.0f;
            const float x_scale = max_abs * (1.0f / 127.0f);

            int isum = 0;
            [unroll]
            for (uint group = 0u; group < 8u; ++group) {
                isum = dot4add_i8packed(pack_activations(vec_base, block, group, inv_scale),
                                        q4_cache[gtid.x * 8u + group],
                                        isum);
            }
            acc += float(isum) * x_scale * load_f16(src0_buf, block_off);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    partials[gtid.x] = acc;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = TG / 2u; stride > 0u; stride >>= 1u) {
        if (gtid.x < stride) {
            partials[gtid.x] += partials[gtid.x + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gtid.x == 0u) {
        dst_buf.Store(dst_off + i2 * dst_nb2 + i3 * dst_nb3 + m * 4u, asuint(partials[0]));
    }
}
