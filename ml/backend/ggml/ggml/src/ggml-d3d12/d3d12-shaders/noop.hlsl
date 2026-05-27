RWByteAddressBuffer dst : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    dst.Store(0, 0);
}
