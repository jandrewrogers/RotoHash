# RotoHash: Hashing for High-Bandwidth Data
RotoHash is a novel 128-bit AES-based hash construction optimized for extreme throughput and extreme hash quality. This implementation is currently state-of-the-art in both dimensions. The motivating use case is detecting data corruption at I/O rates greatly exceeding 100 GB/s. RotoHash delivers exceptional hash quality across all key sizes but has a relatively high fixed overhead and is therefore a poor choice for applications like small string hashing where overhead dominates hashing costs. It is optimized for cases where the average key size is hundreds of bytes or larger. The core algorithm design assumes modern server hardware. 

A "smaller" version of the algorithm is under development that trades some throughput for lower overhead and better compatibility with more limited CPUs. 

## Performance
On an Intel "Sapphire Rapids" test environment @ 3.5 GHz:
- 235 GB/s aligned, bulk
- 130 GB/s unaligned, bulk
- 57 cycles/hash, keys smaller than 256 bytes

On an Intel "Ice Lake" test environment @ 3.5 GHz:
- 170 GB/s aligned, bulk
- 61 GB/s unaligned, bulk
- 78 cycles/hash, keys smaller than 256 bytes

## Quality
No issues detected across extended testing in any test suite e.g. SMHasher3. 

The distribution of random oracle p-values is at the lower end of true cryptographic hash functions. Not as good as SHA-256 but significantly better than any fast non-cryptographic hash function of which I am aware. RotoHash is not intended for cryptographic use cases.

## Algorithm Construction
The algorithm uses three 32-bit primitives: xor, rotate, and AES encrypt [0]. It is designed to support incremental hashing. Hash state of the base algorithm is 256 bytes. This is an unusually large hash state and the primary cause of the relatively high overhead. It also greatly increases the potential for both quality and parallelism; implementation using AVX-512 and VAES was a core design objective due to the extreme throughput requirements. The construction is maximally simple to facilitate analysis.

Porting is straightforward to any hardware platform with AES support. However, the size of the hash state may create performance problems on some microarchitectures. A "small" algorithm construction based on 64 bytes of hash state is under development that is well-suited to CPUs limited to AVX2 or NEON vector instructions.

[0] While AES operates on 128-bit blocks, the internals of AES are 32-bit and can be modeled as such.

## Status
Algorithm analysis is ongoing. This may result in design tweaks to improve algorithm quality at the limits. That will have little impact on current implementation usability.

Alternative algorithms that make somewhat different engineering tradeoffs may be added in the future.
