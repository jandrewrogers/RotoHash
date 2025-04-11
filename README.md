# RotoHash: Checksum for High-Bandwidth Data
RotoHash is a 128-bit hash algorithm optimized for extreme throughput and exceptional hash quality. It was developed to address the need for robust I/O and storage checksums at sustained rates exceeding 100 GB/s but the design is general in nature.

This reference implementation consistently delivers >100 GB/s throughput per core on recent servers. Hash quality is indistinguishable from [MD5](https://en.wikipedia.org/wiki/MD5). RotoHash is highly portable but intended for modern microarchitectures.

This hash design has a high fixed cost but very low marginal cost. It provides state-of-the-art performance for cases where the average key size is hundreds of bytes or larger, such as storage or network I/O. RotoHash is a poor choice for applications where fixed costs dominate performance such as small string hashing. For these use cases an algorithm such as [rapidhash](https://github.com/Nicoshev/rapidhash) may be more appropriate.

## Performance
| CPU Architecture | 256K (Aligned) | 256K (Unaligned) | 1000 Bytes | Less Than 256 Bytes |
| :--------------- | -------------: | ---------------: | ---------: | ------------------: |
| Intel Sapphire Rapids | 240 GB/s | 130 GB/s | 62 cycles/hash | 58 cycles/hash |
| AMD Zen 4             | 147 GB/s | 143 GB/s | 78 cycles/hash | 70 cycles/hash |
| Intel Ice Lake        | 173 GB/s |  61 GB/s | 86 cycles/hash | 82 cycles/hash |

These values are representative of the pure AVX-512 implementation. On older microarchitectures such as Ice Lake, the 128-bit implementation may be faster in some cases.

## Quality
No issues detected under extended testing in test suites such as SMHasher3.

Hash quality is atypically high for an algorithm not specifically designed for cryptographic use. The algorithm can be trivially modified to reduce overhead, thereby increasing performance, while still passing test suites with reduced hash quality. That is not the objective here.

## Construction
The algorithm is built on three 32-bit primitives: xor, rotate, and AES encrypt [0]. It supports incremental hashing. RotoHash has 256 bytes of internal state. This is unusually wide for hash state and incurs a high fixed cost, particularly on older microarchitectures. However, this also increases the potential for parallelism and quality. Support for AVX-512 and vector AES were core design objectives in order to efficiently meet throughput requirements but are not required.

[0] While AES operates on 128-bit blocks, the internals of AES are 32-bit and can be modeled as such.

## Status
The reference implementation currently supports x86-64 with AVX2 (e.g. Intel Haswell) and takes advantage of more advanced features if available. Support for ARM64 will likely be added at some point. Various minor implementation improvements still need to be done.

The baseline algorithm is unlikely to change but analysis and verification of hash quality is ongoing. The algorithm and reference implementation are distributed under the MIT License.
