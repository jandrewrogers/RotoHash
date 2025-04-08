#pragma once

/*
Copyright (c) 2025 Andrew Rogers

This is a minimalist reference implementation of the RotoHash algorithm for both
128-bit and 512-bit platforms. A static VerifyImplementation() method that
checks implementation correctness.
*/

#include <cstdint>
#include <cstring>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

class RotoHash {
private:
    /*
    microarchitecture targets
    - AVX2   (>= Haswell)
    - AVX512 (>= Skylake)
    - AVX512 (>= Icelake)
    - ARM64 (tbd)
    */
#if defined(__x86_64__) && defined(__AVX2__)
    // x86 128-bit primitives
    using Scalar = __m128i;

    static __m128i Xor(__m128i lhs, __m128i rhs) {
        return _mm_xor_si128(lhs, rhs);
    }

    static __m128i Enc(__m128i aes, __m128i key) {
        return _mm_aesenc_si128(aes, key);
    }

    static __m128i Rot(__m128i value, __m128i shift) {
    #if defined(__AVX512F__) && defined(__AVX512VL__)
        return _mm_rolv_epi32(value, shift);
    #else
        // AVX2 does not have a native rotate instruction
        __m128i lshift = _mm_and_si128(_mm_set1_epi32(0x1F),  shift);
        __m128i rshift = _mm_sub_epi32(_mm_set1_epi32(0x20), lshift);
        __m128i lvalue = _mm_sllv_epi32(value, lshift);
        __m128i rvalue = _mm_srlv_epi32(value, rshift);
        return _mm_or_si128(lvalue, rvalue);
    #endif
    }

    #if defined(__AVX512F__) && defined(__VAES__)
        // x86 512-bit implementation
        using Vector = __m512i;

        static __m512i Xor(__m512i lhs, __m512i rhs) {
            return _mm512_xor_si512(lhs, rhs);
        }

        static __m512i Enc(__m512i aes, __m512i key) {
            return _mm512_aesenc_epi128(aes, key);
        }

        static __m512i Rot(__m512i value, __m512i shift) {
            return _mm512_rolv_epi32(value, shift);
        }

        static __m512i Load(const void* ptr) {
            return _mm512_loadu_si512(ptr);
        }

        static __m512i Load(const void* ptr, size_t bytes) {
            uint64_t mask = 0xFFFFFFFFFFFFFFFFull >> (64 - bytes);
            return _mm512_maskz_loadu_epi8(mask, ptr);
        }

        static __m512i SetAll(const uint64_t value) {
            return _mm512_set1_epi64(value);
        }

        template <size_t N>
        static __m128i Lane(const __m512i vector) {
            return _mm512_extracti32x4_epi32(vector, N);
        }

    #else
        static __m128i Load(const void* ptr) {
            return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
        }

        static __m128i Load(const void* ptr, size_t bytes) {
        #if defined(__AVX512F__)
            uint64_t mask = 0xFFFFu >> (sizeof(__m128i) - bytes);
            return _mm_maskz_loadu_epi8(mask, ptr);
        #else
            // TODO: replace with the old _mm_shuffle_epi8 masked load trick
            alignas(sizeof(__m128i)) uint8_t buffer[sizeof(__m128i)];
            memset(buffer, 0, sizeof(buffer));
            memcpy(buffer, ptr, bytes);
            return Load(buffer);
        #endif
        }

        static __m128i SetAll(const uint64_t value) {
            return _mm_set1_epi64x(value);
        }
    #endif
#else
#error unsupported architecture
#endif

public:
    // The 256-byte constant is random but not arbitrary. There is non-obvious
    // structure between bytes created by minimally editing the output of an
    // entropy source to maximize certain properties. Shuffling these bytes
    // randomly or replacing with random numbers may degrade hash quality.
    alignas(64) static constexpr uint8_t Constant[] = {
        0x8Fu, 0x5Bu, 0x86u, 0x39u, 0x77u, 0xFCu, 0x2Au, 0x2Eu,
        0xB2u, 0x70u, 0x4Cu, 0x69u, 0xC2u, 0x65u, 0xD1u, 0x91u,
        0x71u, 0x18u, 0x15u, 0xADu, 0xF5u, 0x62u, 0x95u, 0x3Eu,
        0x6Eu, 0x99u, 0x94u, 0xE3u, 0xB1u, 0x6Cu, 0x30u, 0x6Du,
        0xF8u, 0xCAu, 0x4Bu, 0xEFu, 0xA8u, 0x98u, 0x75u, 0x40u,
        0xD8u, 0x43u, 0x6Bu, 0x0Au, 0x63u, 0x11u, 0x38u, 0x21u,
        0x16u, 0x4Au, 0xA7u, 0x5Du, 0x42u, 0xAAu, 0x8Bu, 0x33u,
        0x47u, 0x19u, 0x59u, 0xBCu, 0xC5u, 0xD4u, 0xF3u, 0xD0u,
        0x7Au, 0x74u, 0x4Eu, 0xB0u, 0x37u, 0x52u, 0x10u, 0x73u,
        0x8Eu, 0x06u, 0x17u, 0x20u, 0xAEu, 0xF2u, 0xD6u, 0x48u,
        0x3Du, 0x8Au, 0xE8u, 0x8Du, 0xC9u, 0x84u, 0x68u, 0x41u,
        0xDDu, 0x1Au, 0x1Eu, 0x2Du, 0xA1u, 0xA3u, 0x8Cu, 0x0Du,
        0x7Bu, 0x45u, 0x83u, 0x04u, 0xC0u, 0xA2u, 0xE5u, 0x67u,
        0xD3u, 0x9Au, 0x9Fu, 0xBDu, 0xC8u, 0x34u, 0x24u, 0xDCu,
        0xFFu, 0x0Cu, 0x51u, 0x7Cu, 0x89u, 0xDFu, 0xA6u, 0xCEu,
        0x5Au, 0x3Au, 0x79u, 0x35u, 0x76u, 0x0Eu, 0x22u, 0x60u,
        0xE4u, 0x80u, 0x9Du, 0x14u, 0xF9u, 0xF4u, 0x7Eu, 0x0Bu,
        0xBBu, 0x58u, 0x6Au, 0x3Fu, 0xC1u, 0x4Du, 0xACu, 0xF0u,
        0xFAu, 0x5Eu, 0xE0u, 0xB8u, 0x92u, 0xB5u, 0x4Fu, 0xB4u,
        0xB3u, 0x08u, 0xECu, 0x3Bu, 0x64u, 0x9Eu, 0xEBu, 0x07u,
        0xBEu, 0x13u, 0x23u, 0x7Du, 0x9Bu, 0x09u, 0x28u, 0x88u,
        0x90u, 0x72u, 0x5Cu, 0xF7u, 0x36u, 0x29u, 0x1Fu, 0x97u,
        0xA4u, 0x25u, 0x56u, 0xEDu, 0x66u, 0x78u, 0xD5u, 0x44u,
        0x61u, 0x57u, 0x2Bu, 0xD2u, 0x54u, 0xFBu, 0xEEu, 0xD9u,
        0x2Cu, 0x02u, 0x05u, 0x2Fu, 0x87u, 0x85u, 0x9Cu, 0xD7u,
        0xE6u, 0x12u, 0x7Fu, 0xC6u, 0xFDu, 0x49u, 0xC3u, 0xEAu,
        0xB7u, 0x03u, 0x26u, 0x1Cu, 0xA5u, 0xE7u, 0x1Bu, 0xFEu,
        0xCCu, 0x81u, 0xB6u, 0x3Cu, 0xF1u, 0x93u, 0x01u, 0xE9u,
        0x27u, 0xF6u, 0x82u, 0xCBu, 0x1Du, 0xBAu, 0xDEu, 0xDAu,
        0x32u, 0x6Fu, 0xC4u, 0xC7u, 0xA0u, 0xABu, 0x53u, 0x46u,
        0xE2u, 0x55u, 0xCFu, 0xCDu, 0x96u, 0x00u, 0x31u, 0x5Fu,
        0x0Fu, 0xA9u, 0xB9u, 0xE1u, 0xBFu, 0x50u, 0xAFu, 0xDBu,
    };

#if defined(__AVX512F__) && defined(__VAES__)
    /*
    Baseline 512-bit algorithm implementation. Every other implementation ground
    truths off of this implementation due to its conciseness and simplicity.
    */
    static Scalar Hash(const void* data, const size_t size, uint64_t seed = 0) {
        size_t bytes = size;

        const Vector c0 = Load(&Constant[sizeof(Vector) * 0]);
        const Vector c1 = Load(&Constant[sizeof(Vector) * 1]);
        const Vector c2 = Load(&Constant[sizeof(Vector) * 2]);
        const Vector c3 = Load(&Constant[sizeof(Vector) * 3]);

        Vector v0 = c0;
        Vector v1 = c1;
        Vector v2 = c2;
        Vector v3 = c3;

        v0 = Xor(v0, SetAll(seed));

        const Vector* key = reinterpret_cast<const Vector*>(data);
        // bulk loop
        if (bytes >= 256)
            do {
                v0 = Enc(v0, Load(key++));
                v1 = Enc(v1, Load(key++));
                v2 = Enc(v2, Load(key++));
                v3 = Enc(v3, Load(key++));
            }
            while ((bytes -= 256) >= 256);

        v1 = Xor(v1, SetAll(size));

        // remainder
        if (bytes > 0) {
            if (bytes >=  64)
                v0 = Enc(v0, Load(key++));
            if (bytes >= 128)
                v1 = Enc(v1, Load(key++));
            if (bytes >= 192)
                v2 = Enc(v2, Load(key++));
            if (bytes %=  64)
                v3 = Enc(v3, Load(key, bytes));
        }
        {   // 4x state reduction
            Vector m0, m1, m2, m3;
            m0 = Enc(Xor(Rot(v1, v0), v0), Rot(c1, v2));
            m1 = Enc(Xor(Rot(v2, v1), v1), Rot(c0, v3));
            m2 = Enc(Xor(Rot(v3, v2), v2), Rot(c3, v0));
            m3 = Enc(Xor(Rot(v0, v3), v3), Rot(c2, v1));

            m0 = Enc(Xor(Rot(v2, v0), m0), Rot(c2, v3));
            m1 = Enc(Xor(Rot(v3, v1), m1), Rot(c3, v2));
            m2 = Enc(Xor(Rot(v0, v2), m2), Rot(c0, v1));
            m3 = Enc(Xor(Rot(v1, v3), m3), Rot(c1, v0));

            m0 = Enc(Xor(Rot(v3, v0), m0), Rot(c3, v1));
            m1 = Enc(Xor(Rot(v0, v1), m1), Rot(c2, v0));
            m2 = Enc(Xor(Rot(v1, v2), m2), Rot(c1, v3));
            m3 = Enc(Xor(Rot(v2, v3), m3), Rot(c0, v2));

            v0 = Xor(Xor(m0, m1), Xor(m2, m3));
        }
        Scalar s0 = Lane<0>(v0);
        Scalar s1 = Lane<1>(v0);
        Scalar s2 = Lane<2>(v0);
        Scalar s3 = Lane<3>(v0);
        {   // 4x state reduction
            Scalar m0, m1, m2, m3;
            m0 = Enc(Xor(Rot(s1, s0), s0), s3);
            m1 = Enc(Xor(Rot(s2, s1), s1), s0);
            m2 = Enc(Xor(Rot(s3, s2), s2), s1);
            m3 = Enc(Xor(Rot(s0, s3), s3), s2);

            m0 = Enc(Xor(Rot(s2, s0), m0), s1);
            m1 = Enc(Xor(Rot(s3, s1), m1), s2);
            m2 = Enc(Xor(Rot(s0, s2), m2), s3);
            m3 = Enc(Xor(Rot(s1, s3), m3), s0);

            m0 = Enc(Xor(Rot(s3, s0), m0), s2);
            m1 = Enc(Xor(Rot(s0, s1), m1), s3);
            m2 = Enc(Xor(Rot(s1, s2), m2), s0);
            m3 = Enc(Xor(Rot(s2, s3), m3), s1);

            s0 = Xor(Xor(m0, m1), Xor(m2, m3));
        }
        return s0;
    }

#else
    /*
    Alternative 128-bit implementations based on microarchitecture capabilities.
    This exists to maximize portability. In most cases these implementations
    still offer >100 GB/s of hash performance on modern CPUs.
    */

    // 128-bit portable implementation
    static Scalar Hash(const void* data, const size_t size, uint64_t seed = 0) {
        size_t bytes = size;

        Scalar c[16] = {
            Load(&Constant[sizeof(Scalar) *  0]),
            Load(&Constant[sizeof(Scalar) *  1]),
            Load(&Constant[sizeof(Scalar) *  2]),
            Load(&Constant[sizeof(Scalar) *  3]),
            Load(&Constant[sizeof(Scalar) *  4]),
            Load(&Constant[sizeof(Scalar) *  5]),
            Load(&Constant[sizeof(Scalar) *  6]),
            Load(&Constant[sizeof(Scalar) *  7]),
            Load(&Constant[sizeof(Scalar) *  8]),
            Load(&Constant[sizeof(Scalar) *  9]),
            Load(&Constant[sizeof(Scalar) * 10]),
            Load(&Constant[sizeof(Scalar) * 11]),
            Load(&Constant[sizeof(Scalar) * 12]),
            Load(&Constant[sizeof(Scalar) * 13]),
            Load(&Constant[sizeof(Scalar) * 14]),
            Load(&Constant[sizeof(Scalar) * 15]),
        };

        Scalar v[16] = {
            c[ 0], c[ 1], c[ 2], c[ 3], c[ 4], c[ 5], c[ 6], c[ 7],
            c[ 8], c[ 9], c[10], c[11], c[12], c[13], c[14], c[15],
        };

        const Scalar sv = SetAll(seed);
        v[ 0] = Xor(v[ 0], sv);
        v[ 1] = Xor(v[ 1], sv);
        v[ 2] = Xor(v[ 2], sv);
        v[ 3] = Xor(v[ 3], sv);

        const Scalar* key = reinterpret_cast<const Scalar*>(data);
        if (bytes >= 256)
            do {
                v[ 0] = Enc(v[ 0], Load(key++));
                v[ 1] = Enc(v[ 1], Load(key++));
                v[ 2] = Enc(v[ 2], Load(key++));
                v[ 3] = Enc(v[ 3], Load(key++));
                v[ 4] = Enc(v[ 4], Load(key++));
                v[ 5] = Enc(v[ 5], Load(key++));
                v[ 6] = Enc(v[ 6], Load(key++));
                v[ 7] = Enc(v[ 7], Load(key++));
                v[ 8] = Enc(v[ 8], Load(key++));
                v[ 9] = Enc(v[ 9], Load(key++));
                v[10] = Enc(v[10], Load(key++));
                v[11] = Enc(v[11], Load(key++));
                v[12] = Enc(v[12], Load(key++));
                v[13] = Enc(v[13], Load(key++));
                v[14] = Enc(v[14], Load(key++));
                v[15] = Enc(v[15], Load(key++));
            }
            while ((bytes -= 256) >= 256);

        const Scalar lv = SetAll(size);
        v[ 4] = Xor(v[ 4], lv);
        v[ 5] = Xor(v[ 5], lv);
        v[ 6] = Xor(v[ 6], lv);
        v[ 7] = Xor(v[ 7], lv);

        if (bytes > 0) {
            if (bytes >= 64) {
                v[ 0] = Enc(v[ 0], Load(key++));
                v[ 1] = Enc(v[ 1], Load(key++));
                v[ 2] = Enc(v[ 2], Load(key++));
                v[ 3] = Enc(v[ 3], Load(key++));
            }
            if (bytes >= 128) {
                v[ 4] = Enc(v[ 4], Load(key++));
                v[ 5] = Enc(v[ 5], Load(key++));
                v[ 6] = Enc(v[ 6], Load(key++));
                v[ 7] = Enc(v[ 7], Load(key++));
            }
            if (bytes >= 192) {
                v[ 8] = Enc(v[ 8], Load(key++));
                v[ 9] = Enc(v[ 9], Load(key++));
                v[10] = Enc(v[10], Load(key++));
                v[11] = Enc(v[11], Load(key++));
            }
            if (bytes %= 64) {
                Scalar tmp[4];
                int index = 0;
                if (bytes >= sizeof(Scalar))
                    do    tmp[index++] = Load(key++);
                    while ((bytes -= sizeof(Scalar)) >= sizeof(Scalar));

                if (bytes %= sizeof(Scalar))
                    tmp[index++] = Load(key, bytes);

                if (index < 4)
                    do    tmp[index] = SetAll(0);
                    while (++index < 4);

                v[12] = Enc(v[12], tmp[0]);
                v[13] = Enc(v[13], tmp[1]);
                v[14] = Enc(v[14], tmp[2]);
                v[15] = Enc(v[15], tmp[3]);
            }
        }

        Scalar m[16] = {
            v[ 0], v[ 1], v[ 2], v[ 3], v[ 4], v[ 5], v[ 6], v[ 7],
            v[ 8], v[ 9], v[10], v[11], v[12], v[13], v[14], v[15]
        };

        m[ 0] = Enc(Xor(Rot(v[ 4], v[ 0]), m[ 0]), Rot(c[ 4], v[ 8]));
        m[ 1] = Enc(Xor(Rot(v[ 5], v[ 1]), m[ 1]), Rot(c[ 5], v[ 9]));
        m[ 2] = Enc(Xor(Rot(v[ 6], v[ 2]), m[ 2]), Rot(c[ 6], v[10]));
        m[ 3] = Enc(Xor(Rot(v[ 7], v[ 3]), m[ 3]), Rot(c[ 7], v[11]));

        m[ 4] = Enc(Xor(Rot(v[ 8], v[ 4]), m[ 4]), Rot(c[ 0], v[12]));
        m[ 5] = Enc(Xor(Rot(v[ 9], v[ 5]), m[ 5]), Rot(c[ 1], v[13]));
        m[ 6] = Enc(Xor(Rot(v[10], v[ 6]), m[ 6]), Rot(c[ 2], v[14]));
        m[ 7] = Enc(Xor(Rot(v[11], v[ 7]), m[ 7]), Rot(c[ 3], v[15]));

        m[ 8] = Enc(Xor(Rot(v[12], v[ 8]), m[ 8]), Rot(c[12], v[ 0]));
        m[ 9] = Enc(Xor(Rot(v[13], v[ 9]), m[ 9]), Rot(c[13], v[ 1]));
        m[10] = Enc(Xor(Rot(v[14], v[10]), m[10]), Rot(c[14], v[ 2]));
        m[11] = Enc(Xor(Rot(v[15], v[11]), m[11]), Rot(c[15], v[ 3]));

        m[12] = Enc(Xor(Rot(v[ 0], v[12]), m[12]), Rot(c[ 8], v[ 4]));
        m[13] = Enc(Xor(Rot(v[ 1], v[13]), m[13]), Rot(c[ 9], v[ 5]));
        m[14] = Enc(Xor(Rot(v[ 2], v[14]), m[14]), Rot(c[10], v[ 6]));
        m[15] = Enc(Xor(Rot(v[ 3], v[15]), m[15]), Rot(c[11], v[ 7]));

        m[ 0] = Enc(Xor(Rot(v[ 8], v[ 0]), m[ 0]), Rot(c[ 8], v[12]));
        m[ 1] = Enc(Xor(Rot(v[ 9], v[ 1]), m[ 1]), Rot(c[ 9], v[13]));
        m[ 2] = Enc(Xor(Rot(v[10], v[ 2]), m[ 2]), Rot(c[10], v[14]));
        m[ 3] = Enc(Xor(Rot(v[11], v[ 3]), m[ 3]), Rot(c[11], v[15]));

        m[ 4] = Enc(Xor(Rot(v[12], v[ 4]), m[ 4]), Rot(c[12], v[ 8]));
        m[ 5] = Enc(Xor(Rot(v[13], v[ 5]), m[ 5]), Rot(c[13], v[ 9]));
        m[ 6] = Enc(Xor(Rot(v[14], v[ 6]), m[ 6]), Rot(c[14], v[10]));
        m[ 7] = Enc(Xor(Rot(v[15], v[ 7]), m[ 7]), Rot(c[15], v[11]));

        m[ 8] = Enc(Xor(Rot(v[ 0], v[ 8]), m[ 8]), Rot(c[ 0], v[ 4]));
        m[ 9] = Enc(Xor(Rot(v[ 1], v[ 9]), m[ 9]), Rot(c[ 1], v[ 5]));
        m[10] = Enc(Xor(Rot(v[ 2], v[10]), m[10]), Rot(c[ 2], v[ 6]));
        m[11] = Enc(Xor(Rot(v[ 3], v[11]), m[11]), Rot(c[ 3], v[ 7]));

        m[12] = Enc(Xor(Rot(v[ 4], v[12]), m[12]), Rot(c[ 4], v[ 0]));
        m[13] = Enc(Xor(Rot(v[ 5], v[13]), m[13]), Rot(c[ 5], v[ 1]));
        m[14] = Enc(Xor(Rot(v[ 6], v[14]), m[14]), Rot(c[ 6], v[ 2]));
        m[15] = Enc(Xor(Rot(v[ 7], v[15]), m[15]), Rot(c[ 7], v[ 3]));

        m[ 0] = Enc(Xor(Rot(v[12], v[ 0]), m[ 0]), Rot(c[12], v[ 4]));
        m[ 1] = Enc(Xor(Rot(v[13], v[ 1]), m[ 1]), Rot(c[13], v[ 5]));
        m[ 2] = Enc(Xor(Rot(v[14], v[ 2]), m[ 2]), Rot(c[14], v[ 6]));
        m[ 3] = Enc(Xor(Rot(v[15], v[ 3]), m[ 3]), Rot(c[15], v[ 7]));

        m[ 4] = Enc(Xor(Rot(v[ 0], v[ 4]), m[ 4]), Rot(c[ 8], v[ 0]));
        m[ 5] = Enc(Xor(Rot(v[ 1], v[ 5]), m[ 5]), Rot(c[ 9], v[ 1]));
        m[ 6] = Enc(Xor(Rot(v[ 2], v[ 6]), m[ 6]), Rot(c[10], v[ 2]));
        m[ 7] = Enc(Xor(Rot(v[ 3], v[ 7]), m[ 7]), Rot(c[11], v[ 3]));

        m[ 8] = Enc(Xor(Rot(v[ 4], v[ 8]), m[ 8]), Rot(c[ 4], v[12]));
        m[ 9] = Enc(Xor(Rot(v[ 5], v[ 9]), m[ 9]), Rot(c[ 5], v[13]));
        m[10] = Enc(Xor(Rot(v[ 6], v[10]), m[10]), Rot(c[ 6], v[14]));
        m[11] = Enc(Xor(Rot(v[ 7], v[11]), m[11]), Rot(c[ 7], v[15]));

        m[12] = Enc(Xor(Rot(v[ 8], v[12]), m[12]), Rot(c[ 0], v[ 8]));
        m[13] = Enc(Xor(Rot(v[ 9], v[13]), m[13]), Rot(c[ 1], v[ 9]));
        m[14] = Enc(Xor(Rot(v[10], v[14]), m[14]), Rot(c[ 2], v[10]));
        m[15] = Enc(Xor(Rot(v[11], v[15]), m[15]), Rot(c[ 3], v[11]));

        v[ 0] = Xor(Xor(m[ 0], m[ 4]), Xor(m[ 8], m[12]));
        v[ 1] = Xor(Xor(m[ 1], m[ 5]), Xor(m[ 9], m[13]));
        v[ 2] = Xor(Xor(m[ 2], m[ 6]), Xor(m[10], m[14]));
        v[ 3] = Xor(Xor(m[ 3], m[ 7]), Xor(m[11], m[15]));

        m[ 0] = Enc(Xor(Rot(v[ 1], v[ 0]), v[ 0]), v[ 3]);
        m[ 1] = Enc(Xor(Rot(v[ 2], v[ 1]), v[ 1]), v[ 0]);
        m[ 2] = Enc(Xor(Rot(v[ 3], v[ 2]), v[ 2]), v[ 1]);
        m[ 3] = Enc(Xor(Rot(v[ 0], v[ 3]), v[ 3]), v[ 2]);

        m[ 0] = Enc(Xor(Rot(v[ 2], v[ 0]), m[ 0]), v[ 1]);
        m[ 1] = Enc(Xor(Rot(v[ 3], v[ 1]), m[ 1]), v[ 2]);
        m[ 2] = Enc(Xor(Rot(v[ 0], v[ 2]), m[ 2]), v[ 3]);
        m[ 3] = Enc(Xor(Rot(v[ 1], v[ 3]), m[ 3]), v[ 0]);

        m[ 0] = Enc(Xor(Rot(v[ 3], v[ 0]), m[ 0]), v[ 2]);
        m[ 1] = Enc(Xor(Rot(v[ 0], v[ 1]), m[ 1]), v[ 3]);
        m[ 2] = Enc(Xor(Rot(v[ 1], v[ 2]), m[ 2]), v[ 0]);
        m[ 3] = Enc(Xor(Rot(v[ 2], v[ 3]), m[ 3]), v[ 1]);

        v[ 0] = Xor(Xor(m[ 0], m[ 1]), Xor(m[ 2], m[ 3]));

        return v[ 0];
    }
#endif

    // Returns true if the implementation produces the expected hash outputs for
    // test cases. The verification algorithm aims to exercise every branch in
    // the reference implementation code.
    static bool VerifyImplementation() {
        // authoritative test vector
        static const char test[] = "\x1B\x1E\xE0\x82\xCB\xB5\x89\xAD\x2F\x56\xC8\x2A\xFE\xE9\xA3\x6F";

        // hashing a key from every size in 0..512 bytes should be sufficient to
        // execute every branch in most reasonable implementations
        char data[512];
        for (int i = 0; i < 512; ++i)
            data[i] = i & 0xFF;

        Scalar hash = Hash(data, 0);
        for (int i = 1; i <= 512; ++i)
            hash = Xor(hash, Hash(data, i));

        char byte[16];
        memcpy(&byte, &hash, sizeof(hash));
        return memcmp(&byte, &test, 16) == 0;
    }
};
