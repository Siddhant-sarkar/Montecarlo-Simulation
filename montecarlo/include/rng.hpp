#pragma once
// Ultra-fast random number generators for Monte Carlo simulation
// Includes: Xoshiro256++, PCG64, Halton quasi-Monte Carlo, Box-Muller, BSM norminv

#include <cstdint>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
//  Xoshiro256++  –  256-bit state, 64-bit output
//  Period: 2^256-1.  Passes all known statistical tests.
//  ~1.3 ns/call (single core, no SIMD)
// ============================================================
class Xoshiro256pp {
    std::array<uint64_t, 4> s_;

    static uint64_t rotl(uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }
    static uint64_t splitmix64(uint64_t& x) noexcept {
        x += UINT64_C(0x9e3779b97f4a7c15);
        uint64_t z = x;
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }

public:
    using result_type = uint64_t;
    static constexpr uint64_t min() noexcept { return 0; }
    static constexpr uint64_t max() noexcept { return UINT64_MAX; }

    explicit Xoshiro256pp(uint64_t seed = 0xdeadbeefcafe42ULL) noexcept {
        s_[0] = splitmix64(seed);
        s_[1] = splitmix64(seed);
        s_[2] = splitmix64(seed);
        s_[3] = splitmix64(seed);
    }

    uint64_t operator()() noexcept {
        const uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];  s_[3] ^= s_[1];
        s_[1] ^= s_[2];  s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform [0,1) with 53-bit mantissa precision
    double uniform() noexcept {
        return static_cast<double>(operator()() >> 11) *
               (1.0 / static_cast<double>(UINT64_C(1) << 53));
    }

    // Box-Muller: returns two independent N(0,1) values
    std::pair<double, double> normal_pair() noexcept {
        double u1;
        do { u1 = uniform(); } while (u1 < 1e-300);
        double u2 = uniform();
        double r = std::sqrt(-2.0 * std::log(u1));
        double a = 2.0 * M_PI * u2;
        return { r * std::cos(a), r * std::sin(a) };
    }

    // Single standard normal
    double normal() noexcept { return normal_pair().first; }

    // Jump ahead 2^128 steps — creates independent parallel streams
    void long_jump() noexcept {
        constexpr uint64_t J[4] = {
            0x76e15d3efefdcbbfULL, 0xc5004e441c522fb3ULL,
            0x77710069854ee241ULL, 0x39109bb02acbe635ULL
        };
        uint64_t t0=0, t1=0, t2=0, t3=0;
        for (auto j : J) {
            for (int b = 0; b < 64; b++) {
                if (j & (UINT64_C(1) << b)) {
                    t0 ^= s_[0]; t1 ^= s_[1];
                    t2 ^= s_[2]; t3 ^= s_[3];
                }
                operator()();
            }
        }
        s_[0]=t0; s_[1]=t1; s_[2]=t2; s_[3]=t3;
    }
};

// ============================================================
//  Halton Low-Discrepancy Sequence (quasi-Monte Carlo)
//  Uses first D prime bases. Up to 40 dimensions.
// ============================================================
class HaltonSequence {
    static constexpr uint32_t PRIMES[40] = {
        2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,
        53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,
        127,131,137,139,149,151,157,163,167,173
    };

    std::vector<uint32_t> bases_;
    std::vector<uint64_t> idx_;

    static double van_der_corput(uint64_t n, uint32_t b) noexcept {
        double result = 0.0, f = 1.0;
        while (n > 0) { f /= b; result += f * (n % b); n /= b; }
        return result;
    }

public:
    explicit HaltonSequence(int dim) : idx_(dim, 1) {
        for (int i = 0; i < std::min(dim, 40); i++)
            bases_.push_back(PRIMES[i]);
    }

    // Returns next dim-dimensional point in [0,1)^dim
    std::vector<double> next() {
        std::vector<double> v(bases_.size());
        for (size_t i = 0; i < bases_.size(); i++)
            v[i] = van_der_corput(idx_[i]++, bases_[i]);
        return v;
    }

    void reset() { std::fill(idx_.begin(), idx_.end(), 1); }
};

// ============================================================
//  Normal inverse CDF — Beasley-Springer-Moro algorithm
//  Max abs error ~1.5e-9 over (0,1)
// ============================================================
inline double norminv(double p) noexcept {
    static const double A[] = { 2.50662823884, -18.61500062529,
                                  41.39119773534, -25.44106049637 };
    static const double B[] = { -8.47351093090, 23.08336743743,
                                  -21.06224101826, 3.13082909833 };
    static const double C[] = {
        0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
        0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
        0.0000321767881768, 0.0000002888167364, 0.0000003960315187
    };
    double y = p - 0.5;
    if (std::abs(y) < 0.42) {
        double r = y * y;
        return y * (((A[3]*r+A[2])*r+A[1])*r+A[0]) /
                   ((((B[3]*r+B[2])*r+B[1])*r+B[0])*r+1.0);
    }
    double r = (y > 0.0) ? std::log(-std::log(1.0-p)) : std::log(-std::log(p));
    double x = C[0]+r*(C[1]+r*(C[2]+r*(C[3]+r*(C[4]+r*(C[5]+r*(C[6]+r*(C[7]+r*C[8])))))));
    return (y < 0.0) ? -x : x;
}

// ============================================================
//  Batch Gaussian generation helpers
// ============================================================

// Generate n Gaussians into pre-allocated array using Box-Muller
inline void gen_gaussian_batch(double* out, size_t n, Xoshiro256pp& rng) noexcept {
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        auto [z1, z2] = rng.normal_pair();
        out[i] = z1; out[i+1] = z2;
    }
    if (i < n) out[i] = rng.normal();
}

// Antithetic: first half standard, second half mirrored
inline void gen_gaussian_antithetic(double* out, size_t n, Xoshiro256pp& rng) noexcept {
    size_t half = n / 2;
    for (size_t i = 0; i < half; i++) {
        out[i] = rng.normal();
        out[i + half] = -out[i];
    }
    if (n & 1) out[n-1] = rng.normal();
}

// Quasi-MC Gaussians from Halton sequence via norminv transform
inline void gen_gaussian_halton(double* out, size_t n, HaltonSequence& seq) noexcept {
    for (size_t i = 0; i < n; i++) {
        auto v = seq.next();
        out[i] = norminv(0.001 + v[0] * 0.998); // avoid extreme tails
    }
}
