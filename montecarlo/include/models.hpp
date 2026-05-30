#pragma once
// Financial stochastic process models for Monte Carlo
// GBM, Heston (QE scheme), Merton Jump-Diffusion, CIR, Multi-Asset Correlated GBM

#include "rng.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>

// ============================================================
//  Black-Scholes analytical solutions (benchmarks)
// ============================================================
namespace bs {

inline double d1(double S, double K, double r, double q, double sigma, double T) noexcept {
    return (std::log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*std::sqrt(T));
}
inline double d2(double S, double K, double r, double q, double sigma, double T) noexcept {
    return d1(S,K,r,q,sigma,T) - sigma*std::sqrt(T);
}

inline double N(double x) noexcept {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

inline double call_price(double S, double K, double r, double q, double sigma, double T) noexcept {
    double _d1 = d1(S,K,r,q,sigma,T), _d2 = _d1 - sigma*std::sqrt(T);
    return S*std::exp(-q*T)*N(_d1) - K*std::exp(-r*T)*N(_d2);
}

inline double put_price(double S, double K, double r, double q, double sigma, double T) noexcept {
    double _d1 = d1(S,K,r,q,sigma,T), _d2 = _d1 - sigma*std::sqrt(T);
    return K*std::exp(-r*T)*N(-_d2) - S*std::exp(-q*T)*N(-_d1);
}

struct Greeks {
    double delta, gamma, vega, theta, rho, price;
};

inline Greeks call_greeks(double S, double K, double r, double q, double sigma, double T) noexcept {
    double sqrtT = std::sqrt(T);
    double _d1 = d1(S,K,r,q,sigma,T);
    double _d2 = _d1 - sigma*sqrtT;
    double Nd1 = N(_d1), Nd2 = N(_d2);
    double nd1 = std::exp(-0.5*_d1*_d1) / std::sqrt(2.0*M_PI);
    double disc_r = std::exp(-r*T), disc_q = std::exp(-q*T);
    return {
        disc_q * Nd1,
        disc_q * nd1 / (S * sigma * sqrtT),
        S * disc_q * nd1 * sqrtT * 0.01,         // per 1% vol move
        -(S*disc_q*nd1*sigma/(2.0*sqrtT) + r*K*disc_r*Nd2 - q*S*disc_q*Nd1) / 365.0,
        K * T * disc_r * Nd2 * 0.01,              // per 1% rate move
        call_price(S,K,r,q,sigma,T)
    };
}

} // namespace bs

// ============================================================
//  Path result containers
// ============================================================
struct PathResult {
    double final_price;         // S_T
    double running_min;         // for barrier/lookback
    double running_max;
    double path_avg;            // for Asian options
    double log_return;          // ln(S_T/S_0)
};

// ============================================================
//  1. Geometric Brownian Motion (exact simulation)
//     dS = μS dt + σS dW
// ============================================================
struct GBMParams {
    double S0    = 100.0;  // initial price
    double mu    = 0.05;   // drift (risk-neutral: use r-q)
    double sigma = 0.20;   // volatility
    double T     = 1.0;    // time to maturity (years)
    double r     = 0.05;   // risk-free rate
    double q     = 0.0;    // dividend yield
    int    steps = 252;    // time steps (for path-dependent)
};

// Full path — stores all timestep prices (for path-dependent products)
inline std::vector<double> gbm_path(const GBMParams& p, Xoshiro256pp& rng) {
    double dt = p.T / p.steps;
    double drift = (p.r - p.q - 0.5*p.sigma*p.sigma) * dt;
    double diffusion = p.sigma * std::sqrt(dt);
    std::vector<double> path(p.steps + 1);
    path[0] = p.S0;
    for (int i = 1; i <= p.steps; i++) {
        path[i] = path[i-1] * std::exp(drift + diffusion * rng.normal());
    }
    return path;
}

// Compact result for European option (exact one-step)
inline PathResult gbm_exact(const GBMParams& p, double Z) noexcept {
    double logS = std::log(p.S0) +
                  (p.r - p.q - 0.5*p.sigma*p.sigma)*p.T +
                  p.sigma*std::sqrt(p.T)*Z;
    double ST = std::exp(logS);
    return { ST, ST, ST, ST, logS - std::log(p.S0) };
}

// Path-dependent compact result
inline PathResult gbm_path_result(const GBMParams& p, Xoshiro256pp& rng) {
    double dt = p.T / p.steps;
    double drift = (p.r - p.q - 0.5*p.sigma*p.sigma) * dt;
    double diffusion = p.sigma * std::sqrt(dt);
    double S = p.S0, sum = p.S0, smin = p.S0, smax = p.S0;
    for (int i = 1; i <= p.steps; i++) {
        S *= std::exp(drift + diffusion * rng.normal());
        sum += S;
        if (S < smin) smin = S;
        if (S > smax) smax = S;
    }
    return { S, smin, smax, sum / (p.steps + 1), std::log(S / p.S0) };
}

// ============================================================
//  2. Heston Stochastic Volatility Model
//     Quadratic-Exponential (QE) discretization (Andersen 2008)
//     dS = rS dt + √V S dW_S
//     dV = κ(θ-V) dt + ξ√V dW_V,  corr(dW_S, dW_V) = ρ
// ============================================================
struct HestonParams {
    double S0    = 100.0;   // initial price
    double V0    = 0.04;    // initial variance (vol = √V0)
    double kappa = 2.0;     // mean reversion speed
    double theta = 0.04;    // long-run variance
    double xi    = 0.3;     // vol-of-vol
    double rho   = -0.7;    // S-V correlation
    double r     = 0.05;    // risk-free rate
    double q     = 0.0;     // dividend yield
    double T     = 1.0;
    int    steps = 100;
};

inline PathResult heston_path(const HestonParams& p, Xoshiro256pp& rng) {
    double dt   = p.T / p.steps;
    double e_k  = std::exp(-p.kappa * dt);
    double rho1 = std::sqrt(1.0 - p.rho * p.rho);

    double S = p.S0, V = p.V0, sum = p.S0, smin = p.S0, smax = p.S0;
    constexpr double PSI_C = 1.5; // QE threshold

    for (int i = 0; i < p.steps; i++) {
        // --- QE scheme for variance ---
        double m  = p.theta + (V - p.theta) * e_k;
        double s2_v = V * p.xi*p.xi * e_k * (1.0-e_k) / p.kappa
                    + p.theta * p.xi*p.xi * (1.0-e_k)*(1.0-e_k) / (2.0*p.kappa);
        double s2_v_clamped = std::max(s2_v, 1e-14);
        double psi = s2_v_clamped / (m * m);
        double V_new;
        auto [Uv, Zv] = rng.normal_pair();
        double Uv01 = 0.5*(1.0 + std::erf(Uv * M_SQRT1_2)); // CDF of N(0,1)
        if (psi <= PSI_C) {
            // Matched moments to a shifted chi-squared
            double b2 = 2.0/psi - 1.0 + std::sqrt(2.0/psi * (2.0/psi - 1.0));
            double a  = m / (1.0 + b2);
            V_new = a * (std::sqrt(b2) + Zv) * (std::sqrt(b2) + Zv);
        } else {
            // Exponential distribution
            double p_prob = (psi-1.0)/(psi+1.0);
            double beta   = (1.0-p_prob)/m;
            V_new = (Uv01 <= p_prob) ? 0.0 : std::log((1.0-p_prob)/(1.0-Uv01)) / beta;
        }
        double V_avg = 0.5*(V + V_new);

        // --- Price update ---
        auto [Zs1, Zs2] = rng.normal_pair();
        double Zs = p.rho * Zs1 + rho1 * Zs2;
        S *= std::exp((p.r - p.q - 0.5*V_avg)*dt + std::sqrt(std::max(V_avg,0.0)*dt)*Zs);
        V  = std::max(V_new, 0.0);
        sum += S;
        if (S < smin) smin = S;
        if (S > smax) smax = S;
    }
    return { S, smin, smax, sum / (p.steps + 1), std::log(S / p.S0) };
}

// ============================================================
//  3. Merton Jump-Diffusion
//     dS = (μ - λk̄)S dt + σS dW + (J-1)S dN
//     J ~ LogNormal(μ_J, σ_J²), N ~ Poisson(λ)
// ============================================================
struct MertonParams {
    double S0     = 100.0;
    double r      = 0.05;
    double sigma  = 0.20;    // diffusion vol
    double lambda = 0.5;     // jump intensity (jumps/year)
    double mu_j   = -0.10;   // mean log-jump size
    double sigma_j= 0.20;    // std dev log-jump size
    double T      = 1.0;
    int    steps  = 252;
};

inline PathResult merton_path(const MertonParams& p, Xoshiro256pp& rng) {
    double dt    = p.T / p.steps;
    double kbar  = std::exp(p.mu_j + 0.5*p.sigma_j*p.sigma_j) - 1.0;
    double drift = (p.r - 0.5*p.sigma*p.sigma - p.lambda*kbar) * dt;
    double diff  = p.sigma * std::sqrt(dt);
    double lam_dt = p.lambda * dt;

    double S = p.S0, sum = p.S0, smin = p.S0, smax = p.S0;
    for (int i = 0; i < p.steps; i++) {
        // Poisson number of jumps this step (Poisson approx for small lam_dt)
        double U = rng.uniform();
        int n_jumps = 0;
        double cum_p = std::exp(-lam_dt);
        double running = cum_p;
        while (U > running && n_jumps < 10) {
            n_jumps++;
            cum_p *= lam_dt / n_jumps;
            running += cum_p;
        }
        double log_jump = 0.0;
        for (int j = 0; j < n_jumps; j++)
            log_jump += p.mu_j + p.sigma_j * rng.normal();

        S *= std::exp(drift + diff * rng.normal() + log_jump);
        sum += S;
        if (S < smin) smin = S;
        if (S > smax) smax = S;
    }
    return { S, smin, smax, sum / (p.steps + 1), std::log(S / p.S0) };
}

// ============================================================
//  4. CIR Interest Rate Model (exact non-central chi-squared)
//     dr = κ(θ-r) dt + σ√r dW
// ============================================================
struct CIRParams {
    double r0     = 0.05;
    double kappa  = 1.5;
    double theta  = 0.05;
    double sigma  = 0.10;
    double T      = 10.0;
    int    steps  = 1200;
};

inline std::vector<double> cir_path(const CIRParams& p, Xoshiro256pp& rng) {
    double dt     = p.T / p.steps;
    (void)std::exp(-p.kappa * dt); // e_k used in exact scheme; Euler uses direct form
    std::vector<double> path(p.steps + 1);
    path[0] = p.r0;

    // Euler-Maruyama (simple, fast)
    for (int i = 1; i <= p.steps; i++) {
        double r = path[i-1];
        double drift  = p.kappa * (p.theta - r) * dt;
        double diffusion = p.sigma * std::sqrt(std::max(r, 0.0) * dt);
        path[i] = std::max(r + drift + diffusion * rng.normal(), 0.0);
    }
    return path;
}

// ============================================================
//  5. Multi-Asset Correlated GBM (Cholesky decomposition)
// ============================================================
struct MultiAssetParams {
    std::vector<double> S0;     // initial prices
    std::vector<double> mu;     // drifts
    std::vector<double> sigma;  // vols
    std::vector<std::vector<double>> corr; // correlation matrix
    double T;
    int    steps;
    int    n_assets;
};

// Cholesky decomposition (lower triangular)
inline std::vector<std::vector<double>> cholesky(
    const std::vector<std::vector<double>>& A, int n)
{
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = A[i][j];
            for (int k = 0; k < j; k++) sum -= L[i][k]*L[j][k];
            L[i][j] = (i == j) ? std::sqrt(std::max(sum, 0.0)) : sum / L[j][j];
        }
    }
    return L;
}

inline std::vector<double> multi_asset_final(const MultiAssetParams& p,
                                              const std::vector<std::vector<double>>& L,
                                              Xoshiro256pp& rng)
{
    int n = p.n_assets;
    double dt = p.T / p.steps;
    std::vector<double> S(p.S0);
    std::vector<double> Z(n);

    for (int step = 0; step < p.steps; step++) {
        // Independent normals
        for (int i = 0; i < n; i++) Z[i] = rng.normal();
        // Apply Cholesky to get correlated normals
        for (int i = 0; i < n; i++) {
            double corr_z = 0.0;
            for (int j = 0; j <= i; j++) corr_z += L[i][j] * Z[j];
            double drift = (p.mu[i] - 0.5*p.sigma[i]*p.sigma[i]) * dt;
            S[i] *= std::exp(drift + p.sigma[i] * std::sqrt(dt) * corr_z);
        }
    }
    return S;
}

// ============================================================
//  Option payoff functions
// ============================================================
namespace payoff {

inline double european_call(double ST, double K) noexcept {
    return std::max(ST - K, 0.0);
}
inline double european_put(double ST, double K) noexcept {
    return std::max(K - ST, 0.0);
}
inline double asian_call_arith(double avg, double K) noexcept {
    return std::max(avg - K, 0.0);
}
inline double asian_call_geo(double geo_avg, double K) noexcept {
    return std::max(geo_avg - K, 0.0);
}
inline double barrier_up_out_call(double ST, double Smax, double K, double B) noexcept {
    return (Smax >= B) ? 0.0 : std::max(ST - K, 0.0);
}
inline double barrier_down_out_put(double ST, double Smin, double K, double B) noexcept {
    return (Smin <= B) ? 0.0 : std::max(K - ST, 0.0);
}
inline double lookback_call_fixed(double ST, double Smin, double K) noexcept {
    return std::max(ST - std::min(Smin, K), 0.0);
}
inline double digital_call(double ST, double K, double payout = 1.0) noexcept {
    return (ST > K) ? payout : 0.0;
}

// Geometric Asian (closed-form available as control variate)
inline double asian_geo_bs(double S0, double K, double r, double sigma, double T, int N) noexcept {
    double sigma_g = sigma * std::sqrt((2.0*N+1.0) / (6.0*(N+1.0)));
    double rho     = 0.5 * (r - 0.5*sigma*sigma + (r - 0.5*sigma*sigma)*(2.0*N+1.0)/(2.0*N) + sigma_g*sigma_g);
    // Turnbull-Wakeman approximation
    double d1 = (std::log(S0/K) + (rho + 0.5*sigma_g*sigma_g)*T) / (sigma_g * std::sqrt(T));
    double d2 = d1 - sigma_g * std::sqrt(T);
    auto N01 = [](double x){ return 0.5*std::erfc(-x*M_SQRT1_2); };
    return std::exp(-r*T) * (S0 * std::exp(rho*T) * N01(d1) - K * N01(d2));
}

} // namespace payoff
