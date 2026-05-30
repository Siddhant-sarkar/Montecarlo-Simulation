// =======================================================================
//  Advanced Monte Carlo Simulation Engine
//  Models: GBM, Heston (QE), Merton Jump-Diffusion, CIR, Multi-Asset
//  Features: Multi-threading, Variance Reduction, Quasi-MC, Full Metrics
// =======================================================================

#include "../include/rng.hpp"
#include "../include/models.hpp"
#include "../include/statistics.hpp"
#include "../include/visualizer.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <string>
#include <sstream>
#include <cstring>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ============================================================
//  Terminal color/formatting
// ============================================================
namespace term {
    const char* RESET  = "\033[0m";
    const char* BOLD   = "\033[1m";
    const char* DIM    = "\033[2m";
    const char* BLUE   = "\033[94m";
    const char* GREEN  = "\033[92m";
    const char* RED    = "\033[91m";
    const char* YELLOW = "\033[93m";
    const char* CYAN   = "\033[96m";
    const char* PURPLE = "\033[95m";
    const char* WHITE  = "\033[97m";
}

static void print_banner() {
    std::cout << "\n"
              << term::BLUE << "╔══════════════════════════════════════════════════════════════╗\n"
              << "║    " << term::BOLD << term::WHITE
              << "C++ Advanced Monte Carlo Simulation Engine" << term::RESET << term::BLUE
              << "              ║\n"
              << "║    Models: GBM · Heston (QE) · Merton Jump · CIR · Portfolio ║\n"
              << "║    Features: Multi-thread · VR · Quasi-MC · Full Analytics    ║\n"
              << "╚══════════════════════════════════════════════════════════════╝"
              << term::RESET << "\n\n";
}

static void print_section(const std::string& title) {
    int w = 66;
    int pad = (w - static_cast<int>(title.size()) - 4) / 2;
    std::string line(w, '-');
    std::cout << "\n" << term::CYAN;
    std::cout << line << "\n";
    std::cout << std::string(pad, ' ') << "  " << title << "\n";
    std::cout << line << term::RESET << "\n";
}

static void print_kv(const std::string& key, const std::string& val,
                     const char* color = nullptr)
{
    std::cout << "  " << term::DIM << std::left << std::setw(30) << key
              << term::RESET << " ";
    if (color) std::cout << color;
    std::cout << val;
    if (color) std::cout << term::RESET;
    std::cout << "\n";
}

// ============================================================
//  Thread-pool based MC runner
//  Each thread gets its own Xoshiro256++ instance (jumped ahead)
// ============================================================
template<typename Fn>
static std::vector<double> run_mt(size_t total_paths, Fn path_fn) {
    unsigned int nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<double> results(total_paths);

    std::vector<std::thread> threads;
    threads.reserve(nt);

    // Stride per thread
    size_t stride = (total_paths + nt - 1) / nt;

    for (unsigned int t = 0; t < nt; t++) {
        size_t begin = t * stride;
        size_t end   = std::min(begin + stride, total_paths);
        if (begin >= end) break;

        threads.emplace_back([&results, begin, end, t, &path_fn]() {
            // Each thread: independent Xoshiro stream (jump ahead t times)
            Xoshiro256pp rng(0x8f4b3d2e1a7c6095ULL + t * 17 + 1);
            for (size_t i = 1; i < t; i++) rng.long_jump();
            for (size_t i = begin; i < end; i++)
                results[i] = path_fn(rng);
        });
    }
    for (auto& th : threads) th.join();
    return results;
}

// ============================================================
//  ASCII histogram (quick terminal preview)
// ============================================================
static void print_ascii_hist(const std::vector<double>& data, const std::string& title,
                              int bins = 20, int bar_w = 40)
{
    auto sorted = data;
    std::sort(sorted.begin(), sorted.end());
    double lo = percentile(sorted, 0.01);
    double hi = percentile(sorted, 0.99);
    double w  = (hi - lo) / bins;
    if (w <= 0) return;

    std::vector<int> counts(bins, 0);
    for (double x : data) {
        if (x < lo || x > hi) continue;
        int b = std::clamp(static_cast<int>((x-lo)/w), 0, bins-1);
        counts[b]++;
    }
    int mx = *std::max_element(counts.begin(), counts.end());

    std::cout << "\n  " << term::BOLD << title << term::RESET << "\n";
    for (int i = 0; i < bins; i++) {
        double center = lo + (i + 0.5) * w;
        int    filled = (counts[i] * bar_w) / mx;
        std::cout << "  " << std::right << std::setw(8) << std::fixed
                  << std::setprecision(2) << center << " │";
        std::cout << term::BLUE;
        for (int f = 0; f < filled; f++) std::cout << '#';
        std::cout << term::RESET;
        std::cout << std::string(bar_w - filled, ' ')
                  << "│ " << counts[i] << "\n";
    }
}

// ============================================================
//  Simulation 1: GBM European Call (10 million paths)
// ============================================================
struct EuropeanResult {
    double price;
    double std_error;
    double ci_lo, ci_hi;
    double elapsed_ms;
    std::vector<double> payoffs;
};

static EuropeanResult run_european_call(
    const GBMParams& p, double K, size_t N_paths, bool antithetic = false)
{
    auto t0 = Clock::now();
    double disc = std::exp(-p.r * p.T);

    std::vector<double> payoffs = run_mt(N_paths, [&](Xoshiro256pp& rng) -> double {
        if (antithetic) {
            auto [z1, z2] = rng.normal_pair();
            double st1 = p.S0 * std::exp((p.r-0.5*p.sigma*p.sigma)*p.T + p.sigma*std::sqrt(p.T)*z1);
            double st2 = p.S0 * std::exp((p.r-0.5*p.sigma*p.sigma)*p.T + p.sigma*std::sqrt(p.T)*(-z1));
            return disc * 0.5 * (payoff::european_call(st1, K) + payoff::european_call(st2, K));
        } else {
            auto r = gbm_exact(p, rng.normal());
            return disc * payoff::european_call(r.final_price, K);
        }
    });

    double elapsed = Ms(Clock::now() - t0).count();
    Stats st = compute_stats(payoffs);
    return { st.mean, st.std_error, st.ci_lower, st.ci_upper, elapsed, std::move(payoffs) };
}

// ============================================================
//  Simulation 2: Asian Arithmetic Call with control variate
// ============================================================
static EuropeanResult run_asian_call(const GBMParams& p, double K, size_t N_paths) {
    auto t0 = Clock::now();
    double disc = std::exp(-p.r * p.T);

    // Geometric Asian closed-form (control variate)
    double geo_analytic = payoff::asian_geo_bs(p.S0, K, p.r, p.sigma, p.T, p.steps);

    std::vector<double> arith_payoffs(N_paths), geo_payoffs(N_paths);

    unsigned int nt = std::max(1u, std::thread::hardware_concurrency());
    size_t stride = (N_paths + nt - 1) / nt;
    std::vector<std::thread> threads;

    for (unsigned int t = 0; t < nt; t++) {
        size_t beg = t * stride, end = std::min(beg + stride, N_paths);
        if (beg >= end) break;
        threads.emplace_back([&, beg, end, t]() {
            Xoshiro256pp rng(0xabcdef1234567890ULL + t * 31 + 1);
            for (size_t i = 1; i < t; i++) rng.long_jump();

            double dt = p.T / p.steps;
            double drift = (p.r - 0.5*p.sigma*p.sigma) * dt;
            double diff  = p.sigma * std::sqrt(dt);

            for (size_t i = beg; i < end; i++) {
                double S = p.S0, arith_sum = p.S0, log_sum = std::log(p.S0);
                for (int step = 1; step <= p.steps; step++) {
                    S *= std::exp(drift + diff * rng.normal());
                    arith_sum += S;
                    log_sum   += std::log(S);
                }
                double arith_avg = arith_sum / (p.steps + 1);
                double geo_avg   = std::exp(log_sum / (p.steps + 1));
                arith_payoffs[i] = disc * payoff::asian_call_arith(arith_avg, K);
                geo_payoffs[i]   = disc * payoff::asian_call_geo(geo_avg, K);
            }
        });
    }
    for (auto& th : threads) th.join();

    // Control variate: X_cv = X - beta*(Y - E[Y])
    // Optimal beta = Cov(X,Y)/Var(Y)
    double mean_g = 0, mean_a = 0;
    for (size_t i = 0; i < N_paths; i++) { mean_g += geo_payoffs[i]; mean_a += arith_payoffs[i]; }
    mean_g /= N_paths; mean_a /= N_paths;

    double cov_xy = 0, var_y = 0;
    for (size_t i = 0; i < N_paths; i++) {
        cov_xy += (arith_payoffs[i]-mean_a) * (geo_payoffs[i]-mean_g);
        var_y  += (geo_payoffs[i]-mean_g) * (geo_payoffs[i]-mean_g);
    }
    double beta = (var_y > 0) ? cov_xy / var_y : 0.0;

    std::vector<double> cv_payoffs(N_paths);
    for (size_t i = 0; i < N_paths; i++)
        cv_payoffs[i] = arith_payoffs[i] - beta * (geo_payoffs[i] - geo_analytic);

    double elapsed = Ms(Clock::now() - t0).count();
    Stats st = compute_stats(cv_payoffs);
    return { st.mean, st.std_error, st.ci_lower, st.ci_upper, elapsed, std::move(cv_payoffs) };
}

// ============================================================
//  Simulation 3: Heston Stochastic Volatility
// ============================================================
static EuropeanResult run_heston_call(const HestonParams& p, double K, size_t N_paths) {
    auto t0 = Clock::now();
    double disc = std::exp(-p.r * p.T);

    std::vector<double> payoffs = run_mt(N_paths, [&](Xoshiro256pp& rng) -> double {
        auto res = heston_path(p, rng);
        return disc * payoff::european_call(res.final_price, K);
    });

    double elapsed = Ms(Clock::now() - t0).count();
    Stats st = compute_stats(payoffs);
    return { st.mean, st.std_error, st.ci_lower, st.ci_upper, elapsed, std::move(payoffs) };
}

// ============================================================
//  Simulation 4: Barrier Options
// ============================================================
struct BarrierResult {
    double up_out_call_price, down_out_put_price;
    double uo_se, dop_se, elapsed_ms;
    std::vector<double> uo_payoffs, dop_payoffs;
};

static BarrierResult run_barrier(const GBMParams& p, double K,
                                  double B_up, double B_down, size_t N_paths)
{
    auto t0 = Clock::now();
    double disc = std::exp(-p.r * p.T);

    std::vector<double> uo(N_paths), dop(N_paths);
    unsigned int nt = std::max(1u, std::thread::hardware_concurrency());
    size_t stride = (N_paths + nt - 1) / nt;
    std::vector<std::thread> threads;

    for (unsigned int t = 0; t < nt; t++) {
        size_t beg = t*stride, end = std::min(beg+stride, N_paths);
        if (beg >= end) break;
        threads.emplace_back([&, beg, end, t]() {
            Xoshiro256pp rng(0xfedcba9876543210ULL + t * 19 + 3);
            for (size_t i = 1; i < t; i++) rng.long_jump();
            for (size_t i = beg; i < end; i++) {
                auto res = gbm_path_result(p, rng);
                uo[i]  = disc * payoff::barrier_up_out_call(res.final_price, res.running_max, K, B_up);
                dop[i] = disc * payoff::barrier_down_out_put(res.final_price, res.running_min, K, B_down);
            }
        });
    }
    for (auto& th : threads) th.join();

    double elapsed = Ms(Clock::now() - t0).count();
    Stats su = compute_stats(uo), sd = compute_stats(dop);
    return { su.mean, sd.mean, su.std_error, sd.std_error, elapsed,
             std::move(uo), std::move(dop) };
}

// ============================================================
//  Simulation 5: Portfolio VaR (10-asset correlated GBM)
// ============================================================
struct PortfolioResult {
    std::vector<double> pnl;
    Stats stats;
    double elapsed_ms;
};

static PortfolioResult run_portfolio_var(size_t N_paths) {
    const int n = 10;
    MultiAssetParams mp;
    mp.n_assets = n;
    mp.T = 1.0 / 252.0;  // 1 trading day
    mp.steps = 1;

    // Equal weights, varied vols
    mp.S0    = { 100,150,80,200,120,90,180,110,140,160 };
    mp.mu    = { 0.08,0.10,0.06,0.12,0.09,0.07,0.11,0.08,0.10,0.09 };
    mp.sigma = { 0.20,0.25,0.18,0.30,0.22,0.16,0.28,0.20,0.24,0.22 };

    // Correlation matrix (realistic equities)
    mp.corr.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        mp.corr[i][i] = 1.0;
        for (int j = 0; j < i; j++) {
            // Sector correlations: 0.3-0.6 range
            mp.corr[i][j] = mp.corr[j][i] = 0.35 + 0.25 * ((i+j) % 3) / 3.0;
        }
    }

    auto L = cholesky(mp.corr, n);

    // Portfolio weights (equal weight)
    double total_value = 0;
    for (double s : mp.S0) total_value += s;
    std::vector<double> weights(n);
    for (int i = 0; i < n; i++) weights[i] = mp.S0[i] / total_value;

    auto t0 = Clock::now();
    std::vector<double> pnl = run_mt(N_paths, [&](Xoshiro256pp& rng) -> double {
        auto S_final = multi_asset_final(mp, L, rng);
        double port_return = 0;
        for (int i = 0; i < n; i++)
            port_return += weights[i] * (S_final[i] - mp.S0[i]) / mp.S0[i];
        return port_return * total_value;
    });

    double elapsed = Ms(Clock::now() - t0).count();
    Stats st = compute_stats(pnl, true);
    return { std::move(pnl), st, elapsed };
}

// ============================================================
//  Simulation 6: Variance reduction comparison
// ============================================================
struct VRComparison {
    struct Method {
        std::string name;
        double price;
        double std_error;
        double elapsed_ms;
        std::vector<double> payoffs;
    };
    std::vector<Method> methods;
};

static VRComparison run_vr_comparison(const GBMParams& p, double K, size_t N = 500'000) {
    VRComparison result;
    double disc = std::exp(-p.r * p.T);

    // 1. Plain MC
    {
        auto t0 = Clock::now();
        auto payoffs = run_mt(N, [&](Xoshiro256pp& rng) -> double {
            return disc * payoff::european_call(gbm_exact(p, rng.normal()).final_price, K);
        });
        double el = Ms(Clock::now()-t0).count();
        Stats st = compute_stats(payoffs);
        result.methods.push_back({"Plain MC", st.mean, st.std_error, el, std::move(payoffs)});
    }

    // 2. Antithetic Variates
    {
        auto t0 = Clock::now();
        auto payoffs = run_mt(N, [&](Xoshiro256pp& rng) -> double {
            auto [z1, z2] = rng.normal_pair();
            double s = p.r - 0.5*p.sigma*p.sigma;
            double st1 = p.S0 * std::exp(s*p.T + p.sigma*std::sqrt(p.T)*z1);
            double st2 = p.S0 * std::exp(s*p.T + p.sigma*std::sqrt(p.T)*(-z1));
            return disc * 0.5 * (payoff::european_call(st1, K) + payoff::european_call(st2, K));
        });
        double el = Ms(Clock::now()-t0).count();
        Stats st = compute_stats(payoffs);
        result.methods.push_back({"Antithetic", st.mean, st.std_error, el, std::move(payoffs)});
    }

    // 3. Quasi-MC (Halton)
    {
        auto t0 = Clock::now();
        HaltonSequence halton(1);
        std::vector<double> payoffs(N);
        for (size_t i = 0; i < N; i++) {
            double u  = halton.next()[0];
            double ST = p.S0 * std::exp(
                (p.r - 0.5*p.sigma*p.sigma)*p.T +
                p.sigma*std::sqrt(p.T)*norminv(0.001 + u*0.998));
            payoffs[i] = disc * payoff::european_call(ST, K);
        }
        double el = Ms(Clock::now()-t0).count();
        Stats st = compute_stats(payoffs);
        result.methods.push_back({"Quasi-MC (Halton)", st.mean, st.std_error, el, std::move(payoffs)});
    }

    // 4. Control Variate (using BS delta as CV)
    {
        auto t0 = Clock::now();
        double bs_price = bs::call_price(p.S0, K, p.r, p.q, p.sigma, p.T);
        auto payoffs = run_mt(N, [&](Xoshiro256pp& rng) -> double {
            double Z  = rng.normal();
            double ST = p.S0 * std::exp((p.r-0.5*p.sigma*p.sigma)*p.T + p.sigma*std::sqrt(p.T)*Z);
            // Control variate: E[S_T] = S0 * e^(rT)
            double cv  = disc * payoff::european_call(ST, K) - (ST*disc - p.S0*std::exp(-p.q*p.T));
            return cv + bs_price; // beta=1 approximation
        });
        double el = Ms(Clock::now()-t0).count();
        Stats st = compute_stats(payoffs);
        result.methods.push_back({"Control Variate", st.mean, st.std_error, el, std::move(payoffs)});
    }

    return result;
}

// ============================================================
//  Simulation 7: Merton Jump-Diffusion
// ============================================================
static EuropeanResult run_merton_call(const MertonParams& mp, double K, size_t N_paths) {
    auto t0 = Clock::now();
    double disc = std::exp(-mp.r * mp.T);

    std::vector<double> payoffs = run_mt(N_paths, [&](Xoshiro256pp& rng) -> double {
        auto res = merton_path(mp, rng);
        return disc * payoff::european_call(res.final_price, K);
    });

    double elapsed = Ms(Clock::now() - t0).count();
    Stats st = compute_stats(payoffs);
    return { st.mean, st.std_error, st.ci_lower, st.ci_upper, elapsed, std::move(payoffs) };
}

// ============================================================
//  Print table helper
// ============================================================
// ============================================================
//  Main
// ============================================================
int main() {
    print_banner();

    const int N_THREADS = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    std::cout << term::BOLD << "  System: " << term::RESET
              << N_THREADS << " hardware threads detected\n\n";

    // ---- Parameters ----
    GBMParams gbm_p;
    gbm_p.S0    = 100.0;
    gbm_p.mu    = 0.08;
    gbm_p.sigma = 0.20;
    gbm_p.T     = 1.0;
    gbm_p.r     = 0.05;
    gbm_p.q     = 0.02;
    gbm_p.steps = 252;

    const double K     = 100.0;  // ATM strike
    const double B_up  = 130.0;  // barrier up
    const double B_down = 80.0;  // barrier down

    HestonParams heston_p;
    heston_p.S0    = 100.0;
    heston_p.V0    = 0.04;
    heston_p.kappa = 2.0;
    heston_p.theta = 0.04;
    heston_p.xi    = 0.3;
    heston_p.rho   = -0.7;
    heston_p.r     = 0.05;
    heston_p.T     = 1.0;
    heston_p.steps = 100;

    MertonParams merton_p;
    merton_p.S0      = 100.0;
    merton_p.r       = 0.05;
    merton_p.sigma   = 0.15;
    merton_p.lambda  = 0.5;
    merton_p.mu_j    = -0.10;
    merton_p.sigma_j = 0.20;
    merton_p.T       = 1.0;
    merton_p.steps   = 252;

    CIRParams cir_p;
    cir_p.r0    = 0.05;
    cir_p.kappa = 1.5;
    cir_p.theta = 0.05;
    cir_p.sigma = 0.10;
    cir_p.T     = 10.0;
    cir_p.steps = 2520;

    ReportData report;
    report.title      = "Advanced Monte Carlo Simulation Report";
    report.n_threads  = N_THREADS;
    report.S0         = gbm_p.S0;
    report.analytical_price = bs::call_price(gbm_p.S0, K, gbm_p.r, gbm_p.q, gbm_p.sigma, gbm_p.T);
    report.analytical_greeks = bs::call_greeks(gbm_p.S0, K, gbm_p.r, gbm_p.q, gbm_p.sigma, gbm_p.T);

    auto total_start = Clock::now();

    // ==========================================================
    //  SIMULATION 1 — European Call (10M paths)
    // ==========================================================
    print_section("1. GBM European Call Option  [10 million paths]");
    std::cout << "  S0=" << gbm_p.S0 << "  K=" << K << "  r=" << gbm_p.r
              << "  q=" << gbm_p.q << "  σ=" << gbm_p.sigma << "  T=" << gbm_p.T << "y\n\n";

    const size_t N_EUROPEAN = 10'000'000;
    auto euro = run_european_call(gbm_p, K, N_EUROPEAN, false);

    print_kv("MC Price:",         fmt_price(euro.price),                  term::GREEN);
    print_kv("Black-Scholes:",    fmt_price(report.analytical_price),      term::BLUE);
    print_kv("Difference:",
             fmt_price(std::abs(euro.price - report.analytical_price), 6), term::YELLOW);
    print_kv("Std Error:",        fmt_price(euro.std_error, 6));
    print_kv("95% CI:",           "[" + fmt_price(euro.ci_lo) + ", " + fmt_price(euro.ci_hi) + "]");
    print_kv("Throughput:",       [&]() -> std::string {
        std::ostringstream ss;
        double pps = N_EUROPEAN / (euro.elapsed_ms / 1000.0);
        ss << std::fixed << std::setprecision(2) << pps/1e6 << " M paths/sec";
        return ss.str();
    }(), term::CYAN);
    print_kv("Elapsed:",          fmt_ms(euro.elapsed_ms));

    // Compute & store stats for report
    report.price_stats = compute_stats(euro.payoffs);

    // Convergence analysis
    report.convergence = convergence_analysis(euro.payoffs, 60);

    // Price distribution
    Xoshiro256pp rng_hist(42);
    std::vector<double> final_prices(100'000);
    for (auto& fp : final_prices)
        fp = gbm_exact(gbm_p, rng_hist.normal()).final_price;
    report.price_hist = make_histogram(final_prices, 80);
    report.logS_mu    = std::log(gbm_p.S0) + (gbm_p.r - gbm_p.q - 0.5*gbm_p.sigma*gbm_p.sigma)*gbm_p.T;
    report.logS_sigma = gbm_p.sigma * std::sqrt(gbm_p.T);

    // Sample paths for chart
    {
        report.time_axis.clear();
        for (int i = 0; i <= gbm_p.steps; i++)
            report.time_axis.push_back(static_cast<double>(i) / gbm_p.steps * gbm_p.T);

        Xoshiro256pp rng_paths(999);
        for (int i = 0; i < 25; i++)
            report.gbm_paths.push_back(gbm_path(gbm_p, rng_paths));
        for (int i = 0; i < 25; i++) {
            // Generate full Heston path for chart
            Xoshiro256pp rng_h(42 + i);
            double dt = heston_p.T / heston_p.steps;
            double ek = std::exp(-heston_p.kappa * dt);
            double rho1 = std::sqrt(1 - heston_p.rho*heston_p.rho);
            double S = heston_p.S0, V = heston_p.V0;
            std::vector<double> path;
            path.push_back(S);
            constexpr double PSI_C = 1.5;
            for (int step = 0; step < heston_p.steps; step++) {
                double m  = heston_p.theta + (V - heston_p.theta)*ek;
                double s2 = std::max(V*heston_p.xi*heston_p.xi*ek*(1-ek)/heston_p.kappa
                    + heston_p.theta*heston_p.xi*heston_p.xi*(1-ek)*(1-ek)/(2*heston_p.kappa), 1e-14);
                double psi = s2 / (m*m);
                double Vn;
                auto [Uv, Zv] = rng_h.normal_pair();
                double Uv01 = 0.5*(1+std::erf(Uv*M_SQRT1_2));
                if (psi <= PSI_C) {
                    double b2 = 2/psi - 1 + std::sqrt(2/psi*(2/psi-1));
                    double a  = m/(1+b2);
                    Vn = a*(std::sqrt(b2)+Zv)*(std::sqrt(b2)+Zv);
                } else {
                    double pp = (psi-1)/(psi+1), beta = (1-pp)/m;
                    Vn = (Uv01<=pp) ? 0 : std::log((1-pp)/(1-Uv01))/beta;
                }
                double Va = 0.5*(V+Vn);
                auto [Zs1, Zs2] = rng_h.normal_pair();
                double Zs = heston_p.rho*Zs1 + rho1*Zs2;
                S *= std::exp((heston_p.r-0.5*Va)*dt + std::sqrt(std::max(Va,0.0)*dt)*Zs);
                V  = std::max(Vn, 0.0);
                path.push_back(S);
            }
            // Downsample to gbm_p.steps points
            std::vector<double> dpath;
            double ratio = static_cast<double>(path.size()) / (gbm_p.steps+1);
            for (int k = 0; k <= gbm_p.steps; k++) {
                int idx = std::min(static_cast<int>(k*ratio), static_cast<int>(path.size())-1);
                dpath.push_back(path[idx]);
            }
            report.heston_paths.push_back(dpath);
        }
    }

    // ==========================================================
    //  SIMULATION 2 — Heston Model (5M paths)
    // ==========================================================
    print_section("2. Heston Stochastic Vol  [5 million paths, QE scheme]");
    std::cout << "  κ=" << heston_p.kappa << "  θ=" << heston_p.theta
              << "  ξ=" << heston_p.xi << "  ρ=" << heston_p.rho << "\n\n";

    auto heston_res = run_heston_call(heston_p, K, 5'000'000);
    print_kv("Heston MC Price:",  fmt_price(heston_res.price), term::PURPLE);
    print_kv("GBM MC Price:",     fmt_price(euro.price),       term::BLUE);
    print_kv("Std Error:",        fmt_price(heston_res.std_error, 6));
    print_kv("Elapsed:",          fmt_ms(heston_res.elapsed_ms));

    // ==========================================================
    //  SIMULATION 3 — Asian Option with Control Variate (5M paths)
    // ==========================================================
    print_section("3. Asian Arithmetic Call  [5M paths + control variate]");

    auto asian_res = run_asian_call(gbm_p, K, 5'000'000);
    double asian_geo_analytic = payoff::asian_geo_bs(gbm_p.S0, K, gbm_p.r, gbm_p.sigma, gbm_p.T, gbm_p.steps);

    print_kv("Asian (CV) Price:", fmt_price(asian_res.price),          term::GREEN);
    print_kv("Geo Asian (BS):",   fmt_price(asian_geo_analytic),        term::BLUE);
    print_kv("Std Error:",        fmt_price(asian_res.std_error, 6));
    print_kv("95% CI:",           "[" + fmt_price(asian_res.ci_lo) + ", " + fmt_price(asian_res.ci_hi) + "]");
    print_kv("Elapsed:",          fmt_ms(asian_res.elapsed_ms));

    // ==========================================================
    //  SIMULATION 4 — Barrier Options (5M paths)
    // ==========================================================
    print_section("4. Barrier Options  [5M paths]");
    std::cout << "  Up-and-out Call: K=" << K << "  B=" << B_up << "\n"
              << "  Down-and-out Put: K=" << K << "  B=" << B_down << "\n\n";

    auto barrier = run_barrier(gbm_p, K, B_up, B_down, 5'000'000);
    print_kv("Up-Out Call:",      fmt_price(barrier.up_out_call_price), term::YELLOW);
    print_kv("  Std Error:",      fmt_price(barrier.uo_se, 6));
    print_kv("Down-Out Put:",     fmt_price(barrier.down_out_put_price),term::YELLOW);
    print_kv("  Std Error:",      fmt_price(barrier.dop_se, 6));
    print_kv("Elapsed:",          fmt_ms(barrier.elapsed_ms));

    // ==========================================================
    //  SIMULATION 5 — Merton Jump-Diffusion (5M paths)
    // ==========================================================
    print_section("5. Merton Jump-Diffusion  [5M paths]");
    std::cout << "  λ=" << merton_p.lambda << "  μJ=" << merton_p.mu_j
              << "  σJ=" << merton_p.sigma_j << "\n\n";

    auto merton_res = run_merton_call(merton_p, K, 5'000'000);
    print_kv("Jump-Diffusion Price:", fmt_price(merton_res.price), term::PURPLE);
    print_kv("GBM Call Price:",       fmt_price(euro.price),       term::BLUE);
    print_kv("Jump Premium:",         fmt_price(merton_res.price - euro.price, 6), term::YELLOW);
    print_kv("Std Error:",            fmt_price(merton_res.std_error, 6));
    print_kv("Elapsed:",              fmt_ms(merton_res.elapsed_ms));

    // ==========================================================
    //  SIMULATION 6 — Portfolio VaR (1M paths, 10 assets)
    // ==========================================================
    print_section("6. Portfolio VaR  [1M paths, 10 correlated assets]");

    auto port = run_portfolio_var(1'000'000);
    auto& ps = port.stats;

    print_kv("Mean Daily P&L:",   "$" + fmt_price(ps.mean, 2), term::GREEN);
    print_kv("Std Dev:",          "$" + fmt_price(ps.std_dev, 2));
    print_kv("VaR 95% (1-day):",  "$" + fmt_price(ps.var_95, 2), term::RED);
    print_kv("VaR 99% (1-day):",  "$" + fmt_price(ps.var_99, 2), term::RED);
    print_kv("CVaR 95%:",         "$" + fmt_price(ps.cvar_95, 2), term::RED);
    print_kv("CVaR 99%:",         "$" + fmt_price(ps.cvar_99, 2), term::RED);
    print_kv("Sharpe (ann.):",    fmt_price(ps.sharpe, 3));
    print_kv("Sortino (ann.):",   fmt_price(ps.sortino, 3));
    print_kv("Elapsed:",          fmt_ms(port.elapsed_ms));

    report.portfolio_pnl   = port.pnl;
    report.portfolio_hist  = make_histogram(port.pnl, 80);
    report.pnl_stats       = ps;
    report.pnl_mean        = ps.mean;
    report.pnl_std         = ps.std_dev;
    report.pnl_hist        = make_histogram(port.pnl, 80);

    // VaR term structure
    {
        std::vector<int> days = {1,2,3,5,7,10,15,21,30,45,63};
        for (int d : days) {
            double scale = std::sqrt(static_cast<double>(d));
            report.var_horizons.push_back(d);
            report.var_95_series.push_back(ps.var_95 * scale);
            report.var_99_series.push_back(ps.var_99 * scale);
        }
    }

    // ==========================================================
    //  SIMULATION 7 — Variance Reduction Comparison (500K paths)
    // ==========================================================
    print_section("7. Variance Reduction Comparison  [500K paths each]");

    auto vr = run_vr_comparison(gbm_p, K, 500'000);
    (void)report.analytical_price; // used in report, not this loop

    std::cout << "\n  " << term::BOLD
              << std::left << std::setw(22) << "Method"
              << std::setw(12) << "Price"
              << std::setw(12) << "Std Error"
              << std::setw(14) << "vs Plain MC"
              << std::setw(10) << "Time" << term::RESET << "\n";
    std::cout << "  " << std::string(70, '-') << "\n";

    double base_se = vr.methods[0].std_error;
    for (auto& m : vr.methods) {
        double reduction = (base_se > 0) ? base_se / m.std_error : 1.0;
        std::cout << "  " << term::BOLD << std::left << std::setw(22) << m.name << term::RESET
                  << std::setw(12) << fmt_price(m.price)
                  << std::setw(12) << fmt_price(m.std_error, 5);
        if (&m == &vr.methods[0])
            std::cout << std::setw(14) << "  baseline";
        else {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << "  ×" << reduction;
            std::cout << term::GREEN << std::setw(14) << ss.str() << term::RESET;
        }
        std::cout << std::setw(10) << fmt_ms(m.elapsed_ms) << "\n";
        report.vr_labels.push_back(m.name);
        report.vr_std_errors.push_back(m.std_error);
        report.vr_prices.push_back(m.price);
    }

    // ==========================================================
    //  SIMULATION 8 — CIR Interest Rate (10-year, 2520 steps)
    // ==========================================================
    print_section("8. CIR Interest Rate Model  [500 paths, 10-year horizon]");
    std::cout << "  r0=" << cir_p.r0 << "  κ=" << cir_p.kappa
              << "  θ=" << cir_p.theta << "  σ=" << cir_p.sigma << "\n\n";

    auto t_cir = Clock::now();
    Xoshiro256pp rng_cir(12345);
    std::vector<std::vector<double>> all_cir_paths;
    for (int i = 0; i < 500; i++)
        all_cir_paths.push_back(cir_path(cir_p, rng_cir));
    double cir_elapsed = Ms(Clock::now()-t_cir).count();

    // Final rate distribution stats
    std::vector<double> final_rates;
    for (auto& p2 : all_cir_paths) final_rates.push_back(p2.back());
    Stats cir_stats = compute_stats(final_rates);

    print_kv("Final Rate Mean:",  fmt_pct(cir_stats.mean));
    print_kv("Final Rate Std:",   fmt_pct(cir_stats.std_dev));
    print_kv("Final Rate P5:",    fmt_pct(cir_stats.p05));
    print_kv("Final Rate P95:",   fmt_pct(cir_stats.p95));
    print_kv("Long-Run θ:",       fmt_pct(cir_p.theta));
    print_kv("Elapsed:",          fmt_ms(cir_elapsed));

    // Store CIR paths for chart (downsampled)
    {
        int keep_n = 12;
        int total_steps = cir_p.steps + 1;
        int skip = total_steps / 200;  // show ~200 points per path
        skip = std::max(skip, 1);

        for (int i = 0; i < keep_n; i++) {
            std::vector<double> dpath;
            for (int s = 0; s < total_steps; s += skip)
                dpath.push_back(all_cir_paths[i][s]);
            report.cir_paths.push_back(dpath);
        }
        for (int s = 0; s < total_steps; s += skip)
            report.cir_time.push_back(static_cast<double>(s) / (total_steps-1) * cir_p.T);
    }

    // ==========================================================
    //  ASCII Charts (terminal preview)
    // ==========================================================
    print_section("Terminal Preview: Distributions");

    // Option payoff histogram
    std::vector<double> sample_payoffs(std::min(final_prices.size(), size_t(50000)));
    Xoshiro256pp rng_sample(777);
    for (auto& sp : sample_payoffs)
        sp = std::exp(-gbm_p.r*gbm_p.T) *
             payoff::european_call(gbm_exact(gbm_p, rng_sample.normal()).final_price, K);
    print_ascii_hist(sample_payoffs, "European Call Payoff Distribution (50K sample)");

    print_ascii_hist(final_rates, "CIR Final Rate Distribution (500 paths, T=10y)");

    // ==========================================================
    //  GREEKS TABLE
    // ==========================================================
    print_section("Option Greeks (Black-Scholes Analytical)");
    auto& G = report.analytical_greeks;
    std::cout << "\n";
    print_kv("Price:",      fmt_price(G.price),              term::GREEN);
    print_kv("Delta:",      fmt_price(G.delta, 6));
    print_kv("Gamma:",      fmt_price(G.gamma, 6));
    print_kv("Vega:",       fmt_price(G.vega, 6) + " (per 1% σ)");
    print_kv("Theta:",      fmt_price(G.theta, 6) + " (per day)");
    print_kv("Rho:",        fmt_price(G.rho, 6) + " (per 1% r)");

    // ==========================================================
    //  TIMING SUMMARY
    // ==========================================================
    double total_ms = Ms(Clock::now() - total_start).count();
    report.elapsed_ms = total_ms;

    size_t total_paths_run = N_EUROPEAN + 5'000'000 + 5'000'000 + 5'000'000
                           + 5'000'000 + 1'000'000 + 4*500'000 + 500*cir_p.steps;
    report.total_paths = total_paths_run;

    print_section("Performance Summary");
    std::cout << "\n";
    print_kv("Total elapsed:",          fmt_ms(total_ms), term::CYAN);
    print_kv("Total simulated paths:",  [&]() -> std::string {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << total_paths_run/1e6 << "M";
        return ss.str();
    }(), term::GREEN);
    print_kv("Threads:",                std::to_string(N_THREADS), term::BLUE);
    print_kv("Paths/sec (peak):",       [&]() -> std::string {
        double pps = N_EUROPEAN / (euro.elapsed_ms/1000.0);
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1) << pps/1e6 << " M/sec";
        return ss.str();
    }(), term::YELLOW);

    // ==========================================================
    //  GENERATE HTML REPORT
    // ==========================================================
    print_section("Generating HTML Report");
    const std::string html_file = "/Users/sonnie/Dev/quant/montecarlo/report.html";
    try {
        write_html_report(report, html_file);
        std::cout << "\n  " << term::GREEN << "✓" << term::RESET
                  << " Report saved: " << term::BOLD << html_file << term::RESET << "\n";
        std::cout << "  Open in browser: " << term::CYAN
                  << "open " << html_file << term::RESET << "\n\n";
    } catch (std::exception& e) {
        std::cerr << "  " << term::RED << "✗ Error: " << e.what() << term::RESET << "\n";
    }

    std::cout << term::BOLD << term::GREEN
              << "\n  All simulations complete.\n" << term::RESET << "\n";

    return 0;
}
