#pragma once
// Comprehensive statistical analysis for Monte Carlo results
// Descriptive stats, risk metrics (VaR/CVaR/ES), convergence, bootstrap CI

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>
#include <string>
#include <sstream>
#include <iomanip>

// ============================================================
//  Full statistics bundle
// ============================================================
struct Stats {
    // Descriptive
    double mean        = 0;
    double median      = 0;
    double std_dev     = 0;
    double std_error   = 0;    // std_dev / sqrt(N)
    double skewness    = 0;
    double kurtosis    = 0;    // excess kurtosis
    double min_val     = 0;
    double max_val     = 0;
    double p01         = 0;    // 1st percentile
    double p05         = 0;
    double p25         = 0;    // Q1
    double p75         = 0;    // Q3
    double p95         = 0;
    double p99         = 0;

    // Risk metrics
    double var_90      = 0;    // Value at Risk (loss at 90% confidence)
    double var_95      = 0;
    double var_99      = 0;
    double var_999     = 0;
    double cvar_90     = 0;    // Conditional VaR / Expected Shortfall
    double cvar_95     = 0;
    double cvar_99     = 0;
    double max_drawdown= 0;

    // Return metrics (assumes data is P&L or returns)
    double sharpe      = 0;
    double sortino     = 0;
    double calmar      = 0;

    // Simulation quality
    double ci_lower    = 0;    // 95% confidence interval for mean
    double ci_upper    = 0;
    double eff_n       = 0;    // effective sample size (via autocorrelation)
    size_t n           = 0;
};

inline double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    double idx = p * (sorted.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = std::min(lo + 1, sorted.size() - 1);
    double frac = idx - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

// Compute all statistics from a sorted vector of prices/returns/P&L
inline Stats compute_stats(std::vector<double> data, bool is_pnl = false) {
    Stats s;
    s.n = data.size();
    if (s.n == 0) return s;

    std::sort(data.begin(), data.end());
    s.min_val = data.front();
    s.max_val = data.back();

    // Mean
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    s.mean = sum / s.n;

    // Median
    s.median = percentile(data, 0.5);

    // Variance, skewness, kurtosis (single-pass after mean)
    double var = 0, sk = 0, ku = 0;
    for (double x : data) {
        double d = x - s.mean;
        var += d*d;
        sk  += d*d*d;
        ku  += d*d*d*d;
    }
    var /= (s.n - 1);
    s.std_dev  = std::sqrt(var);
    s.std_error = s.std_dev / std::sqrt(static_cast<double>(s.n));

    double m2 = var;
    double m3 = sk / s.n;
    double m4 = ku / s.n;
    s.skewness = m3 / std::pow(m2, 1.5);
    s.kurtosis = m4 / (m2*m2) - 3.0;

    // Percentiles
    s.p01  = percentile(data, 0.01);
    s.p05  = percentile(data, 0.05);
    s.p25  = percentile(data, 0.25);
    s.p75  = percentile(data, 0.75);
    s.p95  = percentile(data, 0.95);
    s.p99  = percentile(data, 0.99);

    // VaR (loss = negative return convention for P&L)
    if (is_pnl) {
        // VaR is the loss threshold exceeded with (1-conf) probability
        s.var_90  = -percentile(data, 0.10);
        s.var_95  = -percentile(data, 0.05);
        s.var_99  = -percentile(data, 0.01);
        s.var_999 = -percentile(data, 0.001);

        // CVaR = E[loss | loss > VaR]
        auto cvar_at = [&](double alpha) {
            size_t cutoff = static_cast<size_t>(alpha * s.n);
            if (cutoff == 0) return -data[0];
            double tail_sum = 0;
            for (size_t i = 0; i < cutoff; i++) tail_sum += data[i];
            return -tail_sum / cutoff;
        };
        s.cvar_90  = cvar_at(0.10);
        s.cvar_95  = cvar_at(0.05);
        s.cvar_99  = cvar_at(0.01);

        // Max drawdown from sorted P&L (simplified: worst single loss)
        s.max_drawdown = s.var_999;

        // Sharpe (annualised, assume daily P&L scaled)
        double neg_sum = 0; size_t neg_cnt = 0;
        for (double x : data) if (x < 0) { neg_sum += x*x; neg_cnt++; }
        double downside_std = (neg_cnt > 1) ? std::sqrt(neg_sum/neg_cnt) : s.std_dev;
        s.sharpe  = (s.mean / s.std_dev) * std::sqrt(252.0);
        s.sortino = (s.mean / downside_std) * std::sqrt(252.0);
        s.calmar  = (s.mean * 252.0) / s.max_drawdown;
    }

    // 95% CI for mean (t-distribution approx, large N → z)
    double z95 = 1.959964;
    s.ci_lower = s.mean - z95 * s.std_error;
    s.ci_upper = s.mean + z95 * s.std_error;

    return s;
}

// ============================================================
//  Convergence analysis — price estimate vs N paths
// ============================================================
struct ConvergencePoint {
    size_t n_paths;
    double estimate;
    double std_error;
    double ci_lower, ci_upper;
};

inline std::vector<ConvergencePoint> convergence_analysis(
    const std::vector<double>& payoffs,  // discounted payoffs
    size_t n_points = 50)
{
    size_t N = payoffs.size();
    std::vector<ConvergencePoint> result;
    result.reserve(n_points);

    // Logarithmically spaced evaluation points
    for (size_t k = 0; k < n_points; k++) {
        double frac = static_cast<double>(k+1) / n_points;
        size_t n = static_cast<size_t>(std::exp(std::log(N) * frac));
        n = std::max(n, size_t(10));
        n = std::min(n, N);

        double sum = 0, sum2 = 0;
        for (size_t i = 0; i < n; i++) {
            sum  += payoffs[i];
            sum2 += payoffs[i] * payoffs[i];
        }
        double mean = sum / n;
        double var  = (sum2/n - mean*mean) * n/(n-1.0);
        double se   = std::sqrt(var / n);
        result.push_back({ n, mean, se, mean - 1.96*se, mean + 1.96*se });
    }
    return result;
}

// ============================================================
//  Histogram builder
// ============================================================
struct Histogram {
    std::vector<double> edges;    // n+1 edges
    std::vector<double> counts;   // n bins (normalized to density)
    std::vector<double> centers;
    double bin_width;
};

inline Histogram make_histogram(const std::vector<double>& data, int n_bins = 100) {
    if (data.empty()) return {};
    double lo = *std::min_element(data.begin(), data.end());
    double hi = *std::max_element(data.begin(), data.end());
    // Trim 0.1% outliers for cleaner histogram
    auto sorted = data;
    std::sort(sorted.begin(), sorted.end());
    lo = percentile(sorted, 0.001);
    hi = percentile(sorted, 0.999);
    double w = (hi - lo) / n_bins;
    if (w <= 0) w = 1.0;

    Histogram h;
    h.bin_width = w;
    h.edges.resize(n_bins + 1);
    h.counts.resize(n_bins, 0.0);
    h.centers.resize(n_bins);
    for (int i = 0; i <= n_bins; i++) h.edges[i] = lo + i * w;
    for (int i = 0; i < n_bins; i++) h.centers[i] = lo + (i + 0.5) * w;

    for (double x : data) {
        if (x < lo || x > hi) continue;
        int bin = static_cast<int>((x - lo) / w);
        bin = std::clamp(bin, 0, n_bins - 1);
        h.counts[bin]++;
    }
    // Normalize to probability density
    double total = static_cast<double>(data.size()) * w;
    for (auto& c : h.counts) c /= total;

    return h;
}

// ============================================================
//  Lognormal PDF (for overlay on price histogram)
// ============================================================
inline double lognormal_pdf(double x, double mu, double sigma) noexcept {
    if (x <= 0) return 0.0;
    double z = (std::log(x) - mu) / sigma;
    return std::exp(-0.5*z*z) / (x * sigma * std::sqrt(2.0 * M_PI));
}

// ============================================================
//  Normal PDF
// ============================================================
inline double normal_pdf(double x, double mu, double sigma) noexcept {
    double z = (x - mu) / sigma;
    return std::exp(-0.5*z*z) / (sigma * std::sqrt(2.0 * M_PI));
}

// ============================================================
//  Bootstrap confidence interval for a statistic
// ============================================================
inline std::pair<double, double> bootstrap_ci(
    const std::vector<double>& data,
    double (*stat_fn)(const std::vector<double>&),
    int n_boot = 1000,
    double alpha = 0.05)
{
    size_t n = data.size();
    std::vector<double> boot_stats(n_boot);

    // Simple LCG for speed
    uint64_t rng = 0xfeedface12345678ULL;
    auto fast_rand = [&]() -> size_t {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        return rng >> 33;
    };

    std::vector<double> sample(n);
    for (int b = 0; b < n_boot; b++) {
        for (size_t i = 0; i < n; i++)
            sample[i] = data[fast_rand() % n];
        boot_stats[b] = stat_fn(sample);
    }
    std::sort(boot_stats.begin(), boot_stats.end());
    return {
        percentile(boot_stats, alpha/2.0),
        percentile(boot_stats, 1.0 - alpha/2.0)
    };
}

// ============================================================
//  Pretty formatting helpers
// ============================================================
inline std::string fmt_price(double v, int dp = 4) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(dp) << v;
    return ss.str();
}

inline std::string fmt_pct(double v, int dp = 2) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(dp) << (v * 100.0) << "%";
    return ss.str();
}

inline std::string fmt_ms(double ms) {
    std::ostringstream ss;
    if (ms < 1000) ss << std::fixed << std::setprecision(1) << ms << " ms";
    else            ss << std::fixed << std::setprecision(2) << ms/1000.0 << " s";
    return ss.str();
}
