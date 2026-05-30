#pragma once
// HTML report generator with Chart.js interactive charts
// Produces a single self-contained HTML file

#include "statistics.hpp"
#include "models.hpp"
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>

// ============================================================
//  JSON array helpers
// ============================================================
static std::string to_json_arr(const std::vector<double>& v, int dp = 4) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(dp) << "[";
    for (size_t i = 0; i < v.size(); i++) {
        ss << v[i];
        if (i + 1 < v.size()) ss << ",";
    }
    ss << "]";
    return ss.str();
}

// ============================================================
//  Report data container
// ============================================================
struct ReportData {
    // GBM paths (sampled)
    std::vector<std::vector<double>> sample_paths;  // each inner = one path
    std::vector<double>              time_axis;

    // Price distribution
    Histogram price_hist;
    double    S0, logS_mu, logS_sigma;   // lognormal overlay params

    // P&L / return distribution
    Histogram pnl_hist;
    double    pnl_mean, pnl_std;

    // Convergence data
    std::vector<ConvergencePoint> convergence;
    double analytical_price;

    // VaR term structure
    std::vector<double> var_horizons;   // days
    std::vector<double> var_95_series;
    std::vector<double> var_99_series;

    // Variance reduction comparison
    std::vector<std::string> vr_labels;
    std::vector<double>      vr_std_errors;
    std::vector<double>      vr_prices;

    // Heston vs GBM paths comparison
    std::vector<std::vector<double>> heston_paths;
    std::vector<std::vector<double>> gbm_paths;

    // Greeks table data
    bs::Greeks analytical_greeks;

    // Summary stats
    Stats price_stats;
    Stats pnl_stats;

    // Multi-asset portfolio P&L
    std::vector<double> portfolio_pnl;
    Histogram portfolio_hist;

    // CIR interest rate paths
    std::vector<std::vector<double>> cir_paths;
    std::vector<double>              cir_time;

    // Simulation metadata
    std::string title;
    size_t total_paths;
    double elapsed_ms;
    int    n_threads;
};

// ============================================================
//  ASCII progress bar (for console)
// ============================================================
inline std::string ascii_bar(double fraction, int width = 40) {
    int filled = static_cast<int>(fraction * width);
    std::string bar = "[";
    for (int i = 0; i < width; i++) bar += (i < filled) ? "█" : "░";
    bar += "]";
    return bar;
}

// ============================================================
//  HTML report generation
// ============================================================
inline std::string generate_html(const ReportData& d) {
    std::ostringstream html;

    // ---- HEAD ----
    html << R"(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>)" << d.title << R"(</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e;
    --green: #3fb950; --red: #f85149; --blue: #58a6ff;
    --purple: #bc8cff; --orange: #d29922; --teal: #39d353;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; }
  h1 { text-align: center; padding: 24px; font-size: 1.8rem; color: var(--blue); border-bottom: 1px solid var(--border); }
  h2 { font-size: 1.1rem; color: var(--muted); margin-bottom: 12px; text-transform: uppercase; letter-spacing: .05em; }
  .subtitle { text-align: center; color: var(--muted); padding: 8px 0 20px; font-size: .9rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 20px; padding: 20px; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }
  .panel.wide { grid-column: 1/-1; }
  canvas { max-height: 280px; }
  .canvas-tall canvas { max-height: 340px; }
  table { width: 100%; border-collapse: collapse; font-size: .85rem; }
  th { background: #1c2128; color: var(--muted); text-align: left; padding: 8px 10px;
       font-weight: 600; border-bottom: 1px solid var(--border); }
  td { padding: 7px 10px; border-bottom: 1px solid #21262d; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1c2128; }
  .pos { color: var(--green); } .neg { color: var(--red); }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: .75rem; font-weight: 600; }
  .badge-blue { background: rgba(88,166,255,.15); color: var(--blue); }
  .badge-green { background: rgba(63,185,80,.15); color: var(--green); }
  .badge-orange { background: rgba(210,153,34,.15); color: var(--orange); }
  .meta-row { display: flex; gap: 20px; flex-wrap: wrap; padding: 0 20px 16px; }
  .meta-card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
               padding: 12px 20px; flex: 1; min-width: 140px; }
  .meta-card .label { font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  .meta-card .value { font-size: 1.4rem; font-weight: 700; margin-top: 4px; color: var(--blue); }
  .meta-card .sub { font-size: .75rem; color: var(--muted); margin-top: 2px; }
  footer { text-align: center; padding: 20px; color: var(--muted); font-size: .8rem; border-top: 1px solid var(--border); margin-top: 20px; }
</style>
</head>
<body>
<h1>)";
    html << d.title;
    html << R"(</h1>
<p class="subtitle">)" << d.total_paths << R"( paths &nbsp;·&nbsp; )"
         << d.n_threads << R"( threads &nbsp;·&nbsp; )" << fmt_ms(d.elapsed_ms) << R"( total</p>

)";

    // ---- META CARDS ----
    html << "<div class=\"meta-row\">\n";

    auto meta = [&](const std::string& lbl, const std::string& val, const std::string& sub) {
        html << "  <div class=\"meta-card\"><div class=\"label\">" << lbl << "</div>"
             << "<div class=\"value\">" << val << "</div>"
             << "<div class=\"sub\">" << sub << "</div></div>\n";
    };

    meta("MC Price",   fmt_price(d.price_stats.mean),
         "95% CI [" + fmt_price(d.price_stats.ci_lower) + ", " + fmt_price(d.price_stats.ci_upper) + "]");
    meta("Analytical", fmt_price(d.analytical_price), "Black-Scholes closed-form");
    meta("Std Error",  fmt_price(d.price_stats.std_error, 5), "σ/√N");
    meta("Paths/sec",  [&]() -> std::string {
        double pps = d.total_paths / (d.elapsed_ms / 1000.0);
        std::ostringstream ss;
        if (pps > 1e6) ss << std::fixed << std::setprecision(1) << pps/1e6 << " M";
        else           ss << std::fixed << std::setprecision(0) << pps/1e3 << " K";
        return ss.str();
    }(), "paths per second");
    meta("VaR 99%",    fmt_price(d.pnl_stats.var_99), "1-day 99% Value at Risk");
    meta("CVaR 99%",   fmt_price(d.pnl_stats.cvar_99), "Expected Shortfall");

    html << "</div>\n\n<div class=\"grid\">\n";

    // ============================================================
    // CHART 1: Sample Price Paths
    // ============================================================
    html << R"(<div class="panel canvas-tall">
<h2>Simulated Price Paths (GBM + Heston)</h2>
<canvas id="pathsChart"></canvas>
</div>
)";

    // ============================================================
    // CHART 2: Final Price Distribution
    // ============================================================
    html << R"(<div class="panel">
<h2>Final Price Distribution</h2>
<canvas id="distChart"></canvas>
</div>
)";

    // ============================================================
    // CHART 3: Convergence
    // ============================================================
    html << R"(<div class="panel">
<h2>Option Price Convergence</h2>
<canvas id="convChart"></canvas>
</div>
)";

    // ============================================================
    // CHART 4: P&L Distribution
    // ============================================================
    html << R"(<div class="panel">
<h2>P&amp;L Distribution with VaR / CVaR</h2>
<canvas id="pnlChart"></canvas>
</div>
)";

    // ============================================================
    // CHART 5: Variance Reduction Comparison
    // ============================================================
    html << R"(<div class="panel">
<h2>Variance Reduction Comparison</h2>
<canvas id="vrChart"></canvas>
</div>
)";

    // ============================================================
    // CHART 6: VaR Term Structure
    // ============================================================
    html << R"(<div class="panel">
<h2>VaR Term Structure</h2>
<canvas id="varTSChart"></canvas>
</div>
)";

    // ============================================================
    // CHART 7: CIR Interest Rate Paths
    // ============================================================
    html << R"(<div class="panel canvas-tall">
<h2>CIR Interest Rate Simulation</h2>
<canvas id="cirChart"></canvas>
</div>
)";

    // ============================================================
    // CHART 8: Portfolio P&L
    // ============================================================
    html << R"(<div class="panel">
<h2>Portfolio P&amp;L Distribution (10-Asset)</h2>
<canvas id="portChart"></canvas>
</div>
)";

    // ============================================================
    // TABLE: Statistics Summary
    // ============================================================
    html << R"(<div class="panel wide">
<h2>Full Statistics Summary</h2>
<table>
<tr><th>Metric</th><th>Option Price</th><th>Daily P&amp;L</th></tr>
)";
    auto stats_row = [&](const std::string& metric, double v1, double v2, bool pct=false) {
        auto fmt = [&](double v) -> std::string {
            std::ostringstream ss;
            if (pct) ss << std::fixed << std::setprecision(2) << v*100 << "%";
            else     ss << std::fixed << std::setprecision(4) << v;
            return ss.str();
        };
        html << "<tr><td>" << metric << "</td>"
             << "<td>" << fmt(v1) << "</td>"
             << "<td>" << fmt(v2) << "</td></tr>\n";
    };
    auto& ps = d.price_stats;
    auto& pl = d.pnl_stats;
    stats_row("Mean",         ps.mean,     pl.mean);
    stats_row("Median",       ps.median,   pl.median);
    stats_row("Std Dev",      ps.std_dev,  pl.std_dev);
    stats_row("Std Error",    ps.std_error, pl.std_error);
    stats_row("Skewness",     ps.skewness, pl.skewness);
    stats_row("Excess Kurtosis", ps.kurtosis, pl.kurtosis);
    stats_row("Min",          ps.min_val,  pl.min_val);
    stats_row("Max",          ps.max_val,  pl.max_val);
    stats_row("P1",           ps.p01,      pl.p01);
    stats_row("P5",           ps.p05,      pl.p05);
    stats_row("P25",          ps.p25,      pl.p25);
    stats_row("P75",          ps.p75,      pl.p75);
    stats_row("P95",          ps.p95,      pl.p95);
    stats_row("P99",          ps.p99,      pl.p99);
    stats_row("VaR 90%",      ps.var_90,   pl.var_90);
    stats_row("VaR 95%",      ps.var_95,   pl.var_95);
    stats_row("VaR 99%",      ps.var_99,   pl.var_99);
    stats_row("CVaR 90%",     ps.cvar_90,  pl.cvar_90);
    stats_row("CVaR 95%",     ps.cvar_95,  pl.cvar_95);
    stats_row("CVaR 99%",     ps.cvar_99,  pl.cvar_99);
    html << "</table></div>\n";

    // ============================================================
    // TABLE: Greeks
    // ============================================================
    html << R"(<div class="panel">
<h2>Option Greeks (Analytical)</h2>
<table>
<tr><th>Greek</th><th>Value</th><th>Interpretation</th></tr>
)";
    auto greek_row = [&](const std::string& g, double v, const std::string& interp) {
        html << "<tr><td><b>" << g << "</b></td><td>"
             << std::fixed << std::setprecision(6) << v
             << "</td><td style=\"color:var(--muted);font-size:.8rem\">" << interp << "</td></tr>\n";
    };
    auto& G = d.analytical_greeks;
    greek_row("Price",  G.price,  "Call option fair value");
    greek_row("Delta",  G.delta,  "∂V/∂S — hedge ratio");
    greek_row("Gamma",  G.gamma,  "∂²V/∂S² — convexity");
    greek_row("Vega",   G.vega,   "∂V/∂σ per 1% vol move");
    greek_row("Theta",  G.theta,  "∂V/∂t per calendar day");
    greek_row("Rho",    G.rho,    "∂V/∂r per 1% rate move");
    html << "</table></div>\n";

    html << "</div>\n\n"; // close grid

    // ============================================================
    // JAVASCRIPT — Chart.js rendering
    // ============================================================
    html << "<script>\n";
    html << "const COLORS = { blue:'#58a6ff', green:'#3fb950', red:'#f85149', "
            "purple:'#bc8cff', orange:'#d29922', teal:'#39d353', muted:'#8b949e' };\n";
    html << "const defaults = {\n"
            "  responsive: true,\n"
            "  plugins: { legend: { labels: { color: '#e6edf3', font:{size:11} } } },\n"
            "  scales: {\n"
            "    x: { ticks:{color:'#8b949e'}, grid:{color:'#21262d'} },\n"
            "    y: { ticks:{color:'#8b949e'}, grid:{color:'#21262d'} }\n"
            "  }\n"
            "};\n\n";

    // Helper function
    html << R"(
function makeChart(id, cfg) {
  const ctx = document.getElementById(id).getContext('2d');
  cfg.options = Object.assign({}, defaults, cfg.options||{});
  if (cfg.options.scales) {
    cfg.options.scales = Object.assign({}, defaults.scales, cfg.options.scales);
  }
  cfg.options.plugins = Object.assign({}, defaults.plugins, cfg.options.plugins||{});
  return new Chart(ctx, cfg);
}
)";

    // ------ Chart 1: Paths ------
    html << "\n// ---- Price Paths ----\n";
    html << "makeChart('pathsChart', { type: 'line', data: {\n";
    html << "  labels: " << to_json_arr(d.time_axis, 3) << ",\n";
    html << "  datasets: [\n";

    // GBM paths
    size_t n_gbm = std::min(d.gbm_paths.size(), size_t(15));
    for (size_t i = 0; i < n_gbm; i++) {
        double alpha = (i == 0) ? 0.9 : 0.35;
        html << "    { label: '" << (i==0?"GBM #1":"") << "', data: "
             << to_json_arr(d.gbm_paths[i], 2)
             << ", borderColor: 'rgba(88,166,255," << alpha << ")',"
             << " borderWidth: " << (i==0?1.5:0.8) << ", pointRadius: 0, fill: false },\n";
    }
    // Heston paths
    size_t n_hes = std::min(d.heston_paths.size(), size_t(15));
    for (size_t i = 0; i < n_hes; i++) {
        double alpha = (i == 0) ? 0.9 : 0.35;
        html << "    { label: '" << (i==0?"Heston #1":"") << "', data: "
             << to_json_arr(d.heston_paths[i], 2)
             << ", borderColor: 'rgba(188,140,255," << alpha << ")',"
             << " borderWidth: " << (i==0?1.5:0.8) << ", pointRadius: 0, fill: false },\n";
    }
    html << "  ]\n}, options: { plugins: { legend: { display: true } } } });\n";

    // ------ Chart 2: Price Distribution ------
    html << "\n// ---- Price Distribution ----\n";
    {
        const auto& h = d.price_hist;
        // Build lognormal overlay
        std::vector<double> overlay(h.centers.size());
        for (size_t i = 0; i < h.centers.size(); i++)
            overlay[i] = lognormal_pdf(h.centers[i], d.logS_mu, d.logS_sigma);

        html << "makeChart('distChart', { type: 'bar', data: {\n"
             << "  labels: " << to_json_arr(h.centers, 1) << ",\n"
             << "  datasets: [\n"
             << "    { label: 'MC Density', data: " << to_json_arr(h.counts) << ","
             << "      backgroundColor: 'rgba(88,166,255,0.55)', borderColor: 'transparent' },\n"
             << "    { label: 'Lognormal', data: " << to_json_arr(overlay) << ","
             << "      type: 'line', borderColor: '#f85149', borderWidth: 2, pointRadius: 0, fill: false }\n"
             << "  ]\n}, options: { scales: { x: { ticks: { maxTicksLimit: 10 } } } } });\n";
    }

    // ------ Chart 3: Convergence ------
    html << "\n// ---- Convergence ----\n";
    {
        std::vector<double> ns, est, lo, hi;
        for (auto& c : d.convergence) {
            ns.push_back(static_cast<double>(c.n_paths));
            est.push_back(c.estimate);
            lo.push_back(c.ci_lower);
            hi.push_back(c.ci_upper);
        }
        std::vector<double> analytical_line(ns.size(), d.analytical_price);

        html << "makeChart('convChart', { type: 'line', data: {\n"
             << "  labels: " << to_json_arr(ns, 0) << ",\n"
             << "  datasets: [\n"
             << "    { label: 'MC Estimate', data: " << to_json_arr(est) << ","
             << "      borderColor: COLORS.blue, borderWidth:2, pointRadius:0, fill:false },\n"
             << "    { label: '+2σ', data: " << to_json_arr(hi) << ","
             << "      borderColor: 'rgba(88,166,255,0.3)', borderDash:[4,4], borderWidth:1, pointRadius:0, fill:false },\n"
             << "    { label: '-2σ', data: " << to_json_arr(lo) << ","
             << "      borderColor: 'rgba(88,166,255,0.3)', borderDash:[4,4], borderWidth:1, pointRadius:0, fill:false },\n"
             << "    { label: 'Black-Scholes', data: " << to_json_arr(analytical_line) << ","
             << "      borderColor: COLORS.red, borderDash:[6,3], borderWidth:2, pointRadius:0, fill:false }\n"
             << "  ]\n}, options: { scales: { x: { ticks: { maxTicksLimit: 8 } } } } });\n";
    }

    // ------ Chart 4: P&L Distribution ------
    html << "\n// ---- P&L Distribution ----\n";
    {
        const auto& h = d.pnl_hist;
        std::vector<double> normal_overlay(h.centers.size());
        for (size_t i = 0; i < h.centers.size(); i++)
            normal_overlay[i] = normal_pdf(h.centers[i], d.pnl_mean, d.pnl_std);

        // Color bins: losses (below 0) in red, gains in green
        std::string colors = "[";
        for (size_t i = 0; i < h.centers.size(); i++) {
            colors += (h.centers[i] < 0) ? "'rgba(248,81,73,0.6)'" : "'rgba(63,185,80,0.6)'";
            if (i+1 < h.centers.size()) colors += ",";
        }
        colors += "]";

        html << "makeChart('pnlChart', { type: 'bar', data: {\n"
             << "  labels: " << to_json_arr(h.centers, 1) << ",\n"
             << "  datasets: [\n"
             << "    { label: 'P&L Density', data: " << to_json_arr(h.counts) << ","
             << "      backgroundColor: " << colors << ", borderColor: 'transparent' },\n"
             << "    { label: 'Normal Overlay', data: " << to_json_arr(normal_overlay) << ","
             << "      type: 'line', borderColor: '#d29922', borderWidth:2, pointRadius:0, fill:false }\n"
             << "  ]\n}, options: {\n"
             << "  plugins: { annotation: {} },\n"
             << "  scales: { x: { ticks: { maxTicksLimit: 10 } } }\n"
             << "} });\n";
    }

    // ------ Chart 5: Variance Reduction ------
    html << "\n// ---- Variance Reduction ----\n";
    {
        std::vector<std::string> lbl_js;
        for (auto& l : d.vr_labels) lbl_js.push_back("\"" + l + "\"");
        html << "makeChart('vrChart', { type: 'bar', data: {\n"
             << "  labels: [";
        for (size_t i = 0; i < lbl_js.size(); i++) {
            html << lbl_js[i]; if (i+1<lbl_js.size()) html << ",";
        }
        html << "],\n"
             << "  datasets: [\n"
             << "    { label: 'Std Error', data: " << to_json_arr(d.vr_std_errors, 5) << ","
             << "      backgroundColor: ['rgba(88,166,255,0.7)','rgba(63,185,80,0.7)',"
                      "'rgba(188,140,255,0.7)','rgba(57,211,83,0.7)'], borderRadius: 4 },\n"
             << "    { label: 'Price', data: " << to_json_arr(d.vr_prices, 4) << ","
             << "      backgroundColor: 'transparent', borderColor: COLORS.orange,"
                      " borderWidth: 2, type: 'line', yAxisID: 'y2' }\n"
             << "  ]\n}, options: {\n"
             << "  scales: { y: { title:{display:true,text:'Std Error',color:'#8b949e'} },"
             << "    y2: { position:'right', title:{display:true,text:'Price',color:'#8b949e'},"
                  " grid:{display:false} } }\n"
             << "} });\n";
    }

    // ------ Chart 6: VaR Term Structure ------
    html << "\n// ---- VaR Term Structure ----\n";
    html << "makeChart('varTSChart', { type: 'line', data: {\n"
         << "  labels: " << to_json_arr(d.var_horizons, 0) << ",\n"
         << "  datasets: [\n"
         << "    { label: 'VaR 95%', data: " << to_json_arr(d.var_95_series) << ","
         << "      borderColor: COLORS.orange, borderWidth:2, pointRadius:2, fill:false },\n"
         << "    { label: 'VaR 99%', data: " << to_json_arr(d.var_99_series) << ","
         << "      borderColor: COLORS.red, borderWidth:2, pointRadius:2, fill:false }\n"
         << "  ]\n} });\n";

    // ------ Chart 7: CIR Paths ------
    html << "\n// ---- CIR Paths ----\n";
    html << "makeChart('cirChart', { type: 'line', data: {\n"
         << "  labels: " << to_json_arr(d.cir_time, 2) << ",\n"
         << "  datasets: [\n";
    size_t n_cir = std::min(d.cir_paths.size(), size_t(12));
    for (size_t i = 0; i < n_cir; i++) {
        double alpha = (i==0) ? 0.9 : 0.4;
        html << "    { label:'" << (i==0?"CIR r(t)":"") << "', data: "
             << to_json_arr(d.cir_paths[i], 4)
             << ", borderColor:'rgba(57,211,83," << alpha << ")',"
             << " borderWidth:" << (i==0?2:1) << ", pointRadius:0, fill:false },\n";
    }
    html << "  ]\n} });\n";

    // ------ Chart 8: Portfolio P&L ------
    html << "\n// ---- Portfolio P&L ----\n";
    {
        const auto& h = d.portfolio_hist;
        html << "makeChart('portChart', { type: 'bar', data: {\n"
             << "  labels: " << to_json_arr(h.centers, 1) << ",\n"
             << "  datasets: [{ label: 'Portfolio P&L Density', data: "
             << to_json_arr(h.counts) << ","
             << " backgroundColor: 'rgba(188,140,255,0.6)', borderColor:'transparent' }]\n"
             << "} });\n";
    }

    html << "</script>\n";
    html << "<footer>Generated by C++ Monte Carlo Engine &nbsp;·&nbsp; "
         << d.total_paths << " total paths &nbsp;·&nbsp; " << d.n_threads << " threads</footer>\n";
    html << "</body>\n</html>\n";

    return html.str();
}

inline void write_html_report(const ReportData& d, const std::string& filename) {
    std::ofstream f(filename);
    if (!f) throw std::runtime_error("Cannot write " + filename);
    f << generate_html(d);
    f.close();
}
