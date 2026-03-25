"""
384-Well Potency Assay — Unified Web Application (Improved)
============================================================
Improvements over v1:
  ✓ Robust 4PL fitting with weighted regression & confidence intervals
  ✓ QC flagging (CV%, hook effect, edge effect, flat curves)
  ✓ Grubbs outlier detection on replicates & background wells
  ✓ Multi-plate merge with append mode
  ✓ Reference standard normalization & relative potency
  ✓ Input validation with actionable error messages
  ✓ Consolidated callbacks via unified data pipeline
  ✓ Cleaner style architecture

Plate: 384-well, 2×2 quadruplicates, 12 Ab/plate × 2 plates
Requirements: pip install dash dash-ag-grid plotly pandas openpyxl scipy numpy
"""

import base64, io, math, json
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import dash
from dash import html, dcc, Input, Output, State, callback, no_update, ctx
import dash_ag_grid as dag
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy import stats as sp_stats

# ═══════════════════════════════════════════════════════════════════
# §1  CONSTANTS
# ═══════════════════════════════════════════════════════════════════
ROW_LABELS = [chr(i) for i in range(65, 81)]  # A–P
ABS_PER_PLATE, COLS_PER_AB = 12, 2
MAX_PROTEINS = 6

AB_COLORS = [
    "#3b82f6","#f59e0b","#10b981","#ef4444","#8b5cf6","#ec4899",
    "#14b8a6","#f97316","#6366f1","#84cc16","#06b6d4","#e11d48",
    "#0d9488","#d97706","#7c3aed","#dc2626","#059669","#2563eb",
    "#c026d3","#ea580c","#4f46e5","#65a30d","#0891b2","#be123c",
]

QC_CV_THRESHOLD = 20.0      # flag replicate groups with CV% above this
QC_R2_MIN = 0.85            # minimum R² for acceptable fit
QC_DYNAMIC_RANGE_MIN = 15.0 # minimum top−bottom % for non-flat curve
GRUBBS_ALPHA = 0.05         # significance level for outlier detection

# ═══════════════════════════════════════════════════════════════════
# §2  ENUMS, DATA CLASSES & VALIDATION
# ═══════════════════════════════════════════════════════════════════

class FitStatus(Enum):
    SUCCESS = "success"
    FLAT_CURVE = "flat_curve"
    INSUFFICIENT_POINTS = "insufficient_points"
    POOR_FIT = "poor_fit"
    CONVERGENCE_FAILED = "convergence_failed"
    HOOK_EFFECT = "hook_effect"

FIT_STATUS_LABELS = {
    FitStatus.SUCCESS: ("✓ Good fit", "#10b981"),
    FitStatus.FLAT_CURVE: ("⚠ Flat curve – insufficient dynamic range", "#f59e0b"),
    FitStatus.INSUFFICIENT_POINTS: ("✗ Too few valid points (<4)", "#ef4444"),
    FitStatus.POOR_FIT: ("⚠ Poor fit (R² < 0.85)", "#f59e0b"),
    FitStatus.CONVERGENCE_FAILED: ("✗ Fit did not converge", "#ef4444"),
    FitStatus.HOOK_EFFECT: ("⚠ Hook effect detected – biphasic response", "#f59e0b"),
}

@dataclass
class FitResult:
    status: FitStatus
    params: Optional[Tuple[float,float,float,float]] = None  # top, bot, ec50, hill
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    ec50_ci: Optional[Tuple[float,float]] = None  # 95% CI
    residuals: Optional[List[float]] = None
    message: str = ""

    def to_dict(self):
        d = {
            "status": self.status.value, "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared, "message": self.message,
        }
        if self.params: d["params"] = list(self.params)
        if self.ec50_ci: d["ec50_ci"] = list(self.ec50_ci)
        if self.residuals: d["residuals"] = self.residuals
        return d


@dataclass
class QCFlag:
    level: str          # "warn" or "fail"
    category: str       # "cv", "hook", "edge", "flat", "outlier", "fit"
    message: str
    well: str = ""      # optional well reference

    def to_dict(self):
        return {"level": self.level, "category": self.category,
                "message": self.message, "well": self.well}


def validate_dilution_params(start, factor, n_points, block_vol, stock_vol, conc_factor):
    """Validate dilution block parameters. Returns list of (field_id, message) tuples."""
    errors = []
    if not start or start <= 0:
        errors.append(("start-conc", "Starting concentration must be > 0"))
    if start and start > 1e6:
        errors.append(("start-conc", f"Starting conc {start} nM seems very high. Verify units."))
    if not factor or factor < 2:
        errors.append(("dil-factor", "Dilution factor must be ≥ 2"))
    if not n_points or n_points < 3:
        errors.append(("n-dil", "Need at least 3 dilution points"))
    if not block_vol or block_vol < 10:
        errors.append(("block-vol", "Block volume must be ≥ 10 µL"))
    if not stock_vol or stock_vol < 10:
        errors.append(("stock-vol", "Stock plate volume must be ≥ 10 µL"))
    if not conc_factor or conc_factor < 2:
        errors.append(("conc-factor", "Concentration factor must be ≥ 2"))
    if block_vol and stock_vol and factor:
        xfer = round(stock_vol / (factor - 1), 2)
        if xfer > stock_vol:
            errors.append(("stock-vol", f"Transfer vol ({xfer} µL) exceeds stock vol ({stock_vol} µL). "
                                         "Increase stock vol or decrease dilution factor."))
    return errors


def validate_cell_params(stock, cpw, wv, wells, dead):
    """Validate cell seeding parameters. Returns list of (field_id, message) tuples."""
    errors = []
    if not stock or stock <= 0:
        errors.append(("stock", "Stock density must be > 0"))
    if not cpw or cpw <= 0:
        errors.append(("cpw", "Cells/well must be > 0"))
    if not wv or wv <= 0:
        errors.append(("wv", "Volume per well must be > 0"))
    if not wells or wells <= 0:
        errors.append(("wells", "Total wells must be > 0"))
    if stock and cpw and wv:
        req = cpw / (wv / 1000)  # cells/mL needed
        if req > stock:
            errors.append(("stock", f"Stock density ({stock:,.0f}/mL) is less than required "
                                     f"({req:,.0f}/mL). Increase stock or decrease cells/well."))
    if cpw and stock and cpw > stock:
        errors.append(("cpw", f"Cells/well ({cpw:,}) exceeds stock density – physically impossible"))
    if dead and dead < 0:
        errors.append(("dead", "Dead volume cannot be negative"))
    return errors


# ═══════════════════════════════════════════════════════════════════
# §3  STATISTICAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def grubbs_test(data, alpha=GRUBBS_ALPHA):
    """Iterative Grubbs test. Returns (cleaned_data, outlier_indices)."""
    data = np.array(data, dtype=float)
    outlier_idx = []
    working = list(range(len(data)))
    while len(working) >= 3:
        vals = data[working]
        n = len(vals)
        mean, std = np.mean(vals), np.std(vals, ddof=1)
        if std < 1e-12:
            break
        G = np.max(np.abs(vals - mean)) / std
        suspect_local = int(np.argmax(np.abs(vals - mean)))
        # Critical value from t-distribution
        t_crit = sp_stats.t.ppf(1 - alpha / (2 * n), n - 2)
        G_crit = (n - 1) / np.sqrt(n) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        if G > G_crit:
            outlier_idx.append(working[suspect_local])
            working.pop(suspect_local)
        else:
            break
    cleaned = [data[i] for i in range(len(data)) if i not in outlier_idx]
    return cleaned, outlier_idx


def weighted_mean_sd(values, weights=None):
    """Compute weighted mean and SD. If no weights, uses equal weighting."""
    v = np.array(values, dtype=float)
    if weights is None or len(weights) != len(v):
        return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    w = np.array(weights, dtype=float)
    wm = np.average(v, weights=w)
    wvar = np.average((v - wm)**2, weights=w)
    return float(wm), float(np.sqrt(wvar)) if wvar > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# §4  ROBUST 4PL FITTING
# ═══════════════════════════════════════════════════════════════════

def fourpl(x, top, bottom, ec50, hill):
    return bottom + (top - bottom) / (1.0 + (x / ec50) ** hill)


def fit_4pl_robust(concs, means, sds=None) -> FitResult:
    """
    Weighted 4PL fit with full diagnostics.
    Uses inverse-variance weighting when SDs are available.
    Returns FitResult with status, params, R², adjusted R², EC50 CI.
    """
    # Filter valid points (positive conc, finite mean)
    valid = [(c, m, s) for c, m, s in
             zip(concs, means, sds if sds else [1.0]*len(means))
             if c > 0 and np.isfinite(m)]
    if len(valid) < 4:
        return FitResult(status=FitStatus.INSUFFICIENT_POINTS,
                         message=f"Only {len(valid)} valid points (need ≥ 4)")

    x = np.array([v[0] for v in valid])
    y = np.array([v[1] for v in valid])
    s = np.array([v[2] for v in valid])
    n = len(x)

    # Check for flat curve before fitting
    dynamic_range = abs(max(y) - min(y))
    if dynamic_range < QC_DYNAMIC_RANGE_MIN:
        return FitResult(status=FitStatus.FLAT_CURVE,
                         message=f"Dynamic range {dynamic_range:.1f}% < {QC_DYNAMIC_RANGE_MIN}% threshold")

    # Hook effect detection: check if response increases again at highest concentrations
    sorted_idx = np.argsort(x)[::-1]  # descending conc
    y_sorted = y[sorted_idx]
    if len(y_sorted) >= 4:
        top3 = y_sorted[:3]
        if top3[0] > top3[1] and top3[0] > top3[2] and (top3[0] - min(top3[1:])) > dynamic_range * 0.2:
            return FitResult(status=FitStatus.HOOK_EFFECT,
                             message="Signal increases at highest concentrations – possible hook effect")

    # Weights: inverse variance (with floor to avoid division by zero)
    sigma = None
    if sds is not None and any(si > 0 for si in s):
        sigma = np.maximum(s, np.max(s) * 0.01)  # floor at 1% of max SD

    # Initial guesses — detect curve direction for Hill slope
    # If y decreases with x (killing/inhibition), Hill should be positive
    # If y increases with x (stimulation), Hill should be negative
    sorted_by_x = sorted(zip(x, y), key=lambda t: t[0])
    y_low_x = np.mean([v[1] for v in sorted_by_x[:max(1, n//3)]])
    y_high_x = np.mean([v[1] for v in sorted_by_x[-max(1, n//3):]])
    hill_guess = 1.0 if y_low_x > y_high_x else -1.0  # positive = decreasing curve
    p0 = [max(y), min(y), np.median(x), hill_guess]
    bounds = ([0, 0, 1e-6, -10], [200, 200, 1e8, 10])

    try:
        popt, pcov = curve_fit(fourpl, x, y, p0=p0, sigma=sigma,
                               absolute_sigma=(sigma is not None),
                               bounds=bounds, maxfev=15000)
    except (RuntimeError, ValueError, TypeError) as e:
        return FitResult(status=FitStatus.CONVERGENCE_FAILED, message=str(e))

    top, bottom, ec50, hill = popt
    y_pred = fourpl(x, *popt)
    residuals = (y - y_pred).tolist()

    # R² and adjusted R²
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 4 - 1) if n > 5 else r2

    # EC50 confidence interval from covariance matrix
    ec50_ci = None
    try:
        se = np.sqrt(np.diag(pcov))
        if np.isfinite(se[2]) and se[2] > 0:
            t_val = sp_stats.t.ppf(0.975, n - 4)
            ec50_ci = (max(0, ec50 - t_val * se[2]), ec50 + t_val * se[2])
    except (ValueError, IndexError):
        pass

    # Determine status
    if r2 < QC_R2_MIN:
        status = FitStatus.POOR_FIT
        msg = f"R² = {r2:.4f} < {QC_R2_MIN} threshold"
    else:
        status = FitStatus.SUCCESS
        msg = f"R² = {r2:.4f}, adj-R² = {adj_r2:.4f}"

    return FitResult(
        status=status, params=(top, bottom, ec50, hill),
        r_squared=r2, adj_r_squared=adj_r2,
        ec50_ci=ec50_ci, residuals=residuals, message=msg)


# ═══════════════════════════════════════════════════════════════════
# §5  QC ENGINE
# ═══════════════════════════════════════════════════════════════════

def qc_replicate_group(reps, conc, ab_name, well_label="") -> Tuple[List[float], List[int], List[QCFlag]]:
    """
    QC a single replicate group:
    - Grubbs outlier removal
    - CV% check
    Returns (cleaned_reps, outlier_indices, flags).
    """
    flags = []
    if len(reps) < 2:
        return reps, [], flags

    cleaned, outliers = grubbs_test(reps)
    for oi in outliers:
        flags.append(QCFlag("warn", "outlier",
            f"{ab_name} @ {conc:.2g} nM: Rep {oi+1} = {reps[oi]:.1f} flagged as outlier (Grubbs p<{GRUBBS_ALPHA})",
            well_label))

    if len(cleaned) >= 2:
        mn = np.mean(cleaned)
        sd = np.std(cleaned, ddof=1)
        cv = (sd / mn * 100) if mn > 0 else 0
        if cv > QC_CV_THRESHOLD:
            flags.append(QCFlag("warn", "cv",
                f"{ab_name} @ {conc:.2g} nM: CV = {cv:.1f}% exceeds {QC_CV_THRESHOLD}% threshold",
                well_label))

    return cleaned, outliers, flags


def qc_curve_level(concs, means, fit_result: FitResult, ab_name: str) -> List[QCFlag]:
    """Curve-level QC: edge effects, fit quality, hook effect."""
    flags = []

    if fit_result.status == FitStatus.HOOK_EFFECT:
        flags.append(QCFlag("warn", "hook", f"{ab_name}: {fit_result.message}"))
    elif fit_result.status == FitStatus.FLAT_CURVE:
        flags.append(QCFlag("warn", "flat", f"{ab_name}: {fit_result.message}"))
    elif fit_result.status == FitStatus.POOR_FIT:
        flags.append(QCFlag("warn", "fit", f"{ab_name}: {fit_result.message}"))
    elif fit_result.status in (FitStatus.CONVERGENCE_FAILED, FitStatus.INSUFFICIENT_POINTS):
        flags.append(QCFlag("fail", "fit", f"{ab_name}: {fit_result.message}"))

    # Edge effect: check if outermost concentrations have unusually high residuals
    if fit_result.residuals and len(fit_result.residuals) >= 5:
        res = np.array(fit_result.residuals)
        res_std = np.std(res)
        if res_std > 0:
            if abs(res[0]) > 2.5 * res_std:
                flags.append(QCFlag("warn", "edge",
                    f"{ab_name}: Highest conc residual ({res[0]:.1f}) is >2.5 SD — possible edge effect"))
            if abs(res[-1]) > 2.5 * res_std:
                flags.append(QCFlag("warn", "edge",
                    f"{ab_name}: Lowest conc residual ({res[-1]:.1f}) is >2.5 SD — possible edge effect"))

    return flags


# ═══════════════════════════════════════════════════════════════════
# §6  REFERENCE STANDARD & RELATIVE POTENCY
# ═══════════════════════════════════════════════════════════════════

def compute_relative_potency(ref_ec50, test_ec50):
    """Relative potency = ref_EC50 / test_EC50 × 100%."""
    if ref_ec50 and test_ec50 and test_ec50 > 0:
        return round(ref_ec50 / test_ec50 * 100, 1)
    return None


# ═══════════════════════════════════════════════════════════════════
# §7  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def mg_nM(mg, mw): return mg / mw * 1e6 if mw and mw > 0 else 0
def nM_mg(nM, mw): return nM * mw / 1e6 if mw and mw > 0 else 0

def fmt_nM(v):
    if v <= 0: return "Bkg"
    if v >= 1000: return f"{v/1000:.3g}µM"
    return f"{v:.3g}nM" if v >= 1 else f"{v:.2g}nM"

def hex_rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


# ═══════════════════════════════════════════════════════════════════
# §8  UNIFIED DATA PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════

def build_raw_table(sim_data, ab_name=None):
    names = [ab_name] if ab_name else list(dict.fromkeys(r["Antibody"] for r in sim_data))
    concs = sorted(set(r["Conc_nM"] for r in sim_data), reverse=True)
    tbl = []
    for nm in names:
        for c in concs:
            reps = [r["Viability_%"] for r in sim_data if r["Antibody"] == nm and abs(r["Conc_nM"] - c) < 0.001]
            if len(reps) >= 2:
                mn, sd = np.mean(reps), np.std(reps, ddof=1)
                tbl.append({"Antibody": nm, "Conc (nM)": round(c, 4) if c > 0 else "Bkg",
                            **{f"Rep {i+1}": reps[i] for i in range(len(reps))},
                            "Mean": round(mn, 2), "SD": round(sd, 2),
                            "CV%": round(sd / mn * 100, 1) if mn > 0 else 0})
    return tbl


def run_4pl_analysis_robust(ab_list, sim_data, concs, ref_ab_name=None):
    """
    Unified 4PL analysis with QC flags, outlier detection, weighted fitting,
    and optional reference standard normalization.
    """
    all_flags = []
    analysis = []
    ref_ec50 = None

    for ab in ab_list:
        ad = [r for r in sim_data if r["Antibody"] == ab["name"]]
        ms, cs, sds_list = [], [], []
        for c in concs:
            reps = [r["Viability_%"] for r in ad if abs(r["Conc_nM"] - c) < 0.001]
            if len(reps) >= 2 and c > 0:
                # QC each replicate group with Grubbs
                cleaned, outliers, flags = qc_replicate_group(reps, c, ab["name"])
                all_flags.extend(flags)
                if len(cleaned) >= 2:
                    mn = np.mean(cleaned)
                    sd = np.std(cleaned, ddof=1)
                    ms.append(mn); cs.append(c); sds_list.append(sd)

        # Robust 4PL fit with weights
        fit = fit_4pl_robust(cs, ms, sds_list if sds_list else None)

        # Curve-level QC
        curve_flags = qc_curve_level(cs, ms, fit, ab["name"])
        all_flags.extend(curve_flags)

        if fit.params:
            top, bottom, ec50, hill = fit.params
            # Track reference EC50
            if ab["name"] == ref_ab_name:
                ref_ec50 = ec50
            rel_pot = None
            ci_str = "N/A"
            if fit.ec50_ci:
                ci_str = f"({fit.ec50_ci[0]:.2f} – {fit.ec50_ci[1]:.2f})"
        else:
            top = bottom = ec50 = hill = None
            rel_pot = None; ci_str = "N/A"

        qc_summary = _qc_badge(fit.status)

        entry = {
            "Antibody": ab["name"], "Control": ab.get("ctrl_type", "Positive"),
            "EC50 (nM)": round(ec50, 3) if ec50 else "N/A",
            "EC50 95% CI": ci_str,
            "Top (%)": round(top, 1) if top is not None else "N/A",
            "Bottom (%)": round(bottom, 1) if bottom is not None else "N/A",
            "Hill Slope": round(hill, 3) if hill is not None else "N/A",
            "R²": round(fit.r_squared, 4) if fit.r_squared else "N/A",
            "Adj R²": round(fit.adj_r_squared, 4) if fit.adj_r_squared else "N/A",
            "Max Kill (%)": round(top - bottom, 1) if top is not None and bottom is not None else "N/A",
            "Fit Status": qc_summary,
            "_fit_status_raw": fit.status.value,
            "QC": qc_summary,
            "Rel Potency (%)": "–",
            "_color": ab["color"],
            "_params": list(fit.params) if fit.params else None,
            "_concs": cs, "_means": ms, "_sds": sds_list,
            "_fit_result": fit.to_dict(),
        }
        analysis.append(entry)

    # Second pass: compute relative potency if reference is set
    if ref_ec50 and ref_ab_name:
        for entry in analysis:
            if entry["_params"] and entry["Antibody"] != ref_ab_name:
                rp = compute_relative_potency(ref_ec50, entry["_params"][2])
                entry["Rel Potency (%)"] = f"{rp}%" if rp else "N/A"
            elif entry["Antibody"] == ref_ab_name:
                entry["Rel Potency (%)"] = "100% (ref)"

    return analysis, all_flags


def _qc_badge(status: FitStatus) -> str:
    label, _ = FIT_STATUS_LABELS[status]
    return label


def make_dose_fig(ab_list, analysis, per_row=4):
    na = len(ab_list); nc = min(per_row, na); nr = math.ceil(na / nc) if na > 0 else 1
    titles = [a["name"][:22] for a in ab_list]
    fig = make_subplots(rows=nr, cols=nc, subplot_titles=titles,
                        horizontal_spacing=0.06, vertical_spacing=0.06)

    # ── Compute SHARED x-axis range from all concentration data ──
    all_concs = []
    for ad in analysis:
        if ad.get("_concs"):
            all_concs.extend([c for c in ad["_concs"] if c > 0])
    if all_concs:
        global_xmin = min(all_concs) * 0.3
        global_xmax = max(all_concs) * 3
    else:
        global_xmin, global_xmax = 0.1, 1000

    for idx, (ab, ad) in enumerate(zip(ab_list, analysis)):
        ri, ci = divmod(idx, nc); ri += 1; ci += 1
        if ad.get("_concs"):
            error_y = None
            if ad.get("_sds"):
                error_y = dict(type="data", array=ad["_sds"], visible=True,
                               color=hex_rgba(ab["color"], 0.4), thickness=1.5)
            fig.add_trace(go.Scatter(x=ad["_concs"], y=ad["_means"], mode="markers",
                error_y=error_y,
                marker=dict(size=7, color=ab["color"], line=dict(width=1, color="white")),
                showlegend=False, hovertemplate="%{x:.2f} nM<br>%{y:.1f}%<extra></extra>"), row=ri, col=ci)
            if ad.get("_params"):
                # Use shared x range for fit curve
                xf = np.logspace(np.log10(global_xmin), np.log10(global_xmax), 100)
                yf = [fourpl(x, *ad["_params"]) for x in xf]
                line_color = ab["color"]
                _raw_st = ad.get("_fit_status_raw", ad.get("Fit Status", "success"))
                try:
                    fit_status = FitStatus(_raw_st)
                except ValueError:
                    fit_status = FitStatus.SUCCESS
                if fit_status in (FitStatus.POOR_FIT, FitStatus.HOOK_EFFECT):
                    line_color = "#f59e0b"
                fig.add_trace(go.Scatter(x=xf.tolist(), y=yf, mode="lines",
                    line=dict(color=line_color, width=2,
                              dash="dash" if fit_status != FitStatus.SUCCESS else "solid"),
                    showlegend=False), row=ri, col=ci)
                ev = ad["_params"][2]; ey = fourpl(ev, *ad["_params"])
                # Only show EC50 diamond if within visible range
                if global_xmin <= ev <= global_xmax:
                    fig.add_trace(go.Scatter(x=[ev], y=[ey], mode="markers",
                        marker=dict(size=10, symbol="diamond", color="red", line=dict(width=1.5, color="white")),
                        showlegend=False, hovertemplate=f"EC50: {ev:.2f} nM<extra></extra>"), row=ri, col=ci)
                # CI shading — CLAMP to data range to prevent axis blowout
                ci_data = ad.get("_fit_result", {}).get("ec50_ci") if ad.get("_fit_result") else None
                if ci_data and len(ci_data) == 2:
                    ci_lo = max(ci_data[0], global_xmin * 0.5)
                    ci_hi = min(ci_data[1], global_xmax * 2)
                    if ci_lo < ci_hi and ci_hi / ci_lo < 1e4:
                        fig.add_vrect(x0=ci_lo, x1=ci_hi, fillcolor=ab["color"], opacity=0.07,
                                      line_width=0, row=ri, col=ci)
            # ── Fixed x-axis range (shared across all subplots) ──
            fig.update_xaxes(type="log", range=[np.log10(global_xmin), np.log10(global_xmax)],
                             showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=8), row=ri, col=ci)
            fig.update_yaxes(range=[-5, 115], showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=8), row=ri, col=ci)
    fig.update_layout(height=max(400, nr * 320), plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=50, r=20, t=40, b=30), font=dict(size=10))
    for ann in fig["layout"]["annotations"]: ann["font"] = dict(size=10, color="#334155")
    return fig


def make_overlay_fig(ab_list, analysis):
    """
    Single overlay dose-response figure with all antibodies.
    Plotly legend click toggles visibility — ideal for presentation.
    """
    fig = go.Figure()

    # ── Compute shared x-axis range ──
    all_concs = []
    for ad in analysis:
        if ad.get("_concs"):
            all_concs.extend([c for c in ad["_concs"] if c > 0])
    if all_concs:
        global_xmin = min(all_concs) * 0.3
        global_xmax = max(all_concs) * 3
    else:
        global_xmin, global_xmax = 0.1, 1000

    # Categorize by potency for legend ordering
    entries = []
    for ab, ad in zip(ab_list, analysis):
        ec50_val = ad.get("EC50 (nM)")
        ec50_num = ec50_val if isinstance(ec50_val, (int, float)) else 1e9
        entries.append((ab, ad, ec50_num))
    entries.sort(key=lambda x: x[2])  # sort by EC50 (most potent first)

    for ab, ad, ec50_num in entries:
        if not ad.get("_concs"):
            continue
        color = ab["color"]
        ctrl = ad.get("Control", ab.get("ctrl_type", "Positive"))
        is_neg = (ctrl == "Negative")
        ec50_str = ad.get("EC50 (nM)", "N/A")
        ec50_label = f" (EC50={ec50_str} nM)" if isinstance(ec50_str, (int, float)) else " (Neg Ctrl)" if is_neg else ""
        legend_name = f"{ab['name']}{ec50_label}"

        # Data points
        error_y = None
        if ad.get("_sds"):
            error_y = dict(type="data", array=ad["_sds"], visible=True,
                           color=hex_rgba(color, 0.3), thickness=1)
        fig.add_trace(go.Scatter(
            x=ad["_concs"], y=ad["_means"], mode="markers",
            error_y=error_y,
            marker=dict(size=8, color=color, line=dict(width=1, color="white"),
                        symbol="circle" if not is_neg else "x"),
            name=legend_name, legendgroup=ab["name"],
            showlegend=True,
            visible=True if not is_neg else "legendonly",
            hovertemplate=f"<b>{ab['name']}</b><br>%{{x:.2f}} nM<br>%{{y:.1f}}%<extra></extra>"))

        # Fit curve — use shared x range
        if ad.get("_params"):
            xf = np.logspace(np.log10(global_xmin), np.log10(global_xmax), 200)
            yf = [fourpl(x, *ad["_params"]) for x in xf]
            _raw_st = ad.get("_fit_status_raw", ad.get("Fit Status", "success"))
            try:
                fit_status = FitStatus(_raw_st)
            except ValueError:
                fit_status = FitStatus.SUCCESS
            fig.add_trace(go.Scatter(
                x=xf.tolist(), y=yf, mode="lines",
                line=dict(color=color, width=2.5,
                          dash="dash" if fit_status != FitStatus.SUCCESS else "solid"),
                name=legend_name, legendgroup=ab["name"],
                showlegend=False,
                visible=True if not is_neg else "legendonly",
                hoverinfo="skip"))

            # EC50 diamond marker — only if within visible range
            ev = ad["_params"][2]; ey = fourpl(ev, *ad["_params"])
            if global_xmin <= ev <= global_xmax:
                fig.add_trace(go.Scatter(
                    x=[ev], y=[ey], mode="markers",
                    marker=dict(size=11, symbol="diamond", color=color,
                                line=dict(width=2, color="white")),
                    name=legend_name, legendgroup=ab["name"],
                    showlegend=False,
                    visible=True if not is_neg else "legendonly",
                    hovertemplate=f"<b>{ab['name']}</b><br>EC50: {ev:.3f} nM<extra></extra>"))

    fig.update_xaxes(
        type="log", range=[np.log10(global_xmin), np.log10(global_xmax)],
        showgrid=True, gridcolor="#f0f0f0",
        title=dict(text="Concentration (nM)", font=dict(size=13)),
        tickfont=dict(size=11))
    fig.update_yaxes(
        range=[-5, 120], showgrid=True, gridcolor="#f0f0f0",
        title=dict(text="% Viability (normalized to Bkg)", font=dict(size=13)),
        tickfont=dict(size=11))
    fig.update_layout(
        height=650, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=20, t=50, b=60),
        font=dict(size=11),
        legend=dict(
            orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02,
            font=dict(size=10), bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0", borderwidth=1,
            itemclick="toggle", itemdoubleclick="toggleothers",
            title=dict(text="Click to toggle • Double-click to isolate",
                       font=dict(size=9, color="#64748b"))),
        title=dict(
            text="All Antibodies — Interactive Dose-Response Overlay",
            font=dict(size=15, color="#334155"), x=0.01),
        hovermode="closest",
    )
    return fig


def process_rlu_data(raw, concs, names, ref_name=None):
    """Process raw RLU dict → tables, 4PL, EC50 with full QC pipeline."""
    n_prot = len(names)
    all_flags = []

    # ── Mean / SD table ──
    mean_cols = [{"field": "Point", "width": 65, "cellStyle": {"fontFamily": MF, "fontWeight": "700"}}]
    for nm in names:
        mean_cols += [
            {"field": f"{nm} Mean", "width": 120, "valueFormatter": FMT_AUTO,
             "cellStyle": {"fontFamily": MF, "fontWeight": "600", "color": "#3b82f6"}},
            {"field": f"{nm} SD", "width": 100, "valueFormatter": FMT_AUTO,
             "cellStyle": {"fontFamily": MF, "color": "#64748b"}}
        ]
    mean_rows, means_by = [], {pi: [] for pi in range(n_prot)}
    for ci, c in enumerate(concs):
        row = {"Point": ci + 1}
        for pi, nm in enumerate(names):
            reps = raw.get((pi, ci), [])
            if reps:
                # Outlier detection on each replicate group
                cleaned, outliers, flags = qc_replicate_group(reps, c, nm)
                all_flags.extend(flags)
                mn = np.mean(cleaned) if cleaned else np.mean(reps)
                sd = np.std(cleaned, ddof=1) if len(cleaned) > 1 else 0
                row[f"{nm} Mean"], row[f"{nm} SD"] = round(mn, 1), round(sd, 1)
                means_by[pi].append((c, mn, sd))
        mean_rows.append(row)

    # ── Background with outlier check ──
    bg = {}
    for pi in range(n_prot):
        bg_vals = [m for c, m, s in means_by[pi] if c <= 0]
        if bg_vals:
            bg_cleaned, bg_outliers = grubbs_test(bg_vals)
            bg[pi] = np.mean(bg_cleaned) if bg_cleaned else 1
            for oi in bg_outliers:
                all_flags.append(QCFlag("warn", "outlier",
                    f"{names[pi]}: Background replicate {oi+1} flagged as outlier"))
        else:
            bg[pi] = 1

    # ── Normalized table ──
    norm_cols = [{"field": "Log(nM)", "width": 80, "valueFormatter": FMT_DEC2,
                  "cellStyle": {"fontFamily": MF, "fontWeight": "700"}}]
    for nm in names:
        norm_cols += [
            {"field": f"{nm} Mean%", "width": 100, "valueFormatter": FMT_DEC1,
             "cellStyle": {"fontFamily": MF, "fontWeight": "600", "color": "#10b981"}},
            {"field": f"{nm} CV%", "width": 85, "valueFormatter": FMT_DEC1,
             "cellStyle": {"fontFamily": MF}},
            {"field": f"{nm} N", "width": 50, "cellStyle": {"fontFamily": MF, "color": "#94a3b8"}}
        ]
    norm_rows, norm_d = [], {pi: {"c": [], "m": [], "sd": [], "cv": []} for pi in range(n_prot)}
    for ci, c in enumerate(concs):
        log_c = round(math.log10(c), 5) if c > 0 else -3
        row = {"Log(nM)": log_c}
        for pi, nm in enumerate(names):
            reps = raw.get((pi, ci), [])
            if reps:
                cleaned, _, _ = qc_replicate_group(reps, c, nm)
                mn = np.mean(cleaned) if cleaned else np.mean(reps)
                sd = np.std(cleaned, ddof=1) if len(cleaned) > 1 else 0
                pct = mn / bg[pi] * 100 if bg[pi] > 0 else 0
                sd_pct = sd / bg[pi] * 100 if bg[pi] > 0 else 0
                cv = sd / mn * 100 if mn > 0 else 0
                row[f"{nm} Mean%"] = round(pct, 1)
                row[f"{nm} CV%"] = round(cv, 1)
                row[f"{nm} N"] = len(cleaned)
                if c > 0:
                    norm_d[pi]["c"].append(c)
                    norm_d[pi]["m"].append(pct)
                    norm_d[pi]["sd"].append(sd_pct)
                    norm_d[pi]["cv"].append(cv)
        norm_rows.append(row)

    # ── 4PL per protein with robust fitting ──
    nc_ = min(4, n_prot); nr_ = math.ceil(n_prot / nc_) if n_prot > 0 else 1
    fig = make_subplots(rows=nr_, cols=nc_, subplot_titles=names,
                        horizontal_spacing=0.06, vertical_spacing=0.08)
    ec50_rows = []
    ref_ec50 = None

    # ── Compute SHARED x-axis range from all proteins ──
    all_concs_flat = []
    for pi in range(n_prot):
        all_concs_flat.extend([c for c in norm_d[pi]["c"] if c > 0])
    if all_concs_flat:
        global_xmin = min(all_concs_flat) * 0.3
        global_xmax = max(all_concs_flat) * 3
    else:
        global_xmin, global_xmax = 0.1, 1000

    for pi, nm in enumerate(names):
        ri, ci = divmod(pi, nc_); ri += 1; ci += 1
        cs = norm_d[pi]["c"]
        ms = norm_d[pi]["m"]
        sds = norm_d[pi]["sd"]
        color = AB_COLORS[pi % len(AB_COLORS)]

        if cs:
            fig.add_trace(go.Scatter(x=cs, y=ms, mode="markers",
                error_y=dict(type="data", array=sds, visible=True,
                             color=hex_rgba(color, 0.4), thickness=1.5),
                marker=dict(size=8, color=color, line=dict(width=1.5, color="white")),
                showlegend=False, hovertemplate="%{x:.2f} nM<br>%{y:.1f}%<extra></extra>"),
                row=ri, col=ci)

            fit = fit_4pl_robust(cs, ms, sds)
            curve_flags = qc_curve_level(cs, ms, fit, nm)
            all_flags.extend(curve_flags)

            ec_row = {"Protein": nm, "Fit Status": _qc_badge(fit.status),
                      "_fit_status_raw": fit.status.value,
                      "_color": color, "_concs": cs, "_means": ms, "_sds": sds,
                      "_params": list(fit.params) if fit.params else None,
                      "_fit_result": fit.to_dict(), "Control": "Positive"}

            if fit.params:
                top, bottom, ec50, hill = fit.params
                if nm == ref_name:
                    ref_ec50 = ec50

                # Use shared x range for fit curve
                xf = np.logspace(np.log10(global_xmin), np.log10(global_xmax), 100)
                yf = [fourpl(x, *fit.params) for x in xf]
                line_dash = "dash" if fit.status != FitStatus.SUCCESS else "solid"
                fig.add_trace(go.Scatter(x=xf.tolist(), y=yf, mode="lines",
                    line=dict(color=color, width=2.5, dash=line_dash), showlegend=False),
                    row=ri, col=ci)

                ev = ec50; ey = fourpl(ev, *fit.params)
                # Only show EC50 diamond if within visible range
                if global_xmin <= ev <= global_xmax:
                    fig.add_trace(go.Scatter(x=[ev], y=[ey], mode="markers",
                        marker=dict(size=12, symbol="diamond", color="#ef4444",
                                    line=dict(width=2, color="white")),
                        showlegend=False, hovertemplate=f"EC50: {ev:.2f} nM<extra></extra>"),
                        row=ri, col=ci)

                # CI shading — CLAMP to data range
                if fit.ec50_ci:
                    ci_lo = max(fit.ec50_ci[0], global_xmin * 0.5)
                    ci_hi = min(fit.ec50_ci[1], global_xmax * 2)
                    if ci_lo < ci_hi and ci_hi / ci_lo < 1e4:
                        fig.add_vrect(x0=ci_lo, x1=ci_hi,
                                      fillcolor=color, opacity=0.07, line_width=0,
                                      row=ri, col=ci)

                ci_str = f"({fit.ec50_ci[0]:.2f}–{fit.ec50_ci[1]:.2f})" if fit.ec50_ci else "N/A"
                ec_row.update({
                    "EC50 (nM)": round(ec50, 3), "EC50 95% CI": ci_str,
                    "Top (%)": round(top, 1), "Bottom (%)": round(bottom, 1),
                    "Hill Slope": round(hill, 3),
                    "R²": round(fit.r_squared, 4), "Adj R²": round(fit.adj_r_squared, 4),
                    "Max Kill (%)": round(top - bottom, 1),
                    "Avg CV%": round(np.mean(norm_d[pi]["cv"]), 1) if norm_d[pi]["cv"] else "N/A",
                })
            else:
                ec_row.update({k: "N/A" for k in
                    ["EC50 (nM)", "EC50 95% CI", "Top (%)", "Bottom (%)",
                     "Hill Slope", "R²", "Adj R²", "Max Kill (%)", "Avg CV%"]})
            ec50_rows.append(ec_row)

            # ── Fixed x-axis range (shared across all subplots) ──
            fig.update_xaxes(type="log", range=[np.log10(global_xmin), np.log10(global_xmax)],
                             showgrid=True, gridcolor="#f0f0f0",
                             tickfont=dict(size=9), title=dict(text="nM", font=dict(size=10)),
                             row=ri, col=ci)
            fig.update_yaxes(range=[-5, 130], showgrid=True, gridcolor="#f0f0f0",
                             tickfont=dict(size=9), title=dict(text="% viability", font=dict(size=10)),
                             row=ri, col=ci)

    # Relative potency pass
    if ref_ec50 and ref_name:
        for row in ec50_rows:
            ec = row.get("EC50 (nM)")
            if isinstance(ec, (int, float)) and ec > 0:
                if row["Protein"] == ref_name:
                    row["Rel Potency (%)"] = "100% (ref)"
                else:
                    rp = compute_relative_potency(ref_ec50, ec)
                    row["Rel Potency (%)"] = f"{rp}%" if rp else "N/A"
            else:
                row["Rel Potency (%)"] = "N/A"

    fig.update_layout(height=max(400, nr_ * 320), plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=50, r=20, t=40, b=30), font=dict(size=10))
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12, color="#334155")

    return mean_cols, mean_rows, norm_cols, norm_rows, fig, ec50_rows, all_flags


# ═══════════════════════════════════════════════════════════════════
# §10  STYLES
# ═══════════════════════════════════════════════════════════════════

C = {"bg": "#f1f5f9", "sf": "#ffffff", "sa": "#f8fafc", "bd": "#e2e8f0",
     "tx": "#0f172a", "mt": "#64748b", "ac": "#3b82f6", "al": "#eff6ff",
     "dg": "#ef4444", "wb": "#fffbeb", "wbd": "#fde68a", "wt": "#92400e",
     "ok": "#10b981", "ol": "#ecfdf5"}

FF = "'Segoe UI','Helvetica Neue',Arial,sans-serif"
MF = "'Consolas','Courier New',monospace"

card  = {"background": C["sf"], "border": f"1px solid {C['bd']}", "borderRadius": "10px",
         "boxShadow": "0 1px 3px rgba(15,23,42,.06)", "overflow": "hidden"}
chdr  = {"padding": "14px 20px", "borderBottom": f"1px solid {C['bd']}", "fontWeight": "600",
         "fontSize": "13px", "letterSpacing": ".5px", "color": C["mt"],
         "background": C["sa"], "display": "flex", "alignItems": "center", "gap": "8px"}
lbl   = {"fontSize": "11px", "fontWeight": "600", "color": C["mt"],
         "letterSpacing": ".4px", "marginBottom": "4px"}
inp   = {"width": "100%", "padding": "8px 12px", "border": f"1px solid {C['bd']}",
         "borderRadius": "6px", "fontSize": "14px", "fontFamily": FF, "color": C["tx"]}
inpm  = {**inp, "fontFamily": MF, "fontWeight": "600"}
btn   = {"padding": "10px 20px", "borderRadius": "6px", "fontSize": "13px", "fontWeight": "600",
         "cursor": "pointer", "border": "none", "background": C["ac"], "color": "white", "fontFamily": FF}
btno  = {"padding": "7px 14px", "borderRadius": "6px", "fontSize": "12px", "fontWeight": "600",
         "cursor": "pointer", "border": f"1px solid {C['bd']}", "background": "transparent",
         "color": C["tx"], "fontFamily": FF}
upz   = {"border": f"2px dashed {C['bd']}", "borderRadius": "8px", "padding": "20px",
         "textAlign": "center", "cursor": "pointer", "background": C["sa"]}
ts    = {"fontSize": "14px", "fontWeight": "600", "padding": "14px 24px"}
tsel  = {**ts, "borderTop": f"3px solid {C['ac']}", "color": C["ac"]}
ibox  = {"background": C["ol"], "border": f"1px solid {C['ok']}", "borderRadius": "6px",
         "padding": "8px 14px", "fontSize": "12px", "color": "#065f46", "fontFamily": MF}
err_box = {"background": "#fef2f2", "border": "1px solid #fecaca", "borderRadius": "6px",
           "padding": "10px 14px", "fontSize": "12px", "color": "#991b1b", "fontWeight": "500"}
warn_box = {"background": C["wb"], "border": f"1px solid {C['wbd']}", "borderRadius": "6px",
            "padding": "10px 14px", "fontSize": "12px", "color": C["wt"]}
qc_card = {"background": "#faf5ff", "border": "1px solid #e9d5ff", "borderRadius": "8px",
           "padding": "12px 16px", "marginTop": "16px"}

def minp(id, val, **kw): return dcc.Input(id=id, value=val, type="number", style=inpm, **kw)

# AG Grid number formatters (JS)
FMT_INT  = {"function": "params.value != null && !isNaN(params.value) ? Number(params.value).toLocaleString('en-US',{maximumFractionDigits:0}) : params.value"}
FMT_DEC1 = {"function": "params.value != null && !isNaN(params.value) ? Number(params.value).toLocaleString('en-US',{minimumFractionDigits:1,maximumFractionDigits:1}) : params.value"}
FMT_DEC2 = {"function": "params.value != null && !isNaN(params.value) ? Number(params.value).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}) : params.value"}
FMT_DEC3 = {"function": "params.value != null && !isNaN(params.value) ? Number(params.value).toLocaleString('en-US',{minimumFractionDigits:3,maximumFractionDigits:3}) : params.value"}
FMT_DEC4 = {"function": "params.value != null && !isNaN(params.value) ? Number(params.value).toLocaleString('en-US',{minimumFractionDigits:4,maximumFractionDigits:4}) : params.value"}
FMT_AUTO = {"function": "params.value != null && !isNaN(params.value) ? Number(params.value).toLocaleString('en-US') : params.value"}


def render_qc_panel(flags):
    """Render QC flags as a styled panel."""
    if not flags:
        return html.Div([
            html.Span("✓ ", style={"color": C["ok"], "fontWeight": "700"}),
            html.Span("All QC checks passed", style={"fontWeight": "600", "color": C["ok"]})
        ], style={**ibox, "marginTop": "12px"})

    warns = [f for f in flags if f["level"] == "warn"]
    fails = [f for f in flags if f["level"] == "fail"]
    items = []
    if fails:
        items.append(html.Div([
            html.Div(f"✗ {len(fails)} FAIL", style={"fontWeight": "700", "color": C["dg"], "marginBottom": "6px"}),
            *[html.Div(f"• {f['message']}", style={"fontSize": "11px", "color": "#991b1b", "marginLeft": "12px",
                                                     "lineHeight": "1.6"}) for f in fails]
        ], style=err_box))
    if warns:
        items.append(html.Div([
            html.Div(f"⚠ {len(warns)} WARNING{'S' if len(warns)>1 else ''}",
                     style={"fontWeight": "700", "color": C["wt"], "marginBottom": "6px"}),
            *[html.Div(f"• {f['message']}", style={"fontSize": "11px", "color": C["wt"], "marginLeft": "12px",
                                                     "lineHeight": "1.6"}) for f in warns[:15]],
            *([] if len(warns) <= 15 else [html.Div(f"... and {len(warns)-15} more",
                style={"fontSize": "11px", "color": C["mt"], "marginLeft": "12px", "fontStyle": "italic"})])
        ], style=warn_box))
    return html.Div(items, style={"display": "flex", "flexDirection": "column", "gap": "10px", "marginTop": "12px"})


def render_validation_errors(errors):
    """Render validation errors as a styled alert."""
    if not errors:
        return None
    return html.Div([
        html.Div("⚠ Validation Errors", style={"fontWeight": "700", "color": "#991b1b", "marginBottom": "6px"}),
        *[html.Div(f"• {msg}", style={"fontSize": "12px", "color": "#991b1b", "marginLeft": "10px",
                                       "lineHeight": "1.7"}) for _, msg in errors]
    ], style=err_box)


# ═══════════════════════════════════════════════════════════════════
# §11  LAYOUT
# ═══════════════════════════════════════════════════════════════════

app = dash.Dash(__name__, title="Potency Assay — 384-Well (v2)", suppress_callback_exceptions=True)

app.layout = html.Div([
    # ── Stores ──
    dcc.Store(id="conc-store", data=[]),
    dcc.Store(id="ab-store", data=[]),
    dcc.Store(id="prep-store", data=[]),
    dcc.Store(id="grid-store", data=[]),
    dcc.Store(id="sim-store", data=[]),
    dcc.Store(id="analysis-store", data=[]),
    dcc.Store(id="merged-plates", data={}),   # NEW: multi-plate accumulator
    dcc.Store(id="qc-flags-store", data=[]),   # NEW: QC flags

    # ── Header ──
    html.Div([
        html.Div([
            html.H1("Potency Assay — 384-Well Plate",
                     style={"fontSize": "22px", "fontWeight": "700", "margin": "0"}),
            html.P("v2 — Robust 4PL | QC Flags | Multi-Plate | Outlier Detection",
                   style={"fontSize": "13px", "opacity": ".7", "margin": "3px 0 0 0"}),
        ]),
        html.Div([
            html.Div([html.Label("Date", style={**lbl, "color": "rgba(255,255,255,.5)"}),
                      dcc.Input(id="exp-date", type="text", placeholder="YYYY-MM-DD",
                                style={**inp, "background": "rgba(255,255,255,.1)", "color": "white",
                                       "border": "1px solid rgba(255,255,255,.2)", "width": "130px"})]),
            html.Div([html.Label("Lot", style={**lbl, "color": "rgba(255,255,255,.5)"}),
                      dcc.Input(id="lot-num", type="text", placeholder="LOT-XXX",
                                style={**inp, "background": "rgba(255,255,255,.1)", "color": "white",
                                       "border": "1px solid rgba(255,255,255,.2)", "width": "130px"})]),
            html.Div([html.Label("Cell Line", style={**lbl, "color": "rgba(255,255,255,.5)"}),
                      dcc.Input(id="cell-line", type="text", placeholder="e.g. Raji",
                                style={**inp, "background": "rgba(255,255,255,.1)", "color": "white",
                                       "border": "1px solid rgba(255,255,255,.2)", "width": "130px"})]),
        ], style={"display": "flex", "gap": "12px", "alignItems": "flex-end"}),
    ], style={"background": "linear-gradient(135deg,#0f172a,#1e3a5f)", "color": "white",
              "padding": "20px 32px", "display": "flex", "justifyContent": "space-between",
              "alignItems": "center"}),

    # ═══ TABS ═══
    html.Div([dcc.Tabs(id="main-tabs", value="tab-prep", children=[

        # ═══════ TAB 1: DILUTION BLOCK PREPARATION ═══════
        dcc.Tab(label="🧪 Dilution Block Preparation", value="tab-prep", style=ts, selected_style=tsel,
        children=[html.Div([
            html.Div([
                # Left: Upload antibodies
                html.Div([
                    html.Div("1. Load Antibodies (up to 24)", style=chdr),
                    html.Div([
                        dcc.Upload(id="reagent-upload", children=html.Div([
                            html.Div("Click or drag Excel", style={"fontSize": "13px", "fontWeight": "500", "color": C["mt"]}),
                            html.Div("AB_ID | MW (kDa) | Conc (mg/mL) | Control Type",
                                     style={"fontSize": "10px", "color": C["mt"], "marginTop": "4px", "fontFamily": MF}),
                        ], style=upz), accept=".xlsx,.xls,.csv"),
                        html.Div(id="upload-status", style={"marginTop": "8px", "fontSize": "12px"}),
                        html.Div(id="ab-summary", style={"marginTop": "10px"}),
                    ], style={"padding": "16px"}),
                ], style={**card, "flex": "1"}),

                # Right: Dilution parameters
                html.Div([
                    html.Div("2. Dilution & Stock Plate Parameters", style=chdr),
                    html.Div([
                        html.Div([
                            html.Div([html.Label("Starting Conc (nM)", style={**lbl, "color": C["ac"], "fontSize": "12px"}),
                                      minp("start-conc", 100, step=10)], style={"flex": "1"}),
                            html.Div([html.Label("Dilution Factor", style=lbl),
                                      dcc.Dropdown(id="dil-factor", options=[{"label": f"{x}x", "value": x} for x in [2,3,4,5,10]],
                                                   value=2, clearable=False)], style={"flex": "1"}),
                            html.Div([html.Label("# Points", style=lbl),
                                      dcc.Dropdown(id="n-dil", options=[{"label": str(x), "value": x} for x in range(5, 11)],
                                                   value=7, clearable=False)], style={"flex": "1"}),
                        ], style={"display": "flex", "gap": "10px", "marginBottom": "12px"}),
                        html.Div([
                            html.Div([html.Label("Block Vol (µL)", style=lbl), minp("block-vol", 500, step=50)], style={"flex": "1"}),
                            html.Div([html.Label("Culture Vol (µL)", style=lbl), minp("cult-vol", 50, step=5)], style={"flex": "1"}),
                            html.Div([html.Label("Add Into Culture (µL)", style=lbl), minp("add-vol", 10, step=1)], style={"flex": "1"}),
                        ], style={"display": "flex", "gap": "10px", "marginBottom": "12px"}),
                        html.Div([
                            html.Div([html.Label("Stock Plate Vol (µL)", style=lbl), minp("stock-vol", 200, step=10)], style={"flex": "1"}),
                            html.Div([html.Label("Conc Factor", style=lbl),
                                      dcc.Dropdown(id="conc-factor", options=[{"label": f"{x}x", "value": x} for x in [2,3,4,5,10]],
                                                   value=5, clearable=False)], style={"flex": "1"}),
                            html.Div([html.Label("Transfer Ratio", style=lbl),
                                      html.Div(id="xfer-label", style={**ibox, "textAlign": "center"})], style={"flex": "1"}),
                        ], style={"display": "flex", "gap": "10px", "marginBottom": "14px"}),
                        html.Div(id="validation-errors", style={"marginBottom": "10px"}),
                        html.Button("Calculate & Generate Layout", id="calc-btn", n_clicks=0,
                                    style={**btn, "width": "100%", "padding": "12px"}),
                    ], style={"padding": "16px"}),
                ], style={**card, "flex": "1"}),
            ], style={"display": "flex", "gap": "20px", "marginBottom": "24px", "alignItems": "flex-start"}),

            # Concentration Series Table
            html.Div([html.Div("Concentration Series", style=chdr),
                dag.AgGrid(id="conc-grid", columnDefs=[
                    {"field": "Point", "width": 70, "cellStyle": {"fontFamily": MF, "fontWeight": "700"}},
                    {"field": "Conc (nM)", "width": 130, "valueFormatter": FMT_AUTO, "cellStyle": {"fontFamily": MF, "fontWeight": "600", "color": "#3b82f6"}},
                    {"field": "Log(nM)", "width": 100, "valueFormatter": FMT_DEC2, "cellStyle": {"fontFamily": MF}},
                    {"field": "Working Vol (µL)", "width": 140, "valueFormatter": FMT_AUTO, "cellStyle": {"fontFamily": MF}},
                ], rowData=[], defaultColDef={"sortable": True, "resizable": True},
                    dashGridOptions={"domLayout": "autoHeight"}, style={"width": "100%"}, className="ag-theme-alpine"),
            ], style={**card, "marginBottom": "24px"}),

            # Plate Viz
            html.Div([
                html.Div([html.Span("384-Well Layout (2×2 Quad)", style={"fontWeight": "600", "fontSize": "13px",
                    "letterSpacing": ".5px", "color": C["mt"]}),
                    html.Span(style={"flex": "1"}),
                    html.Span(id="plate-sub", style={"fontSize": "11px", "color": C["mt"], "fontFamily": MF})],
                    style={**chdr, "justifyContent": "space-between"}),
                dcc.Tabs(id="plate-tabs", value="p1", children=[
                    dcc.Tab(label="Plate 1 (Ab 1–12)", value="p1", style={"fontSize": "12px", "fontWeight": "600"},
                            selected_style={"fontSize": "12px", "fontWeight": "600", "borderTop": f"3px solid {C['ac']}"}),
                    dcc.Tab(label="Plate 2 (Ab 13–24)", value="p2", style={"fontSize": "12px", "fontWeight": "600"},
                            selected_style={"fontSize": "12px", "fontWeight": "600", "borderTop": f"3px solid {C['ac']}"}),]),
                dcc.Graph(id="plate-fig", figure=go.Figure(), config={"displayModeBar": False}, style={"height": "560px"}),
                html.Div(id="plate-leg", style={"display": "flex", "gap": "10px", "flexWrap": "wrap",
                    "padding": "10px 20px", "borderTop": f"1px solid {C['bd']}", "fontSize": "11px"}),
            ], style={**card, "marginBottom": "24px"}),

            # Stock Prep Table
            html.Div([
                html.Div([html.Span("Stock Plate Prep (Serial Dilution)", style={"fontWeight": "600", "fontSize": "13px",
                    "letterSpacing": ".5px", "color": C["mt"]}),
                    html.Span(style={"flex": "1"}),
                    html.Button("Export .xlsx", id="exp-prep-btn", n_clicks=0, style=btno), dcc.Download(id="dl-prep")],
                    style={**chdr, "justifyContent": "space-between"}),
                html.Div([html.Label("Show:", style={"fontSize": "12px", "fontWeight": "600", "color": C["mt"], "marginRight": "8px"}),
                    dcc.Dropdown(id="prep-filter", placeholder="Select...", clearable=False,
                                 style={"flex": "1", "fontSize": "13px"})],
                    style={"display": "flex", "alignItems": "center", "padding": "10px 20px",
                           "borderBottom": f"1px solid {C['bd']}"}),
                dag.AgGrid(id="prep-grid", columnDefs=[
                    {"field": "Well", "width": 85, "pinned": "left", "cellStyle": {"fontFamily": MF, "fontWeight": "700"}},
                    {"field": "Antibody", "width": 140},
                    {"field": "Block", "width": 60},
                    {"field": "Stock (nM)", "width": 115, "valueFormatter": FMT_AUTO, "cellStyle": {"fontFamily": MF, "fontWeight": "600", "color": "#10b981"}},
                    {"field": "Ab (µL)", "width": 90, "valueFormatter": FMT_DEC2, "cellStyle": {"fontFamily": MF, "fontWeight": "700", "color": "#3b82f6"}},
                    {"field": "Medium (µL)", "width": 100, "valueFormatter": FMT_DEC2, "cellStyle": {"fontFamily": MF, "fontWeight": "700", "color": "#f59e0b"}},
                    {"field": "Xfer (µL)", "width": 90, "valueFormatter": FMT_DEC2, "cellStyle": {"fontFamily": MF, "fontWeight": "600", "color": "#8b5cf6"}},
                    {"field": "Total (µL)", "width": 80, "valueFormatter": FMT_DEC2, "cellStyle": {"fontFamily": MF}},
                    {"field": "Note", "width": 250, "cellStyle": {"color": "#64748b", "fontSize": "11px"}},
                ], rowData=[], defaultColDef={"sortable": True, "filter": True, "resizable": True},
                    dashGridOptions={"animateRows": True, "domLayout": "autoHeight"},
                    style={"width": "100%"}, className="ag-theme-alpine"),
            ], style={**card, "marginBottom": "24px"}),
            html.Div(id="protocol-note", style={"marginBottom": "24px"}),
        ], style={"padding": "24px 0"})]),

        # ═══════ TAB 2: CELL CALCULATION ═══════
        dcc.Tab(label="🧬 Cell Calculation", value="tab-cell", style=ts, selected_style=tsel,
        children=[html.Div([
            dcc.Store(id="cc-all-store", data={}),
            html.Div([
                html.Div([
                    html.Span("Cancer Cell Seeding Calculator", style={"fontWeight": "600", "fontSize": "13px",
                        "letterSpacing": ".5px", "color": C["mt"]}),
                    html.Span(style={"flex": "1"}),
                    html.Div([
                        html.Label("Number of Cell Lines:", style={"fontSize": "12px", "fontWeight": "600",
                                   "color": C["mt"], "marginRight": "8px", "whiteSpace": "nowrap"}),
                        dcc.Dropdown(id="cc-num-lines", options=[{"label": str(x), "value": x} for x in range(1, 6)],
                                     value=1, clearable=False, style={"width": "70px", "fontSize": "13px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                ], style={**chdr, "justifyContent": "space-between"}),
            ], style={**card, "marginBottom": "16px"}),

            *[html.Div(id=f"cc-section-{i}", children=[
                html.Div([
                    html.Div([
                        html.Span(f"Cell Line {i+1}", style={"fontWeight": "600", "fontSize": "14px", "color": AB_COLORS[i]}),
                        html.Span(" — Cancer Cell Stock", style={"fontWeight": "600", "fontSize": "13px",
                            "letterSpacing": ".5px", "color": C["mt"]}),
                        html.Span(style={"flex": "1"}),
                        html.Span("ADC Cytotoxicity Assay", style={"fontSize": "11px", "color": C["mt"]}),
                    ], style={**chdr, "justifyContent": "space-between", "borderLeft": f"4px solid {AB_COLORS[i]}"}),
                    html.Div([html.Div([
                        html.Div([html.Label("Cell Line Name", style=lbl),
                                  dcc.Input(id=f"cc-name-{i}", type="text",
                                            placeholder=["e.g. SK-BR-3 (HER2+)", "e.g. Raji (CD20+)",
                                                         "e.g. NCI-N87 (HER2+)", "e.g. MDA-MB-231",
                                                         "e.g. A549 (EGFR+)"][i],
                                            style={**inp, "fontSize": "13px"})], style={"flex": "1.2"}),
                        html.Div([html.Label("Stock Density (Cells/mL)", style={**lbl, "color": C["ac"], "fontSize": "12px"}),
                                  minp(f"cc-stock-{i}", 1000000, step=100000)], style={"flex": "1"}),
                        html.Div([html.Label("Cells/Well", style={**lbl, "color": C["ac"], "fontSize": "12px"}),
                                  minp(f"cc-cpw-{i}", 5000, step=500)], style={"flex": "0.7"}),
                        html.Div([html.Label("µL/Well", style=lbl),
                                  minp(f"cc-wv-{i}", 40, step=5)], style={"flex": "0.5"}),
                        html.Div([html.Label("Total Wells", style=lbl),
                                  minp(f"cc-wells-{i}", 500, step=1)], style={"flex": "0.6"}),
                        html.Div([html.Label("Dead Vol (µL)", style=lbl),
                                  minp(f"cc-dead-{i}", 5000, step=1000)], style={"flex": "0.7"}),
                    ], style={"display": "flex", "gap": "12px"})], style={"padding": "14px 16px"}),
                ], style={**card, "marginBottom": "4px"}),
                html.Div(id=f"cc-val-{i}"),  # validation errors
                html.Div(id=f"cc-result-{i}"),
                html.Div(id=f"cc-summary-{i}", style={"marginBottom": "20px"}),
            ], style={"display": "none"} if i > 0 else {}) for i in range(5)],

            html.Div(id="cc-print-section", children=[
                html.Div([
                    html.Div([
                        html.Span("All Cell Lines — Preparation Summary", style={"fontWeight": "600", "fontSize": "13px",
                            "letterSpacing": ".5px", "color": C["mt"]}),
                        html.Span(style={"flex": "1"}),
                        html.Button("🖨 Print", id="cc-print-btn",
                                    style={**btn, "background": "#0f172a", "padding": "8px 16px", "fontSize": "12px"}),
                        html.Button("⬇ Export .xlsx", id="cc-export-btn", style={**btno, "marginLeft": "8px"}),
                        dcc.Download(id="dl-cellcalc"),
                    ], style={**chdr, "justifyContent": "space-between"}),
                    html.Div(id="cc-print-content", style={"padding": "16px"}),
                ], style=card),
            ], style={"display": "none"}),
        ], style={"padding": "24px 0"})]),

        # ═══════ TAB 3: DATA ANALYSIS ═══════
        dcc.Tab(label="📊 Data Analysis", value="tab-data", style=ts, selected_style=tsel,
        children=[html.Div([
            # Load Data card
            html.Div([
                html.Div("Load Data", style=chdr),
                html.Div([
                    html.Div([
                        dcc.Upload(id="data-upload", children=html.Div([
                            html.Div("Upload raw plate reader data or treatment results",
                                     style={"fontSize": "13px", "fontWeight": "500", "color": C["mt"]}),
                            html.Div("RLU data: Protein | Conc_nM | Rep_1..4  —or—  Viability data: Antibody | Conc_nM | Rep_1..4",
                                     style={"fontSize": "10px", "color": C["mt"], "marginTop": "4px", "fontFamily": MF}),
                        ], style=upz), accept=".xlsx,.xls,.csv"),
                        html.Div(id="data-status", style={"marginTop": "8px", "fontSize": "12px"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "20px", "padding": "16px", "alignItems": "stretch"}),
            ], style={**card, "marginBottom": "24px"}),

            # Paste from Plate Reader
            html.Div([
                html.Div([
                    html.Span("Paste from Plate Reader (EnVision / 384-Well Grid)", style={"fontWeight": "600", "fontSize": "13px",
                        "letterSpacing": ".5px", "color": C["mt"]}),
                    html.Span(style={"flex": "1"}),
                    html.Div([
                        html.Label("Plate:", style={"fontSize": "12px", "fontWeight": "600", "color": C["mt"], "marginRight": "6px"}),
                        dcc.Dropdown(id="paste-plate", options=[{"label": "Plate 1", "value": 1}, {"label": "Plate 2", "value": 2}],
                                     value=1, clearable=False, style={"width": "110px", "fontSize": "12px"}),
                        # NEW: Merge mode toggle
                        html.Label("Mode:", style={"fontSize": "12px", "fontWeight": "600", "color": C["mt"],
                                                    "marginLeft": "16px", "marginRight": "6px"}),
                        dcc.Dropdown(id="paste-mode", options=[
                            {"label": "Replace", "value": "replace"},
                            {"label": "Merge / Append", "value": "merge"},
                        ], value="replace", clearable=False, style={"width": "140px", "fontSize": "12px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                ], style={**chdr, "justifyContent": "space-between"}),
                html.Div([
                    html.Div([
                        html.Div("Copy the 384-well readout from EnVision and paste below (16 rows × 24 columns).",
                                 style={"fontSize": "12px", "color": C["mt"], "marginBottom": "8px"}),
                        html.Div("Accepts: tab-separated grid, with or without row/column headers. Empty wells are ignored.",
                                 style={"fontSize": "11px", "color": C["mt"], "marginBottom": "10px", "fontStyle": "italic"}),
                        dcc.Textarea(id="paste-grid", placeholder="Paste 384-well plate data here...\n\n"
                            "Example (tab-separated):\n"
                            "  1\t2\t3\t4\t...\t24\n"
                            "A\t45230\t44980\t38100\t37650\t...\n"
                            "B\t45100\t45300\t38400\t37900\t...\n"
                            "...",
                            style={"width": "100%", "height": "200px", "fontFamily": MF, "fontSize": "11px",
                                   "padding": "10px", "border": f"1px solid {C['bd']}", "borderRadius": "6px",
                                   "resize": "vertical", "background": C["sa"]}),
                    ], style={"flex": "3"}),
                    html.Div([
                        html.Button("📋 Parse & Analyze Plate", id="parse-plate-btn", n_clicks=0,
                                    style={**btn, "background": "#0d9488", "width": "100%", "padding": "14px",
                                           "fontSize": "13px", "marginBottom": "10px"}),
                        html.Div(id="paste-status", style={"fontSize": "12px"}),
                        html.Div([
                            html.Div("How it works:", style={"fontSize": "11px", "fontWeight": "600", "color": C["mt"], "marginBottom": "4px"}),
                            html.Div("1. Generate plate layout in Tab 1", style={"fontSize": "11px", "color": C["mt"]}),
                            html.Div("2. Run assay on plate reader", style={"fontSize": "11px", "color": C["mt"]}),
                            html.Div("3. Copy 16×24 grid from EnVision", style={"fontSize": "11px", "color": C["mt"]}),
                            html.Div("4. Paste here → auto-mapped to antibodies", style={"fontSize": "11px", "color": C["mt"]}),
                            html.Div("5. Use 'Merge' mode for Plate 2 data", style={"fontSize": "11px", "color": C["mt"], "fontWeight": "600"}),
                        ], style={"marginTop": "12px", "padding": "10px", "background": C["wb"],
                                  "borderRadius": "6px", "border": f"1px solid {C['wbd']}"}),
                    ], style={"flex": "1", "display": "flex", "flexDirection": "column"}),
                ], style={"display": "flex", "gap": "16px", "padding": "16px", "alignItems": "flex-start"}),
            ], style={**card, "marginBottom": "24px"}),

            # Reference standard selector (NEW)
            html.Div([
                html.Div([
                    html.Span("Reference Standard & Relative Potency", style={"fontWeight": "600", "fontSize": "13px",
                        "letterSpacing": ".5px", "color": C["mt"]}),
                    html.Span(style={"flex": "1"}),
                    html.Div([
                        html.Label("Reference Ab:", style={"fontSize": "12px", "fontWeight": "600",
                                                            "color": C["mt"], "marginRight": "8px"}),
                        dcc.Dropdown(id="ref-ab-select", placeholder="None (no normalization)",
                                     clearable=True, style={"width": "220px", "fontSize": "12px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                ], style={**chdr, "justifyContent": "space-between"}),
                html.Div(id="ref-info", style={"padding": "10px 20px", "fontSize": "12px", "color": C["mt"]}),
            ], style={**card, "marginBottom": "24px"}),

            # QC Panel (NEW)
            html.Div([
                html.Div([html.Span("🔍 Quality Control Flags", style={"fontWeight": "600", "fontSize": "13px",
                    "letterSpacing": ".5px", "color": C["mt"]})], style=chdr),
                html.Div(id="qc-panel", style={"padding": "12px 20px"}),
            ], style={**card, "marginBottom": "24px"}),

            # Download All
            html.Div([
                html.Button("⬇ Download All Analysis (.xlsx)", id="exp-all-btn", n_clicks=0,
                            style={**btn, "background": "#0f172a", "width": "100%", "padding": "14px",
                                   "fontSize": "14px", "letterSpacing": ".5px"}),
                dcc.Download(id="dl-all"),
            ], style={"marginBottom": "24px"}),

            # Interactive Overlay Dose-Response (NEW)
            html.Div([html.Div([html.Span("🎯 Interactive Overlay — All Antibodies", style={"fontWeight": "600", "fontSize": "13px",
                "letterSpacing": ".5px", "color": C["mt"]}), html.Span(style={"flex": "1"}),
                html.Span("Click legend to toggle • Double-click to isolate",
                          style={"fontSize": "11px", "color": C["mt"], "fontStyle": "italic"})],
                style={**chdr, "justifyContent": "space-between"}),
                dcc.Graph(id="overlay-fig", figure=go.Figure(),
                          config={"displayModeBar": True, "toImageButtonOptions": {
                              "format": "svg", "filename": "dose_response_overlay", "height": 650, "width": 1200, "scale": 2}}),
            ], style={**card, "marginBottom": "24px"}),

            # Dose-response individual subplots
            html.Div([html.Div([html.Span("📈 Dose-Response (4PL)", style={"fontWeight": "600", "fontSize": "13px",
                "letterSpacing": ".5px", "color": C["mt"]}), html.Span(style={"flex": "1"}),
                html.Button("Export EC50", id="exp-ec50-btn", n_clicks=0, style=btno), dcc.Download(id="dl-ec50")],
                style={**chdr, "justifyContent": "space-between"}),
                dcc.Graph(id="dose-fig", figure=go.Figure(), config={"displayModeBar": True}),
            ], style={**card, "marginBottom": "24px"}),

            # EC50 Summary (updated columns)
            html.Div([html.Div("EC50 Summary", style=chdr),
                dag.AgGrid(id="ec50-grid", columnDefs=[
                    {"field": "Protein", "width": 140, "cellStyle": {"fontWeight": "600"}},
                    {"field": "EC50 (nM)", "width": 110, "valueFormatter": FMT_AUTO,
                     "cellStyle": {"fontFamily": MF, "fontWeight": "700", "color": "#3b82f6"}},
                    {"field": "EC50 95% CI", "width": 155, "cellStyle": {"fontFamily": MF, "fontSize": "11px", "color": "#64748b"}},
                    {"field": "Top (%)", "width": 80, "valueFormatter": FMT_DEC1, "cellStyle": {"fontFamily": MF}},
                    {"field": "Bottom (%)", "width": 90, "valueFormatter": FMT_DEC1, "cellStyle": {"fontFamily": MF}},
                    {"field": "Hill Slope", "width": 90, "valueFormatter": FMT_DEC3, "cellStyle": {"fontFamily": MF}},
                    {"field": "R²", "width": 70, "valueFormatter": FMT_DEC4, "cellStyle": {"fontFamily": MF}},
                    {"field": "Adj R²", "width": 75, "valueFormatter": FMT_DEC4, "cellStyle": {"fontFamily": MF}},
                    {"field": "Max Kill (%)", "width": 95, "valueFormatter": FMT_DEC1,
                     "cellStyle": {"fontFamily": MF, "fontWeight": "600", "color": "#ef4444"}},
                    {"field": "Avg CV%", "width": 80, "valueFormatter": FMT_DEC1, "cellStyle": {"fontFamily": MF}},
                    {"field": "Fit Status", "width": 240},
                    {"field": "Rel Potency (%)", "width": 120, "cellStyle": {"fontFamily": MF, "fontWeight": "600", "color": "#8b5cf6"}},
                ], rowData=[], defaultColDef={"sortable": True, "resizable": True},
                    dashGridOptions={"domLayout": "autoHeight"}, style={"width": "100%"}, className="ag-theme-alpine"),
            ], style={**card, "marginBottom": "24px"}),

            # Mean/SD
            html.Div([html.Div([html.Span("Mean & SD (Raw)", style={"fontWeight": "600", "fontSize": "13px",
                "letterSpacing": ".5px", "color": C["mt"]}), html.Span(style={"flex": "1"}),
                html.Button("Export", id="exp-mean-btn", n_clicks=0, style=btno), dcc.Download(id="dl-mean")],
                style={**chdr, "justifyContent": "space-between"}),
                dag.AgGrid(id="mean-grid", columnDefs=[], rowData=[], defaultColDef={"sortable": True, "resizable": True},
                    dashGridOptions={"domLayout": "autoHeight"}, style={"width": "100%"}, className="ag-theme-alpine"),
            ], style={**card, "marginBottom": "24px"}),

            # Normalized
            html.Div([html.Div([html.Span("Normalized % & CV (GraphPad Ready)", style={"fontWeight": "600", "fontSize": "13px",
                "letterSpacing": ".5px", "color": C["mt"]}), html.Span(style={"flex": "1"}),
                html.Button("Export", id="exp-norm-btn", n_clicks=0, style=btno), dcc.Download(id="dl-norm")],
                style={**chdr, "justifyContent": "space-between"}),
                dag.AgGrid(id="norm-grid", columnDefs=[], rowData=[], defaultColDef={"sortable": True, "resizable": True},
                    dashGridOptions={"domLayout": "autoHeight"}, style={"width": "100%"}, className="ag-theme-alpine"),
            ], style={**card, "marginBottom": "24px"}),

            # Raw reps
            html.Div([html.Div([html.Span("Raw Replicate Data", style={"fontWeight": "600", "fontSize": "13px",
                "letterSpacing": ".5px", "color": C["mt"]}), html.Span(style={"flex": "1"}),
                html.Button("Export", id="exp-raw-btn", n_clicks=0, style=btno), dcc.Download(id="dl-raw")],
                style={**chdr, "justifyContent": "space-between"}),
                html.Div([html.Label("Filter:", style={"fontSize": "12px", "fontWeight": "600", "color": C["mt"], "marginRight": "8px"}),
                    dcc.Dropdown(id="raw-filter", placeholder="All", clearable=True, style={"flex": "1", "fontSize": "13px"})],
                    style={"display": "flex", "alignItems": "center", "padding": "10px 20px",
                           "borderBottom": f"1px solid {C['bd']}"}),
                dag.AgGrid(id="raw-grid", columnDefs=[
                    {"field": "Antibody", "width": 155, "cellStyle": {"fontWeight": "500"}},
                    {"field": "Conc (nM)", "width": 100, "valueFormatter": FMT_AUTO, "cellStyle": {"fontFamily": MF}},
                    *[{"field": f"Rep {i+1}", "width": 78, "valueFormatter": FMT_DEC2, "cellStyle": {"fontFamily": MF}} for i in range(4)],
                    {"field": "Mean", "width": 78, "valueFormatter": FMT_DEC2, "cellStyle": {"fontFamily": MF, "fontWeight": "700", "color": "#3b82f6"}},
                    {"field": "SD", "width": 72, "valueFormatter": FMT_DEC2, "cellStyle": {"fontFamily": MF}},
                    {"field": "CV%", "width": 72, "valueFormatter": FMT_DEC1, "cellStyle": {"fontFamily": MF, "fontWeight": "600"}},
                ], rowData=[], defaultColDef={"sortable": True, "filter": True, "resizable": True},
                    dashGridOptions={"animateRows": True}, style={"height": "420px", "width": "100%"}, className="ag-theme-alpine"),
            ], style={**card, "marginBottom": "32px"}),
        ], style={"padding": "24px 0"})]),

    ])], style={"maxWidth": "1500px", "margin": "0 auto", "padding": "0 24px 24px"}),
], style={"fontFamily": FF, "background": C["bg"], "minHeight": "100vh", "color": C["tx"]})


# ═══════════════════════════════════════════════════════════════════
# §12  CALLBACKS
# ═══════════════════════════════════════════════════════════════════

# ── Transfer ratio label ──
@callback(Output("xfer-label", "children"), Input("conc-factor", "value"))
def xfer_lbl(cf):
    return f"1 vol + {(cf or 5) - 1} vol medium"


# ── Upload Antibodies ──
@callback(Output("ab-store", "data"), Output("upload-status", "children"),
          Output("ab-summary", "children"), Output("ref-ab-select", "options"),
          Input("reagent-upload", "contents"), State("reagent-upload", "filename"),
          prevent_initial_call=True)
def load_ab(contents, fn):
    if not contents:
        return no_update, no_update, no_update, no_update
    _, cs = contents.split(",")
    dec = base64.b64decode(cs)
    try:
        df = (pd.read_csv(io.StringIO(dec.decode("utf-8"))) if fn.endswith(".csv")
              else pd.read_excel(io.BytesIO(dec)))

        def fc(cols, cands):
            for ca in cands:
                for dc in cols:
                    if ca.lower().replace(" ", "") in dc.lower().replace(" ", ""):
                        return dc
            return None

        cid = fc(df.columns, ["AB_ID", "antibody", "name", "sample"])
        cmw = fc(df.columns, ["MW", "molecular_weight", "kDa"])
        cco = fc(df.columns, ["Concentration", "conc", "mg/mL", "stock"])
        cct = fc(df.columns, ["Control", "control_type", "type", "ctrl"])

        if not cid or not cco:
            return (no_update,
                    html.Span(f"Missing required columns (need AB_ID + Concentration). Found: {list(df.columns)}",
                              style={"color": C["dg"]}),
                    no_update, no_update)

        ab_list, seen = [], {}
        for i, row in df.iterrows():
            if i >= 24: break
            nm = str(row[cid]).strip()
            if nm in seen:
                seen[nm] += 1; nm = f"{nm}_{seen[nm]}"
            else:
                seen[nm] = 1
            co = float(row[cco])
            mw = float(row[cmw]) if cmw and pd.notna(row.get(cmw)) else 150
            ct = str(row[cct]).strip() if cct and pd.notna(row.get(cct)) else "Positive"
            if ct not in ("Positive", "Negative"):
                ct = "Positive"
            ab_list.append({
                "name": nm, "conc_mg": co, "mw": mw,
                "color": AB_COLORS[i % len(AB_COLORS)],
                "col_start": (i % ABS_PER_PLATE) * COLS_PER_AB + 1,
                "plate": 1 + i // ABS_PER_PLATE, "idx": i, "ctrl_type": ct
            })

        n = len(ab_list)
        np_ = sum(1 for a in ab_list if a["ctrl_type"] == "Positive")
        st = html.Span([
            html.Strong(f"{n} antibodies", style={"color": C["ac"]}),
            html.Span(f" ({np_} pos, {n - np_} neg) from {fn}")
        ])
        rows = [html.Tr([
            html.Td(f"P{a['plate']}", style={"padding": "2px 6px", "fontFamily": MF, "fontSize": "10px"}),
            html.Td(html.Span("", style={"width": "8px", "height": "8px", "borderRadius": "50%",
                "background": a["color"], "display": "inline-block"}), style={"padding": "2px 4px"}),
            html.Td(a["name"], style={"padding": "2px 6px", "fontSize": "11px", "fontWeight": "500"}),
            html.Td(f"{a['conc_mg']} mg/mL", style={"padding": "2px 6px", "fontFamily": MF, "fontSize": "10px"}),
            html.Td(a["ctrl_type"], style={"padding": "2px 6px", "fontSize": "10px",
                "color": C["dg"] if a["ctrl_type"] == "Negative" else C["ok"]}),
        ], style={"background": C["wb"] if a["ctrl_type"] == "Negative" else "transparent"})
            for a in ab_list]
        tbl = html.Table([html.Tbody(rows)],
                         style={"width": "100%", "borderCollapse": "collapse", "background": C["sa"]})
        ref_opts = [{"label": a["name"], "value": a["name"]} for a in ab_list]
        return ab_list, st, tbl, ref_opts
    except Exception as e:
        return no_update, html.Span(f"Error: {e}", style={"color": C["dg"]}), no_update, no_update


# ── Calc plate layout + stock prep + concentration series with VALIDATION ──
@callback(
    Output("plate-sub", "children"), Output("prep-filter", "options"), Output("prep-filter", "value"),
    Output("protocol-note", "children"), Output("prep-store", "data"), Output("grid-store", "data"),
    Output("conc-grid", "rowData"), Output("conc-store", "data"), Output("validation-errors", "children"),
    Input("calc-btn", "n_clicks"), State("ab-store", "data"),
    State("start-conc", "value"), State("dil-factor", "value"), State("n-dil", "value"),
    State("block-vol", "value"), State("stock-vol", "value"), State("conc-factor", "value"),
    prevent_initial_call=True)
def calc_prep(n, ab_list, start, df_, npts, bv, sv, cf):
    _n = no_update
    if not ab_list:
        return _n, _n, _n, _n, _n, _n, _n, _n, render_validation_errors([("ab", "Load antibodies first")])

    # Validate inputs
    errors = validate_dilution_params(start, df_, npts, bv, sv, cf)
    if errors:
        return _n, _n, _n, _n, _n, _n, _n, _n, render_validation_errors(errors)

    start = max(start or 100, 0.01)
    df_ = max(df_ or 2, 2)
    npts = npts or 7
    bv = max(bv or 500, 10)
    sv = max(sv or 200, 10)
    cf = max(cf or 5, 2)

    twv = round(bv / (1 - 1 / df_), 2)
    concs = [start / (df_ ** i) for i in range(npts)] + [0]
    conc_rows = [{"Point": i + 1, "Conc (nM)": round(c, 6),
                  "Log(nM)": round(math.log10(c), 5) if c > 0 else -3,
                  "Working Vol (µL)": twv} for i, c in enumerate(concs)]

    tv = round(sv / (df_ - 1), 2)
    pva = sv + tv
    prep, gd = [], []
    for ab in ab_list:
        mw, sc = ab["mw"], ab["conc_mg"]
        for ci, wn in enumerate(concs):
            is_bg = (wn <= 0)
            sn = wn * cf if not is_bg else 0
            sm = nM_mg(sn, mw) if not is_bg else 0
            bl = f"{ROW_LABELS[ci * 2]}-{ROW_LABELS[ci * 2 + 1]}" if ci * 2 + 1 < 16 else "BKG"
            cs_ = ab["col_start"]
            well = f"{ROW_LABELS[min(ci * 2, 15)]}{cs_}"
            if is_bg:
                av, mv, xv, note = 0, sv, 0, "Medium only"
            elif ci == 0:
                av = (sm * pva) / sc if sc > 0 else 0
                mv = pva - av
                xv = tv
                note = f"From stock; xfer {tv}µL → next"
                if av > pva:
                    note = f"⚠ EXCEEDS {av:.1f}µL"
                    av, mv = pva, 0
            else:
                av, mv = 0, sv
                prev = ROW_LABELS[(ci - 1) * 2]
                xv = tv if ci < len(concs) - 2 else 0
                note = (f"Add {tv}µL from {prev}-block" +
                        (", xfer out" if xv else f", discard {tv}µL"))

            prep.append({
                "Well": well, "Antibody": ab["name"], "Block": bl, "Plate": ab["plate"],
                "Stock (nM)": round(sn, 4) if not is_bg else 0,
                "Ab (µL)": round(av, 2), "Medium (µL)": round(max(mv, 0), 2),
                "Xfer (µL)": round(xv, 2), "Total (µL)": round(pva, 2) if ci == 0 else sv,
                "Note": note
            })
            for dr in range(2):
                for dc in range(2):
                    ri = min(ci * 2 + dr, 15)
                    gd.append({"row": ri, "col": cs_ + dc, "work_nM": wn,
                               "ab_name": ab["name"], "color": ab["color"],
                               "is_bg": is_bg, "plate": ab["plate"],
                               "ctrl": ab.get("ctrl_type", "Positive")})

    ab_names = list(dict.fromkeys(r["Antibody"] for r in prep))
    n_concs = len(concs)
    blk_opts = [{"label": f"Block {ROW_LABELS[i*2]}-{ROW_LABELS[i*2+1]} — all Ab", "value": f"BLK:{i}"}
                for i in range(min(n_concs, 8))]
    ab_opts = [{"label": nm, "value": f"AB:{nm}"} for nm in ab_names]
    dd = ([{"label": "━ By Block", "value": "", "disabled": True}] + blk_opts +
          [{"label": "━ By Antibody", "value": "", "disabled": True}] + ab_opts)
    sub = f"12 Ab/plate × 2 plates | 2×2 quad | {df_}x dilution"
    prot = html.Div([
        html.Strong("Protocol:"), html.Br(),
        html.Span(f"1. Row A-B block: Ab + medium → {pva}µL. Stock = {cf}x."), html.Br(),
        html.Span(f"2. Serial dilute: xfer {tv}µL between blocks through to last dilution."), html.Br(),
        html.Span(f"3. Stamp to assay plate: {sv // cf}µL stock + {sv // cf * (cf - 1)}µL medium (1:{cf})."),
    ], style={"background": C["wb"], "border": f"1px solid {C['wbd']}", "borderRadius": "8px",
              "padding": "14px 20px", "fontSize": "13px", "lineHeight": "1.8", "color": C["wt"]})
    return sub, dd, "BLK:0", prot, prep, gd, conc_rows, concs, None


# ── Plate viz ──
@callback(Output("plate-fig", "figure"), Output("plate-leg", "children"),
          Input("plate-tabs", "value"), Input("grid-store", "data"), State("ab-store", "data"))
def render_plate(tab, gd, ab_list):
    if not gd:
        return go.Figure(), []
    pn = 1 if tab == "p1" else 2
    wells = [w for w in gd if w["plate"] == pn]
    pabs = [a for a in (ab_list or []) if a["plate"] == pn]
    filled = {(w["row"], w["col"]) for w in wells}
    fig = go.Figure()
    for r in range(16):
        for c in range(1, 25):
            if (r, c) not in filled:
                fig.add_trace(go.Scatter(x=[c], y=[15 - r], mode="markers",
                    marker=dict(size=18, color="#f1f5f9", line=dict(color="#e2e8f0", width=1)),
                    hovertext=f"<b>{ROW_LABELS[r]}{c}</b><br>Empty", hoverinfo="text", showlegend=False))
    drawn = set()
    for w in sorted(wells, key=lambda x: (x["row"], x["col"])):
        r, c, bg = w["row"], w["col"], w["is_bg"]
        base = w["color"]
        a = 0.12 if bg else max(0.15, 1.0 - r / 16)
        wc = hex_rgba(base, a)
        bc = hex_rgba(base, 0.3 if bg else min(1, a + 0.2))
        txt = "Bkg" if bg else fmt_nM(w["work_nM"])
        tc = C["mt"] if bg else ("white" if a > 0.45 else C["tx"])
        show = w["ab_name"] not in drawn
        drawn.add(w["ab_name"])
        fig.add_trace(go.Scatter(x=[c], y=[15 - r], mode="markers+text",
            marker=dict(size=18, color=wc, line=dict(color=bc, width=1)),
            text=[txt], textposition="middle center", textfont=dict(size=5, family=MF, color=tc),
            hovertext=(f"<b>{ROW_LABELS[r]}{c}</b><br>{w['ab_name']}<br>{w['work_nM']:.4g} nM"
                       if not bg else f"<b>{ROW_LABELS[r]}{c}</b><br>Bkg"),
            hoverinfo="text", hoverlabel=dict(bgcolor="#0f172a", font=dict(size=12, color="white")),
            name=w["ab_name"], showlegend=show, legendgroup=w["ab_name"]))
    fig.update_layout(
        xaxis=dict(range=[0.2, 24.8], tickvals=list(range(1, 25)), ticktext=[str(i) for i in range(1, 25)],
                   side="top", fixedrange=True, tickfont=dict(family=MF, size=9, color="#64748b"),
                   showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.8, 15.8], tickvals=list(range(16)), ticktext=list(reversed(ROW_LABELS)),
                   fixedrange=True, tickfont=dict(family=MF, size=9, color="#64748b"),
                   showgrid=False, zeroline=False),
        margin=dict(l=40, r=20, t=40, b=10), plot_bgcolor="white", paper_bgcolor="white",
        hovermode="closest", height=560,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, font=dict(size=9)),
        shapes=[dict(type="rect", x0=0.3, x1=24.7, y0=-0.7, y1=15.7,
                     line=dict(color="#e2e8f0", width=1))])
    for i in range(1, 8):
        fig.add_shape(type="line", x0=0.3, x1=24.7, y0=15 - i * 2 + 0.5, y1=15 - i * 2 + 0.5,
                      line=dict(color="#cbd5e1", width=0.5, dash="dot"))
    leg = [html.Div([
        html.Div(style={"width": "40px", "height": "12px", "borderRadius": "3px",
            "background": f"linear-gradient(90deg,{a['color']},{hex_rgba(a['color'], 0.15)})",
            "display": "inline-block"}),
        html.Span(f"C{a['col_start']}-{a['col_start']+1}: {a['name']}",
                  style={"fontSize": "10px", "fontWeight": "500"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "4px"}) for a in pabs]
    return fig, leg


# ── Prep filter ──
@callback(Output("prep-grid", "rowData"), Input("prep-filter", "value"),
          State("prep-store", "data"), prevent_initial_call=True)
def filt_prep(sel, data):
    if not data or not sel:
        return no_update
    if sel.startswith("BLK:"):
        idx = int(sel[4:])
        bl = f"{ROW_LABELS[idx * 2]}-{ROW_LABELS[idx * 2 + 1]}"
        return [r for r in data if r["Block"] == bl]
    if sel.startswith("AB:"):
        return [r for r in data if r["Antibody"] == sel[3:]]
    return no_update


# ── Cell calc: show/hide sections ──
@callback(
    *[Output(f"cc-section-{i}", "style") for i in range(5)],
    Output("cc-print-section", "style"),
    Input("cc-num-lines", "value"))
def toggle_cc_sections(n):
    n = n or 1
    styles = [{} if i < n else {"display": "none"} for i in range(5)]
    return *styles, ({} if n >= 2 else {"display": "none"})


# ── Cell calc: shared computation with VALIDATION ──
def _cc_compute(stock, cpw, wv, wells, dead):
    stock = stock or 1e6; cpw = cpw or 5000; wv = wv or 40; wells = wells or 500; dead = dead or 5000
    req = cpw / (wv / 1000)
    minv = wells * wv; prep = minv + dead
    add_s = req * prep / stock; add_m = prep - add_s
    tot = cpw * wells; dil = stock / req if req > 0 else 0
    return dict(req_conc=req, prep_vol=prep, add_stock=add_s, add_medium=add_m,
                total_cells=tot, dilution=dil, min_vol=minv,
                stock=stock, cpw=cpw, wv=wv, wells=wells, dead=dead)

def _cc_html(d):
    sS = {"fontSize": "11px", "fontWeight": "700", "color": C["ok"], "letterSpacing": "1px",
          "marginBottom": "4px", "textTransform": "uppercase"}
    sN = {"fontSize": "18px", "fontWeight": "600", "fontFamily": MF, "color": C["ac"]}
    sL = {"fontSize": "12px", "color": C["mt"], "marginBottom": "2px"}
    sF = {"fontSize": "11px", "color": C["mt"], "fontFamily": MF}
    sR = {"display": "flex", "gap": "16px", "alignItems": "flex-end", "padding": "12px 0"}
    sB = {"borderBottom": f"1px solid {C['bd']}"}
    warns = []
    if d["add_stock"] > d["prep_vol"]:
        warns.append(f"Stock density too low! Need {d['add_stock']:,.0f} µL stock.")
    if d["req_conc"] > d["stock"]:
        warns.append(f"Stock ({d['stock']:,.0f}/mL) < required ({d['req_conc']:,.0f}/mL).")
    r = html.Div([
        html.Div([
            html.Div([html.Div("Step 1 — Required Concentration", style=sS),
                      html.Div("Cell concentration in working suspension",
                               style={"fontSize": "13px", "color": C["tx"]})], style={"flex": "2"}),
            html.Div([html.Div("Required Conc", style=sL),
                      html.Div(f"{d['req_conc']:,.0f} Cells/mL", style=sN),
                      html.Div(f"= {d['cpw']:,} Cells ÷ {d['wv']} µL", style=sF)],
                     style={"flex": "1.5", "textAlign": "right"}),
        ], style={**sR, **sB}),
        html.Div([
            html.Div([html.Div("Step 2 — Total Working Volume", style=sS),
                      html.Div("Volume to prepare including dead volume",
                               style={"fontSize": "13px", "color": C["tx"]})], style={"flex": "2"}),
            html.Div([html.Div("Working Volume", style=sL),
                      html.Div(f"{d['prep_vol']:,.0f} µL", style=sN),
                      html.Div(f"= ({d['wells']:,} × {d['wv']}) + {d['dead']:,.0f} dead", style=sF)],
                     style={"flex": "1.5", "textAlign": "right"}),
        ], style={**sR, **sB}),
        html.Div([
            html.Div([html.Div("Step 3 — Mix Stock Cells + Medium", style=sS),
                      html.Div(f"Dilute {d['dilution']:,.1f}× to reach {d['req_conc']:,.0f} Cells/mL",
                               style={"fontSize": "13px", "color": C["tx"]})], style={"flex": "1.2"}),
            html.Div([html.Div([
                html.Div([html.Div("Add Stock", style=sL),
                          html.Div(f"{d['add_stock']:,.1f} µL", style={**sN, "color": "#3b82f6"})],
                         style={"textAlign": "center", "flex": "1"}),
                html.Div("+", style={"fontSize": "20px", "color": C["mt"], "padding": "0 4px", "alignSelf": "center"}),
                html.Div([html.Div("Add Medium", style=sL),
                          html.Div(f"{max(d['add_medium'], 0):,.1f} µL", style={**sN, "color": "#f59e0b"})],
                         style={"textAlign": "center", "flex": "1"}),
                html.Div("=", style={"fontSize": "20px", "color": C["mt"], "padding": "0 4px", "alignSelf": "center"}),
                html.Div([html.Div("Total", style=sL),
                          html.Div(f"{d['prep_vol']:,.0f} µL", style={**sN, "color": C["tx"]})],
                         style={"textAlign": "center", "flex": "1"}),
            ], style={"display": "flex", "alignItems": "flex-end"})], style={"flex": "2"}),
        ], style={**sR, **sB}),
        html.Div([
            html.Div([html.Div("Step 4 — Dispense", style=sS),
                      html.Div(f"Pipette {d['wv']} µL/well into {d['wells']:,} wells",
                               style={"fontSize": "13px", "color": C["tx"]})], style={"flex": "2"}),
            html.Div([html.Div("Total Cells Seeded", style=sL),
                      html.Div(f"{d['total_cells']:,.0f} Cells", style=sN)],
                     style={"flex": "1.5", "textAlign": "right"}),
        ], style=sR),
        *[html.Div(w, style=err_box) for w in warns],
    ])
    cr = {"flex": "1", "textAlign": "center", "padding": "10px", "borderRadius": "8px"}
    s = html.Div([html.Div([
        html.Div([html.Div("Stock", style={"fontSize": "10px", "color": C["mt"]}),
                  html.Div(f"{d['stock']:,.0f}/mL", style={"fontSize": "13px", "fontWeight": "600", "fontFamily": MF})],
                 style={**cr, "background": C["sa"]}),
        html.Div([html.Div("Required", style={"fontSize": "10px", "color": C["mt"]}),
                  html.Div(f"{d['req_conc']:,.0f}/mL", style={"fontSize": "13px", "fontWeight": "600", "fontFamily": MF})],
                 style={**cr, "background": C["sa"]}),
        html.Div([html.Div("Dilution", style={"fontSize": "10px", "color": C["mt"]}),
                  html.Div(f"{d['dilution']:,.1f}×", style={"fontSize": "13px", "fontWeight": "600", "fontFamily": MF})],
                 style={**cr, "background": C["sa"]}),
        html.Div([html.Div("Add Stock", style={"fontSize": "10px", "color": C["mt"]}),
                  html.Div(f"{d['add_stock']:,.1f} µL", style={"fontSize": "13px", "fontWeight": "600", "fontFamily": MF, "color": "#3b82f6"})],
                 style={**cr, "background": C["al"]}),
        html.Div([html.Div("Add Medium", style={"fontSize": "10px", "color": C["mt"]}),
                  html.Div(f"{max(d['add_medium'],0):,.1f} µL", style={"fontSize": "13px", "fontWeight": "600", "fontFamily": MF, "color": "#f59e0b"})],
                 style={**cr, "background": C["sa"]}),
    ], style={"display": "flex", "gap": "8px"})], style={**card, "padding": "12px"})
    return r, s


def _mk_cc(idx):
    @callback(
        Output(f"cc-result-{idx}", "children"), Output(f"cc-summary-{idx}", "children"),
        Output(f"cc-val-{idx}", "children"),
        Output("cc-all-store", "data", allow_duplicate=True),
        Input(f"cc-stock-{idx}", "value"), Input(f"cc-cpw-{idx}", "value"),
        Input(f"cc-wv-{idx}", "value"), Input(f"cc-wells-{idx}", "value"),
        Input(f"cc-dead-{idx}", "value"),
        State(f"cc-name-{idx}", "value"), State("cc-all-store", "data"),
        prevent_initial_call=True)
    def _calc(stock, cpw, wv, wells, dead, name, all_d):
        # Validate
        errors = validate_cell_params(stock, cpw, wv, wells, dead)
        if errors:
            return None, None, render_validation_errors(errors), all_d or {}
        d = _cc_compute(stock, cpw, wv, wells, dead)
        r, s = _cc_html(d)
        all_d = all_d or {}
        all_d[str(idx)] = {**d, "name": name or f"Cell Line {idx+1}"}
        return r, s, None, all_d

for _i in range(5):
    _mk_cc(_i)


# ── Print Summary ──
@callback(Output("cc-print-content", "children"), Input("cc-all-store", "data"),
          State("cc-num-lines", "value"))
def cc_print(all_d, n):
    if not all_d: return no_update
    n = n or 1
    hs = {"padding": "8px 12px", "background": C["sa"], "borderBottom": f"2px solid {C['bd']}",
          "fontSize": "12px", "fontWeight": "600", "color": C["mt"], "textAlign": "left"}
    header = html.Tr([html.Th(h, style=hs) for h in
        ["Cell Line", "Stock (Cells/mL)", "Cells/Well", "µL/Well", "Wells",
         "Req'd Conc (Cells/mL)", "Dilution", "Add Stock (µL)", "Add Medium (µL)",
         "Total Vol (µL)", "Total Cells"]])
    rows = []
    cs = {"padding": "8px 12px", "fontSize": "12px", "fontFamily": MF,
          "borderBottom": f"1px solid {C['bd']}"}
    for k in sorted(all_d.keys()):
        i = int(k)
        if i >= n: continue
        d = all_d[k]; nm = d.get("name", f"Cell Line {i+1}")
        rows.append(html.Tr([
            html.Td(nm, style={**cs, "fontWeight": "600", "fontFamily": FF,
                                "borderLeft": f"4px solid {AB_COLORS[i]}"}),
            html.Td(f"{d['stock']:,.0f}", style=cs),
            html.Td(f"{d['cpw']:,}", style=cs),
            html.Td(f"{d['wv']}", style=cs),
            html.Td(f"{d['wells']:,}", style=cs),
            html.Td(f"{d['req_conc']:,.0f}", style={**cs, "fontWeight": "600"}),
            html.Td(f"{d['dilution']:,.1f}×", style=cs),
            html.Td(f"{d['add_stock']:,.1f}", style={**cs, "fontWeight": "700", "color": "#3b82f6"}),
            html.Td(f"{max(d['add_medium'],0):,.1f}", style={**cs, "color": "#f59e0b"}),
            html.Td(f"{d['prep_vol']:,.0f}", style=cs),
            html.Td(f"{d['total_cells']:,.0f}", style=cs),
        ]))
    return html.Div([
        html.Div([
            html.Div("Cancer Cell Preparation — All Cell Lines",
                     style={"fontSize": "16px", "fontWeight": "600", "marginBottom": "4px"}),
            html.Div("Print or export this summary for your lab notebook",
                     style={"fontSize": "12px", "color": C["mt"]})
        ], style={"marginBottom": "16px", "paddingBottom": "12px",
                  "borderBottom": f"2px solid {C['tx']}"}),
        html.Table([html.Thead(header), html.Tbody(rows)],
                   style={"width": "100%", "borderCollapse": "collapse", "marginBottom": "16px"}),
        html.Div("Units: Density in Cells/mL | Volumes in µL",
                 style={"fontSize": "11px", "color": C["mt"]}),
    ])


# ── Print button ──
app.clientside_callback(
    "function(){window.print();return window.dash_clientside.no_update;}",
    Output("cc-print-btn", "n_clicks"), Input("cc-print-btn", "n_clicks"),
    prevent_initial_call=True)


# ── Export Cell Calc xlsx ──
@callback(Output("dl-cellcalc", "data"), Input("cc-export-btn", "n_clicks"),
          State("cc-all-store", "data"), State("cc-num-lines", "value"),
          prevent_initial_call=True)
def cc_export(nc, all_d, n):
    if not all_d: return no_update
    n = n or 1; rows = []
    for k in sorted(all_d.keys()):
        i = int(k)
        if i >= n: continue
        d = all_d[k]; nm = d.get("name", f"Cell Line {i+1}")
        rows.append({
            "Cell Line": nm, "Stock (Cells/mL)": d["stock"], "Cells/Well": d["cpw"],
            "µL/Well": d["wv"], "Total Wells": d["wells"],
            "Req'd Conc (Cells/mL)": round(d["req_conc"]),
            "Dilution": round(d["dilution"], 1),
            "Add Stock (µL)": round(d["add_stock"], 1),
            "Add Medium (µL)": round(max(d["add_medium"], 0), 1),
            "Total Vol (µL)": round(d["prep_vol"]),
            "Dead Vol (µL)": d["dead"], "Total Cells": d["total_cells"]
        })
    return dcc.send_data_frame(pd.DataFrame(rows).to_excel,
        f"cell_prep_{datetime.now():%Y%m%d_%H%M%S}.xlsx", index=False)


# ═══════════════════════════════════════════════════════════════════
# §13  CONSOLIDATED DATA ANALYSIS CALLBACK
# ═══════════════════════════════════════════════════════════════════
#
# Instead of 3+ separate callbacks writing to the same outputs, we use
# ONE unified callback dispatched by ctx.triggered_id.

def _parse_plate_grid(text):
    """Parse pasted 16×24 grid text → 2D array of floats."""
    lines = [l for l in text.strip().split("\n") if l.strip()]
    grid = []
    for line in lines:
        cells = line.replace(",", "").split("\t") if "\t" in line else line.split()
        nums = []
        for c in cells:
            c = c.strip()
            if not c: continue
            if len(c) == 1 and c.upper() in "ABCDEFGHIJKLMNOP": continue
            try:
                nums.append(float(c))
            except ValueError:
                continue
        if nums:
            grid.append(nums)
    return grid


@callback(
    Output("data-status", "children"),
    Output("mean-grid", "columnDefs"), Output("mean-grid", "rowData"),
    Output("norm-grid", "columnDefs"), Output("norm-grid", "rowData"),
    Output("dose-fig", "figure"), Output("overlay-fig", "figure"),
    Output("ec50-grid", "rowData"),
    Output("analysis-store", "data"),
    Output("raw-grid", "rowData"), Output("raw-filter", "options"),
    Output("sim-store", "data"),
    Output("paste-status", "children"),
    Output("qc-panel", "children"),
    Output("ref-ab-select", "options", allow_duplicate=True),
    Output("merged-plates", "data"),
    # Inputs — all triggers
    Input("data-upload", "contents"),
    Input("parse-plate-btn", "n_clicks"),
    # States
    State("conc-store", "data"), State("ab-store", "data"),
    State("data-upload", "filename"),
    State("paste-grid", "value"), State("paste-plate", "value"),
    State("paste-mode", "value"),
    State("grid-store", "data"),
    State("ref-ab-select", "value"),
    State("merged-plates", "data"),
    prevent_initial_call=True)
def unified_analysis(
    upload_contents, parse_n,
    concs, ab_list, upload_fn,
    paste_text, paste_plate, paste_mode,
    grid_data, ref_ab, merged_data
):
    """Single dispatcher for all data analysis pathways."""
    trigger = ctx.triggered_id
    _n = no_update
    empty = tuple([_n] * 16)

    def err(msg, paste_msg=None):
        return (html.Span(msg, style={"color": C["dg"]}),
                _n, _n, _n, _n, _n, _n, _n, _n, _n, _n, _n,
                html.Span(paste_msg or msg, style={"color": C["dg"]}) if paste_msg or msg else _n,
                _n, _n, _n)

    # ─── UPLOAD DATA ───
    if trigger == "data-upload":
        if not upload_contents:
            return empty
        _, cs = upload_contents.split(",")
        dec = base64.b64decode(cs)
        try:
            df = (pd.read_csv(io.StringIO(dec.decode("utf-8"))) if upload_fn.endswith(".csv")
                  else pd.read_excel(io.BytesIO(dec)))

            def fc(cols, cands):
                for ca in cands:
                    for dc in cols:
                        if ca.lower().replace(" ", "") in dc.lower().replace(" ", ""):
                            return dc
                return None

            cp = fc(df.columns, ["Protein", "Antibody", "AB_ID", "name", "sample"])
            cc = fc(df.columns, ["Concentration", "conc", "Conc_nM", "nM"])
            rcs = sorted([c for c in df.columns if "rep" in c.lower() or c.startswith("Rep")])
            cct = fc(df.columns, ["Control", "control_type", "type", "ctrl"])

            if not cp or not cc or len(rcs) < 2:
                return err(f"Missing columns (need ID + Conc + ≥2 Reps). Found: {list(df.columns)}")

            sample_vals = df[rcs[0]].dropna().head(20)
            is_rlu = sample_vals.mean() > 500
            names = list(dict.fromkeys(df[cp].astype(str)))
            concs_up = sorted(df[cc].unique(), reverse=True)

            if is_rlu:
                raw = {}
                for pi, nm in enumerate(names):
                    for ci, c in enumerate(concs_up):
                        mask = (df[cp].astype(str) == nm) & (df[cc] == c)
                        sub = df.loc[mask, rcs]
                        if not sub.empty:
                            raw[(pi, ci)] = [float(v) for v in sub.iloc[0].values if pd.notna(v)]
                mc, mr, nc, nr, fig, ec, flags = process_rlu_data(raw, list(concs_up), names[:MAX_PROTEINS], ref_ab)
                st = html.Span(["✅ ", html.Strong(f"{len(names)} proteins (RLU)", style={"color": C["ok"]}),
                                 f" from {upload_fn}"])
                qc = render_qc_panel([f.to_dict() if hasattr(f, 'to_dict') else f for f in flags])
                ref_opts = [{"label": n, "value": n} for n in names]
                pseudo_ab = [{"name": e["Protein"], "color": e["_color"], "ctrl_type": e.get("Control", "Positive")} for e in ec]
                ofig = make_overlay_fig(pseudo_ab, ec)
                return st, mc, mr, nc, nr, fig, ofig, ec, ec, [], [], [], _n, qc, ref_opts, {}
            else:
                sim_data = []
                for _, row in df.iterrows():
                    nm = str(row[cp]); c = float(row[cc])
                    ct = str(row[cct]).strip() if cct and pd.notna(row.get(cct)) else "Positive"
                    for ri, rc in enumerate(rcs):
                        if pd.notna(row[rc]):
                            sim_data.append({"Antibody": nm, "Conc_nM": c, "Replicate": ri + 1,
                                             "Viability_%": float(row[rc]), "Control": ct})
                uab = [{"name": nm, "color": AB_COLORS[i % len(AB_COLORS)],
                         "ctrl_type": next((r["Control"] for r in sim_data if r["Antibody"] == nm), "Positive")}
                        for i, nm in enumerate(names)]
                raw_tbl = build_raw_table(sim_data)
                analysis, flags = run_4pl_analysis_robust(uab, sim_data, list(concs_up), ref_ab)
                fig = make_dose_fig(uab, analysis)
                fopts = [{"label": a["name"], "value": a["name"]} for a in uab]
                ec_tbl = [{k: v for k, v in a.items() if not k.startswith("_")} for a in analysis]
                st = html.Span(["✅ ", html.Strong(f"{len(uab)} antibodies (viability)", style={"color": C["ok"]}),
                                 f" from {upload_fn}"])
                qc = render_qc_panel([f.to_dict() for f in flags])
                ref_opts = [{"label": a["name"], "value": a["name"]} for a in uab]
                ofig = make_overlay_fig(uab, analysis)
                return st, [], [], [], [], fig, ofig, ec_tbl, analysis, raw_tbl, fopts, sim_data, _n, qc, ref_opts, {}
        except Exception as e:
            return err(f"Error: {e}")

    # ─── PARSE PLATE GRID (with multi-plate merge) ───
    if trigger == "parse-plate-btn":
        if not paste_text or not paste_text.strip():
            return err("Please paste plate reader data first.", "Please paste plate reader data first.")
        if not grid_data:
            return err("Generate plate layout in Tab 1 first.", "Generate plate layout in Tab 1 first.")

        plate_num = paste_plate or 1
        grid = _parse_plate_grid(paste_text)
        if len(grid) < 2:
            return err(f"Could not parse grid. Found {len(grid)} rows. Expected ≥2 rows.")

        n_rows = len(grid)
        n_cols = max(len(r) for r in grid) if grid else 0
        for r in grid:
            while len(r) < n_cols:
                r.append(0)

        well_map = {(w["row"], w["col"]): w for w in grid_data if w["plate"] == plate_num}
        if not well_map:
            return err(f"No layout data for Plate {plate_num}.")

        grouped = defaultdict(list)
        ab_info = {}
        matched = 0
        for r_idx in range(min(n_rows, 16)):
            for c_idx in range(min(n_cols, 24)):
                col_1based = c_idx + 1
                key = (r_idx, col_1based)
                if key in well_map:
                    w = well_map[key]
                    val = grid[r_idx][c_idx]
                    if val > 0:
                        grouped[(w["ab_name"], w["work_nM"])].append(val)
                        ab_info[w["ab_name"]] = {"color": w["color"], "ctrl": w.get("ctrl", "Positive")}
                        matched += 1

        if matched == 0:
            return err(f"No wells matched. Grid: {n_rows}×{n_cols}, layout wells: {len(well_map)}.")

        # ── Multi-plate merge ──
        merged = merged_data or {}
        if paste_mode == "merge" and merged:
            # Merge new data into existing
            for k, v in grouped.items():
                mk = f"{k[0]}|{k[1]}"  # serialize tuple key
                if mk in merged:
                    merged[mk] = merged[mk] + v
                else:
                    merged[mk] = v
            # Also keep ab_info
            for k, v in ab_info.items():
                merged[f"__info__{k}"] = v
        else:
            # Replace
            merged = {}
            for k, v in grouped.items():
                merged[f"{k[0]}|{k[1]}"] = v
            for k, v in ab_info.items():
                merged[f"__info__{k}"] = v

        # Reconstruct grouped from merged store
        final_grouped = defaultdict(list)
        final_ab_info = {}
        for k, v in merged.items():
            if k.startswith("__info__"):
                final_ab_info[k[8:]] = v
            else:
                parts = k.rsplit("|", 1)
                ab_name = parts[0]
                conc = float(parts[1])
                final_grouped[(ab_name, conc)] = v

        ab_names = list(dict.fromkeys(
            w["ab_name"] for w in sorted(well_map.values(), key=lambda x: x["col"])))
        # Also include ab_names from merged data
        for k in final_grouped:
            if k[0] not in ab_names:
                ab_names.append(k[0])
        conc_list = sorted(set(k[1] for k in final_grouped), reverse=True)

        raw_rlu = {}
        for pi, ab in enumerate(ab_names):
            for ci, c in enumerate(conc_list):
                reps = final_grouped.get((ab, c), [])
                if reps:
                    raw_rlu[(pi, ci)] = reps

        if not raw_rlu:
            return err("Could not map pasted values to plate layout.")

        mc, mr, nc, nr, fig, ec, flags = process_rlu_data(raw_rlu, conc_list, ab_names, ref_ab)

        # Build raw table for raw grid
        sim_data = []
        for ab in ab_names:
            info = final_ab_info.get(ab, {"ctrl": "Positive"})
            ctrl = info.get("ctrl", "Positive") if isinstance(info, dict) else "Positive"
            for c in conc_list:
                reps = final_grouped.get((ab, c), [])
                for ri, v in enumerate(reps):
                    sim_data.append({"Antibody": ab, "Conc_nM": c, "Replicate": ri + 1,
                                     "Viability_%": v, "Control": ctrl})
        raw_tbl = build_raw_table(sim_data) if sim_data else []
        fopts = [{"label": a, "value": a} for a in ab_names]

        merge_label = " (merged)" if paste_mode == "merge" and merged_data else ""
        st = html.Span(["✅ ", html.Strong(f"Plate {plate_num}: {len(ab_names)} antibodies{merge_label}",
                                            style={"color": C["ok"]}),
                         f" × {len(conc_list)} concs | {matched} wells parsed"])
        ps = html.Span(["✅ ", html.Strong(f"{matched} wells mapped{merge_label}", style={"color": C["ok"]})])
        qc = render_qc_panel([f.to_dict() if hasattr(f, 'to_dict') else f for f in flags])
        ref_opts = [{"label": a, "value": a} for a in ab_names]
        pseudo_ab = [{"name": e["Protein"], "color": e["_color"], "ctrl_type": e.get("Control", "Positive")} for e in ec]
        ofig = make_overlay_fig(pseudo_ab, ec)
        return st, mc, mr, nc, nr, fig, ofig, ec, ec, raw_tbl, fopts, sim_data, ps, qc, ref_opts, merged

    return empty


# ── Reference info text ──
@callback(Output("ref-info", "children"), Input("ref-ab-select", "value"))
def ref_info(val):
    if not val:
        return html.Span("No reference selected — EC50 values shown as absolute. "
                         "Select a reference antibody to compute relative potency.",
                         style={"fontStyle": "italic"})
    return html.Span([
        html.Strong(val, style={"color": C["ac"]}),
        " set as reference standard. Other antibodies will show relative potency (%) "
        "where Rel Potency = Reference EC50 ÷ Test EC50 × 100."
    ])


# ── Raw filter ──
@callback(Output("raw-grid", "rowData", allow_duplicate=True),
          Input("raw-filter", "value"), State("sim-store", "data"),
          prevent_initial_call=True)
def filt_raw(sel, sim):
    if not sim: return no_update
    return build_raw_table(sim, sel if sel else None)


# ═══════════════════════════════════════════════════════════════════
# §14  EXPORTS
# ═══════════════════════════════════════════════════════════════════

@callback(Output("dl-prep", "data"), Input("exp-prep-btn", "n_clicks"),
          State("prep-store", "data"), prevent_initial_call=True)
def ep(n, d):
    return dcc.send_data_frame(pd.DataFrame(d).to_excel,
        f"prep_{datetime.now():%Y%m%d_%H%M%S}.xlsx", index=False) if d else no_update

@callback(Output("dl-mean", "data"), Input("exp-mean-btn", "n_clicks"),
          State("mean-grid", "rowData"), prevent_initial_call=True)
def em(n, d):
    return dcc.send_data_frame(pd.DataFrame(d).to_excel,
        f"mean_{datetime.now():%Y%m%d_%H%M%S}.xlsx", index=False) if d else no_update

@callback(Output("dl-norm", "data"), Input("exp-norm-btn", "n_clicks"),
          State("norm-grid", "rowData"), prevent_initial_call=True)
def en(n, d):
    return dcc.send_data_frame(pd.DataFrame(d).to_excel,
        f"norm_{datetime.now():%Y%m%d_%H%M%S}.xlsx", index=False) if d else no_update

@callback(Output("dl-ec50", "data"), Input("exp-ec50-btn", "n_clicks"),
          State("ec50-grid", "rowData"), prevent_initial_call=True)
def ee(n, d):
    return dcc.send_data_frame(pd.DataFrame(d).to_excel,
        f"ec50_{datetime.now():%Y%m%d_%H%M%S}.xlsx", index=False) if d else no_update

@callback(Output("dl-raw", "data"), Input("exp-raw-btn", "n_clicks"),
          State("sim-store", "data"), prevent_initial_call=True)
def er(n, d):
    return dcc.send_data_frame(pd.DataFrame(d).to_excel,
        f"raw_{datetime.now():%Y%m%d_%H%M%S}.xlsx", index=False) if d else no_update

@callback(Output("dl-all", "data"), Input("exp-all-btn", "n_clicks"),
          State("mean-grid", "rowData"), State("norm-grid", "rowData"),
          State("ec50-grid", "rowData"), State("sim-store", "data"),
          State("prep-store", "data"), prevent_initial_call=True)
def exp_all(n, mean_d, norm_d, ec50_d, raw_d, prep_d):
    if not any([mean_d, norm_d, ec50_d, raw_d]):
        return no_update
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        if ec50_d:
            pd.DataFrame(ec50_d).to_excel(w, sheet_name="EC50 Summary", index=False)
        if mean_d:
            pd.DataFrame(mean_d).to_excel(w, sheet_name="Mean & SD", index=False)
        if norm_d:
            pd.DataFrame(norm_d).to_excel(w, sheet_name="Normalized %", index=False)
        if raw_d:
            pd.DataFrame(raw_d).to_excel(w, sheet_name="Raw Data", index=False)
        if prep_d:
            pd.DataFrame(prep_d).to_excel(w, sheet_name="Stock Prep", index=False)
    buf.seek(0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dcc.send_bytes(buf.getvalue(), f"full_analysis_{ts}.xlsx")


# ═══════════════════════════════════════════════════════════════════
# §15  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  Potency Assay — 384-Well")
    print("  http://127.0.0.1:8050")
    print("  Robust 4PL | QC Flags | Multi-Plate | Validation")
    print("=" * 56 + "\n")
    app.run(debug=True, port=8050)