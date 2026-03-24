"""
384-Well Potency Assay — Unified Web Application
==================================================
Four tabs:
  1. Dilution Block  — serial dilution setup, concentration series
  2. Reagent Prep    — upload antibodies (up to 24), 384-well plate layout, stock plate prep
  3. Cell Calc       — target & effector cell seeding math
  4. Data Analysis   — upload/simulate, raw Mean/SD, normalized %, 4PL, EC50

Plate: 384-well, 2×2 quadruplicates, 12 Ab/plate × 2 plates
Requirements: pip install dash dash-ag-grid plotly pandas openpyxl scipy numpy
"""

import base64, io, math
from datetime import datetime
import numpy as np, pandas as pd
import dash
from dash import html, dcc, Input, Output, State, callback, no_update
import dash_ag_grid as dag
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

# ═══ Constants ═══
ROW_LABELS = [chr(i) for i in range(65, 81)]  # A-P
ABS_PER_PLATE, COLS_PER_AB = 12, 2
MAX_PROTEINS = 6  # per dilution block
AB_COLORS = [
    "#3b82f6","#f59e0b","#10b981","#ef4444","#8b5cf6","#ec4899",
    "#14b8a6","#f97316","#6366f1","#84cc16","#06b6d4","#e11d48",
    "#0d9488","#d97706","#7c3aed","#dc2626","#059669","#2563eb",
    "#c026d3","#ea580c","#4f46e5","#65a30d","#0891b2","#be123c",
]

# ═══ Helpers ═══
def fourpl(x, top, bottom, ec50, hill):
    return bottom + (top - bottom) / (1.0 + (x / ec50) ** hill)

def fit_4pl(concs, means):
    v = [(c, m) for c, m in zip(concs, means) if c > 0 and np.isfinite(m)]
    if len(v) < 4: return None
    x, y = np.array([i[0] for i in v]), np.array([i[1] for i in v])
    try:
        popt, _ = curve_fit(fourpl, x, y, p0=[max(y), min(y), np.median(x), -1.0],
                            bounds=([0, 0, 1e-6, -10], [200, 200, 1e8, 10]), maxfev=10000)
        return tuple(popt)
    except: return None

def mg_nM(mg, mw): return mg / mw * 1e6 if mw and mw > 0 else 0
def nM_mg(nM, mw): return nM * mw / 1e6 if mw and mw > 0 else 0
def fmt_nM(v):
    if v <= 0: return "Bkg"
    if v >= 1000: return f"{v/1000:.3g}\u00b5M"
    return f"{v:.3g}nM" if v >= 1 else f"{v:.2g}nM"

def hex_rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

# ═══ Simulation ═══
SIM_PROFILES = [
    dict(top=42000, bot=8000, ec50=5.0, hill=-1.2),
    dict(top=44000, bot=12000, ec50=15.0, hill=-0.9),
    dict(top=43000, bot=18000, ec50=35.0, hill=-1.5),
    dict(top=41000, bot=6000, ec50=2.0, hill=-1.0),
    dict(top=45000, bot=22000, ec50=50.0, hill=-0.7),
    dict(top=43000, bot=40000, ec50=1e5, hill=-1.0),
]
SIM_AB = [
    dict(top=(92,100), bot=(5,35), ec50=(0.05,500), hill=(-1.8,-0.6), cv=0.10),
    dict(top=(90,100), bot=(80,98), ec50=(1e4,1e6), hill=(-1,-1), cv=0.06),
]

def sim_plate_data(concs, n_prot, seed=42):
    rng = np.random.RandomState(seed)
    raw = {}
    for pi in range(n_prot):
        p = SIM_PROFILES[pi % len(SIM_PROFILES)]
        for ci, c in enumerate(concs):
            tv = p["top"] if c <= 0 else fourpl(c, p["top"], p["bot"], p["ec50"], p["hill"])
            raw[(pi, ci)] = [max(500, round(tv + rng.normal(0, tv * 0.08))) for _ in range(4)]
    return raw

def sim_ab_response(ab_list, concs, seed=42):
    rng = np.random.RandomState(seed)
    results = []
    for ab in ab_list:
        pr = SIM_AB[0] if ab.get("ctrl_type", "Positive") == "Positive" else SIM_AB[1]
        top, bot = rng.uniform(*pr["top"]), rng.uniform(*pr["bot"])
        ec50 = 10 ** rng.uniform(np.log10(pr["ec50"][0]), np.log10(pr["ec50"][1]))
        hill = rng.uniform(*pr["hill"])
        for c in concs:
            tv = top if c <= 0 else fourpl(c, top, bot, ec50, hill)
            for rep in range(4):
                m = max(0, min(150, tv + rng.normal(0, abs(tv) * pr["cv"])))
                results.append({"Antibody": ab["name"], "Conc_nM": c, "Replicate": rep + 1,
                                "Viability_%": round(m, 2), "Control": ab.get("ctrl_type", "Positive")})
    return results

# ═══ Shared analysis functions ═══
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

def run_4pl_analysis(ab_list, sim_data, concs):
    analysis = []
    for ab in ab_list:
        ad = [r for r in sim_data if r["Antibody"] == ab["name"]]
        ms, cs, cvs = [], [], []
        for c in concs:
            reps = [r["Viability_%"] for r in ad if abs(r["Conc_nM"] - c) < 0.001]
            if len(reps) >= 2 and c > 0:
                mn, sd = np.mean(reps), np.std(reps, ddof=1)
                ms.append(mn); cs.append(c); cvs.append(sd / mn * 100 if mn > 0 else 0)
        params = fit_4pl(cs, ms)
        if params:
            top, bottom, ec50, hill = params
            yp = [fourpl(x, *params) for x in cs]
            ssr = sum((m - p) ** 2 for m, p in zip(ms, yp))
            sst = sum((m - np.mean(ms)) ** 2 for m in ms)
            r2 = 1 - ssr / sst if sst > 0 else 0
        else:
            top = bottom = ec50 = hill = r2 = None
        analysis.append({
            "Antibody": ab["name"], "Control": ab.get("ctrl_type", "Positive"),
            "EC50 (nM)": round(ec50, 3) if ec50 else "N/A",
            "Top (%)": round(top, 1) if top else "N/A", "Bottom (%)": round(bottom, 1) if bottom else "N/A",
            "Hill Slope": round(hill, 3) if hill else "N/A",
            "R\u00b2": round(r2, 4) if r2 is not None else "N/A",
            "Max Kill (%)": round(top - bottom, 1) if top is not None and bottom is not None else "N/A",
            "Avg CV%": round(np.mean(cvs), 1) if cvs else "N/A",
            "_color": ab["color"], "_params": list(params) if params else None,
            "_concs": cs, "_means": ms})
    return analysis

def make_dose_fig(ab_list, analysis, per_row=4):
    na = len(ab_list); nc = min(per_row, na); nr = math.ceil(na / nc)
    fig = make_subplots(rows=nr, cols=nc, subplot_titles=[a["name"][:22] for a in ab_list],
                        horizontal_spacing=0.06, vertical_spacing=0.06)
    for idx, (ab, ad) in enumerate(zip(ab_list, analysis)):
        ri, ci = divmod(idx, nc); ri += 1; ci += 1
        if ad["_concs"]:
            fig.add_trace(go.Scatter(x=ad["_concs"], y=ad["_means"], mode="markers",
                marker=dict(size=7, color=ab["color"], line=dict(width=1, color="white")),
                showlegend=False, hovertemplate="%{x:.2f} nM<br>%{y:.1f}%<extra></extra>"), row=ri, col=ci)
            if ad["_params"]:
                xf = np.logspace(np.log10(min(ad["_concs"]) * 0.3), np.log10(max(ad["_concs"]) * 3), 100)
                yf = [fourpl(x, *ad["_params"]) for x in xf]
                fig.add_trace(go.Scatter(x=xf.tolist(), y=yf, mode="lines",
                    line=dict(color=ab["color"], width=2), showlegend=False), row=ri, col=ci)
                ev = ad["_params"][2]; ey = fourpl(ev, *ad["_params"])
                fig.add_trace(go.Scatter(x=[ev], y=[ey], mode="markers",
                    marker=dict(size=10, symbol="diamond", color="red", line=dict(width=1.5, color="white")),
                    showlegend=False, hovertemplate=f"EC50: {ev:.2f} nM<extra></extra>"), row=ri, col=ci)
            fig.update_xaxes(type="log", showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=8), row=ri, col=ci)
            fig.update_yaxes(range=[-5, 115], showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=8), row=ri, col=ci)
    fig.update_layout(height=max(400, nr * 320), plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=50, r=20, t=40, b=30), font=dict(size=10))
    for ann in fig["layout"]["annotations"]: ann["font"] = dict(size=10, color="#334155")
    return fig

def process_rlu_data(raw, concs, names):
    """Process raw RLU dict → mean/SD table, normalized table, 4PL, EC50."""
    n_prot = len(names)
    mean_cols = [{"field": "Point", "width": 65, "cellStyle": {"fontFamily": MF, "fontWeight": "700"}}]
    for nm in names:
        mean_cols += [{"field": f"{nm} Mean", "width": 120, "valueFormatter": FMT_AUTO, "cellStyle": {"fontFamily": MF, "fontWeight": "600", "color": "#3b82f6"}},
                      {"field": f"{nm} SD", "width": 100, "valueFormatter": FMT_AUTO, "cellStyle": {"fontFamily": MF, "color": "#64748b"}}]
    mean_rows, means_by = [], {pi: [] for pi in range(n_prot)}
    for ci, c in enumerate(concs):
        row = {"Point": ci + 1}
        for pi, nm in enumerate(names):
            reps = raw.get((pi, ci), [])
            if reps:
                mn, sd = np.mean(reps), np.std(reps, ddof=1) if len(reps) > 1 else 0
                row[f"{nm} Mean"], row[f"{nm} SD"] = round(mn, 1), round(sd, 1)
                means_by[pi].append((c, mn, sd))
        mean_rows.append(row)

    bg = {pi: next((m for c, m, s in means_by[pi] if c <= 0), 1) for pi in range(n_prot)}
    norm_cols = [{"field": "Log(nM)", "width": 80, "valueFormatter": FMT_DEC2, "cellStyle": {"fontFamily": MF, "fontWeight": "700"}}]
    for nm in names:
        norm_cols += [{"field": f"{nm} Mean%", "width": 100, "valueFormatter": FMT_DEC1, "cellStyle": {"fontFamily": MF, "fontWeight": "600", "color": "#10b981"}},
                      {"field": f"{nm} CV%", "width": 85, "valueFormatter": FMT_DEC1, "cellStyle": {"fontFamily": MF}},
                      {"field": f"{nm} N", "width": 50, "cellStyle": {"fontFamily": MF, "color": "#94a3b8"}}]
    norm_rows, norm_d = [], {pi: {"c": [], "m": [], "cv": []} for pi in range(n_prot)}
    for ci, c in enumerate(concs):
        log_c = round(math.log10(c), 5) if c > 0 else -3
        row = {"Log(nM)": log_c}
        for pi, nm in enumerate(names):
            reps = raw.get((pi, ci), [])
            if reps:
                mn, sd = np.mean(reps), np.std(reps, ddof=1) if len(reps) > 1 else 0
                pct = mn / bg[pi] * 100 if bg[pi] > 0 else 0
                cv = sd / bg[pi] * 100 if bg[pi] > 0 else 0
                row[f"{nm} Mean%"], row[f"{nm} CV%"], row[f"{nm} N"] = round(pct, 1), round(cv, 1), len(reps)
                if c > 0:
                    norm_d[pi]["c"].append(c); norm_d[pi]["m"].append(pct)
                    norm_d[pi]["cv"].append(abs(cv / pct * 100) if pct > 0 else 0)
        norm_rows.append(row)

    # 4PL
    nc_ = min(4, n_prot); nr_ = math.ceil(n_prot / nc_)
    fig = make_subplots(rows=nr_, cols=nc_, subplot_titles=names,
                        horizontal_spacing=0.06, vertical_spacing=0.08)
    ec50_rows = []
    for pi, nm in enumerate(names):
        ri, ci = divmod(pi, nc_); ri += 1; ci += 1
        cs, ms = norm_d[pi]["c"], norm_d[pi]["m"]
        color = AB_COLORS[pi % len(AB_COLORS)]
        if cs:
            sds = [norm_d[pi]["cv"][j] * ms[j] / 100 if ms[j] > 0 else 0 for j in range(len(ms))]
            fig.add_trace(go.Scatter(x=cs, y=ms, mode="markers",
                error_y=dict(type="data", array=sds, visible=True, color=hex_rgba(color, 0.4), thickness=1.5),
                marker=dict(size=8, color=color, line=dict(width=1.5, color="white")),
                showlegend=False, hovertemplate="%{x:.2f} nM<br>%{y:.1f}%<extra></extra>"), row=ri, col=ci)
            params = fit_4pl(cs, ms)
            if params:
                top, bottom, ec50, hill = params
                xf = np.logspace(np.log10(min(cs) * 0.3), np.log10(max(cs) * 3), 100)
                yf = [fourpl(x, *params) for x in xf]
                fig.add_trace(go.Scatter(x=xf.tolist(), y=yf, mode="lines",
                    line=dict(color=color, width=2.5), showlegend=False), row=ri, col=ci)
                ev = ec50; ey = fourpl(ev, *params)
                fig.add_trace(go.Scatter(x=[ev], y=[ey], mode="markers",
                    marker=dict(size=12, symbol="diamond", color="#ef4444", line=dict(width=2, color="white")),
                    showlegend=False, hovertemplate=f"EC50: {ev:.2f} nM<extra></extra>"), row=ri, col=ci)
                yp = [fourpl(x, *params) for x in cs]
                ssr = sum((a - b) ** 2 for a, b in zip(ms, yp))
                sst = sum((a - np.mean(ms)) ** 2 for a in ms)
                r2 = 1 - ssr / sst if sst > 0 else 0
                ec50_rows.append({"Protein": nm, "EC50 (nM)": round(ec50, 3), "Top (%)": round(top, 1),
                    "Bottom (%)": round(bottom, 1), "Hill Slope": round(hill, 3), "R\u00b2": round(r2, 4),
                    "Max Kill (%)": round(top - bottom, 1), "Avg CV%": round(np.mean(norm_d[pi]["cv"]), 1)})
            else:
                ec50_rows.append({"Protein": nm, "EC50 (nM)": "N/A", "Top (%)": "N/A", "Bottom (%)": "N/A",
                    "Hill Slope": "N/A", "R\u00b2": "N/A", "Max Kill (%)": "N/A", "Avg CV%": "N/A"})
            fig.update_xaxes(type="log", showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9),
                             title=dict(text="nM", font=dict(size=10)), row=ri, col=ci)
            fig.update_yaxes(range=[-5, 130], showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9),
                             title=dict(text="% viability", font=dict(size=10)), row=ri, col=ci)
    fig.update_layout(height=max(400, nr_ * 320), plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=50, r=20, t=40, b=30), font=dict(size=10))
    for ann in fig["layout"]["annotations"]: ann["font"] = dict(size=12, color="#334155")
    return mean_cols, mean_rows, norm_cols, norm_rows, fig, ec50_rows

# ═══ App + Styles ═══
app = dash.Dash(__name__, title="Potency Assay \u2014 384-Well", suppress_callback_exceptions=True)

C = {"bg": "#f1f5f9", "sf": "#ffffff", "sa": "#f8fafc", "bd": "#e2e8f0", "tx": "#0f172a",
     "mt": "#64748b", "ac": "#3b82f6", "al": "#eff6ff", "dg": "#ef4444",
     "wb": "#fffbeb", "wbd": "#fde68a", "wt": "#92400e", "ok": "#10b981", "ol": "#ecfdf5"}
FF = "'Segoe UI','Helvetica Neue',Arial,sans-serif"
MF = "'Consolas','Courier New',monospace"
card = {"background": C["sf"], "border": f"1px solid {C['bd']}", "borderRadius": "10px",
        "boxShadow": "0 1px 3px rgba(15,23,42,.06)", "overflow": "hidden"}
chdr = {"padding": "14px 20px", "borderBottom": f"1px solid {C['bd']}", "fontWeight": "600",
        "fontSize": "13px", "letterSpacing": ".5px",
        "color": C["mt"], "background": C["sa"], "display": "flex", "alignItems": "center", "gap": "8px"}
lbl = {"fontSize": "11px", "fontWeight": "600", "color": C["mt"],
       "letterSpacing": ".4px", "marginBottom": "4px"}
inp = {"width": "100%", "padding": "8px 12px", "border": f"1px solid {C['bd']}",
       "borderRadius": "6px", "fontSize": "14px", "fontFamily": FF, "color": C["tx"]}
inpm = {**inp, "fontFamily": MF, "fontWeight": "600"}
btn = {"padding": "10px 20px", "borderRadius": "6px", "fontSize": "13px", "fontWeight": "600",
       "cursor": "pointer", "border": "none", "background": C["ac"], "color": "white", "fontFamily": FF}
btno = {"padding": "7px 14px", "borderRadius": "6px", "fontSize": "12px", "fontWeight": "600",
        "cursor": "pointer", "border": f"1px solid {C['bd']}", "background": "transparent", "color": C["tx"], "fontFamily": FF}
upz = {"border": f"2px dashed {C['bd']}", "borderRadius": "8px", "padding": "20px",
       "textAlign": "center", "cursor": "pointer", "background": C["sa"]}
ts = {"fontSize": "14px", "fontWeight": "600", "padding": "14px 24px"}
tsel = {**ts, "borderTop": f"3px solid {C['ac']}", "color": C["ac"]}
ibox = {"background": C["ol"], "border": f"1px solid {C['ok']}", "borderRadius": "6px",
        "padding": "8px 14px", "fontSize": "12px", "color": "#065f46", "fontFamily": MF}

def minp(id, val, **kw): return dcc.Input(id=id, value=val, type="number", style=inpm, **kw)

# AG Grid number formatters (JS)
FMT_INT = {"function": "params.value != null && !isNaN(params.value) ? Number(params.value).toLocaleString('en-US',{maximumFractionDigits:0}) : params.value"}
FMT_DEC1 = {"function": "params.value != null && !isNaN(params.value) ? Number(params.value).toLocaleString('en-US',{minimumFractionDigits:1,maximumFractionDigits:1}) : params.value"}
FMT_DEC2 = {"function": "params.value != null && !isNaN(params.value) ? Number(params.value).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}) : params.value"}
FMT_DEC3 = {"function": "params.value != null && !isNaN(params.value) ? Number(params.value).toLocaleString('en-US',{minimumFractionDigits:3,maximumFractionDigits:3}) : params.value"}
FMT_DEC4 = {"function": "params.value != null && !isNaN(params.value) ? Number(params.value).toLocaleString('en-US',{minimumFractionDigits:4,maximumFractionDigits:4}) : params.value"}
FMT_AUTO = {"function": "params.value != null && !isNaN(params.value) ? Number(params.value).toLocaleString('en-US') : params.value"}

# ═══════════════════════ LAYOUT ═══════════════════════
app.layout = html.Div([
    dcc.Store(id="conc-store", data=[]),
    dcc.Store(id="ab-store", data=[]),
    dcc.Store(id="prep-store", data=[]),
    dcc.Store(id="grid-store", data=[]),
    dcc.Store(id="sim-store", data=[]),
    dcc.Store(id="analysis-store", data=[]),

    # Header
    html.Div([
        html.Div([
            html.H1("Potency Assay \u2014 384-Well Plate", style={"fontSize": "22px", "fontWeight": "700", "margin": "0"}),
            html.P("2\u00d72 Quadruplicate | Serial Dilution | Cell Calculation | 4PL EC50",
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
              "padding": "20px 32px", "display": "flex", "justifyContent": "space-between", "alignItems": "center"}),

    # ═══ TABS ═══
    html.Div([dcc.Tabs(id="main-tabs", value="tab-prep", children=[

        # ══════════ TAB 1: DILUTION BLOCK PREPARATION ══════════
        dcc.Tab(label="\U0001F9EA Dilution Block Preparation", value="tab-prep", style=ts, selected_style=tsel, children=[html.Div([
            html.Div([
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
                html.Div([
                    html.Div("2. Dilution & Stock Plate Parameters", style=chdr),
                    html.Div([
                        html.Div([
                            html.Div([html.Label("Starting Conc (nM)", style={**lbl, "color": C["ac"], "fontSize": "12px"}),
                                      minp("start-conc", 100, step=10)], style={"flex": "1"}),
                            html.Div([html.Label("Dilution Factor", style=lbl),
                                      dcc.Dropdown(id="dil-factor", options=[{"label": f"{x}x", "value": x} for x in [2, 3, 4, 5, 10]],
                                                   value=2, clearable=False)], style={"flex": "1"}),
                            html.Div([html.Label("# Points", style=lbl),
                                      dcc.Dropdown(id="n-dil", options=[{"label": str(x), "value": x} for x in range(5, 11)],
                                                   value=7, clearable=False)], style={"flex": "1"}),
                        ], style={"display": "flex", "gap": "10px", "marginBottom": "12px"}),
                        html.Div([
                            html.Div([html.Label("Block Vol (\u00b5L)", style=lbl), minp("block-vol", 500, step=50)], style={"flex": "1"}),
                            html.Div([html.Label("Culture Vol (\u00b5L)", style=lbl), minp("cult-vol", 50, step=5)], style={"flex": "1"}),
                            html.Div([html.Label("Add Into Culture (\u00b5L)", style=lbl), minp("add-vol", 10, step=1)], style={"flex": "1"}),
                        ], style={"display": "flex", "gap": "10px", "marginBottom": "12px"}),
                        html.Div([
                            html.Div([html.Label("Stock Plate Vol (\u00b5L)", style=lbl), minp("stock-vol", 200, step=10)], style={"flex": "1"}),
                            html.Div([html.Label("Conc Factor", style=lbl),
                                      dcc.Dropdown(id="conc-factor", options=[{"label": f"{x}x", "value": x} for x in [2, 3, 4, 5, 10]],
                                                   value=5, clearable=False)], style={"flex": "1"}),
                            html.Div([html.Label("Transfer Ratio", style=lbl),
                                      html.Div(id="xfer-label", style={**ibox, "textAlign": "center"})], style={"flex": "1"}),
                        ], style={"display": "flex", "gap": "10px", "marginBottom": "14px"}),
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
                    {"field": "Working Vol (\u00b5L)", "width": 140, "valueFormatter": FMT_AUTO, "cellStyle": {"fontFamily": MF}},
                ], rowData=[], defaultColDef={"sortable": True, "resizable": True},
                    dashGridOptions={"domLayout": "autoHeight"}, style={"width": "100%"}, className="ag-theme-alpine"),
            ], style={**card, "marginBottom": "24px"}),
            # Plate Viz
            html.Div([
                html.Div([html.Span("384-Well Layout (2\u00d72 Quad)", style={"fontWeight": "600", "fontSize": "13px",
                    "letterSpacing": ".5px", "color": C["mt"]}),
                    html.Span(style={"flex": "1"}), html.Span(id="plate-sub", style={"fontSize": "11px", "color": C["mt"], "fontFamily": MF})],
                    style={**chdr, "justifyContent": "space-between"}),
                dcc.Tabs(id="plate-tabs", value="p1", children=[
                    dcc.Tab(label="Plate 1 (Ab 1\u201312)", value="p1", style={"fontSize": "12px", "fontWeight": "600"},
                            selected_style={"fontSize": "12px", "fontWeight": "600", "borderTop": f"3px solid {C['ac']}"}),
                    dcc.Tab(label="Plate 2 (Ab 13\u201324)", value="p2", style={"fontSize": "12px", "fontWeight": "600"},
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
                    style={"display": "flex", "alignItems": "center", "padding": "10px 20px", "borderBottom": f"1px solid {C['bd']}"}),
                dag.AgGrid(id="prep-grid", columnDefs=[
                    {"field": "Well", "width": 85, "pinned": "left", "cellStyle": {"fontFamily": MF, "fontWeight": "700"}},
                    {"field": "Antibody", "width": 140}, {"field": "Block", "width": 60},
                    {"field": "Stock (nM)", "width": 115, "valueFormatter": FMT_AUTO, "cellStyle": {"fontFamily": MF, "fontWeight": "600", "color": "#10b981"}},
                    {"field": "Ab (\u00b5L)", "width": 90, "valueFormatter": FMT_DEC2, "cellStyle": {"fontFamily": MF, "fontWeight": "700", "color": "#3b82f6"}},
                    {"field": "Medium (\u00b5L)", "width": 100, "valueFormatter": FMT_DEC2, "cellStyle": {"fontFamily": MF, "fontWeight": "700", "color": "#f59e0b"}},
                    {"field": "Xfer (\u00b5L)", "width": 90, "valueFormatter": FMT_DEC2, "cellStyle": {"fontFamily": MF, "fontWeight": "600", "color": "#8b5cf6"}},
                    {"field": "Total (\u00b5L)", "width": 80, "valueFormatter": FMT_DEC2, "cellStyle": {"fontFamily": MF}},
                    {"field": "Note", "width": 250, "cellStyle": {"color": "#64748b", "fontSize": "11px"}},
                ], rowData=[], defaultColDef={"sortable": True, "filter": True, "resizable": True},
                    dashGridOptions={"animateRows": True, "domLayout": "autoHeight"},
                    style={"width": "100%"}, className="ag-theme-alpine"),
            ], style={**card, "marginBottom": "24px"}),
            html.Div(id="protocol-note", style={"marginBottom": "24px"}),
        ], style={"padding": "24px 0"})]),

        # ══════════ TAB 3: CELL CALCULATION ══════════
        dcc.Tab(label="\U0001F9EC Cell Calculation", value="tab-cell", style=ts, selected_style=tsel, children=[html.Div([
            dcc.Store(id="cc-all-store", data={}),

            # How many cell lines?
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

            # All 5 cell line sections (always in DOM; hidden ones get display:none)
            *[html.Div(id=f"cc-section-{i}", children=[
                html.Div([
                    html.Div([
                        html.Span(f"Cell Line {i+1}", style={"fontWeight": "600", "fontSize": "14px",
                            "color": AB_COLORS[i]}),
                        html.Span(" \u2014 Cancer Cell Stock", style={"fontWeight": "600", "fontSize": "13px",
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
                        html.Div([html.Label("\u00b5L/Well", style=lbl),
                                  minp(f"cc-wv-{i}", 40, step=5)], style={"flex": "0.5"}),
                        html.Div([html.Label("Total Wells", style=lbl),
                                  minp(f"cc-wells-{i}", 500, step=1)], style={"flex": "0.6"}),
                        html.Div([html.Label("Dead Vol (\u00b5L)", style=lbl),
                                  minp(f"cc-dead-{i}", 5000, step=1000)], style={"flex": "0.7"}),
                    ], style={"display": "flex", "gap": "12px"})], style={"padding": "14px 16px"}),
                ], style={**card, "marginBottom": "4px"}),
                html.Div(id=f"cc-result-{i}"),
                html.Div(id=f"cc-summary-{i}", style={"marginBottom": "20px"}),
            ], style={"display": "none"} if i > 0 else {}) for i in range(5)],

            # Print summary section (hidden when only 1 line)
            html.Div(id="cc-print-section", children=[
                html.Div([
                    html.Div([
                        html.Span("All Cell Lines \u2014 Preparation Summary", style={"fontWeight": "600", "fontSize": "13px",
                            "letterSpacing": ".5px", "color": C["mt"]}),
                        html.Span(style={"flex": "1"}),
                        html.Button("\U0001F5A8 Print", id="cc-print-btn",
                                    style={**btn, "background": "#0f172a", "padding": "8px 16px", "fontSize": "12px"}),
                        html.Button("\u2B07 Export .xlsx", id="cc-export-btn",
                                    style={**btno, "marginLeft": "8px"}),
                        dcc.Download(id="dl-cellcalc"),
                    ], style={**chdr, "justifyContent": "space-between"}),
                    html.Div(id="cc-print-content", style={"padding": "16px"}),
                ], style=card),
            ], style={"display": "none"}),

        ], style={"padding": "24px 0"})]),

        # ══════════ TAB 4: DATA ANALYSIS ══════════
        dcc.Tab(label="\U0001F4CA Data Analysis", value="tab-data", style=ts, selected_style=tsel, children=[html.Div([
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
                    ], style={"flex": "2"}),
                    html.Div([
                        html.Div("or simulate", style={"textAlign": "center", "color": C["mt"], "fontSize": "12px", "marginBottom": "8px"}),
                        html.Button("\U0001F9EA Simulate RLU Data", id="sim-rlu-btn", n_clicks=0,
                                    style={**btn, "background": "#10b981", "width": "100%", "padding": "12px", "marginBottom": "8px"}),
                        html.Button("\U0001F4CA Simulate Viability Data", id="sim-via-btn", n_clicks=0,
                                    style={**btn, "background": "#8b5cf6", "width": "100%", "padding": "12px"}),
                    ], style={"flex": "1", "display": "flex", "flexDirection": "column", "justifyContent": "center"}),
                ], style={"display": "flex", "gap": "20px", "padding": "16px", "alignItems": "stretch"}),
            ], style={**card, "marginBottom": "24px"}),

            # ── Paste from Plate Reader ──
            html.Div([
                html.Div([
                    html.Span("Paste from Plate Reader (EnVision / 384-Well Grid)", style={"fontWeight": "600", "fontSize": "13px",
                        "letterSpacing": ".5px", "color": C["mt"]}),
                    html.Span(style={"flex": "1"}),
                    html.Div([
                        html.Label("Plate:", style={"fontSize": "12px", "fontWeight": "600", "color": C["mt"], "marginRight": "6px"}),
                        dcc.Dropdown(id="paste-plate", options=[{"label": "Plate 1", "value": 1}, {"label": "Plate 2", "value": 2}],
                                     value=1, clearable=False, style={"width": "110px", "fontSize": "12px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                ], style={**chdr, "justifyContent": "space-between"}),
                html.Div([
                    html.Div([
                        html.Div("Copy the 384-well readout from EnVision and paste below (16 rows \u00d7 24 columns).",
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
                        html.Button("\U0001F4CB Parse & Analyze Plate", id="parse-plate-btn", n_clicks=0,
                                    style={**btn, "background": "#0d9488", "width": "100%", "padding": "14px",
                                           "fontSize": "13px", "marginBottom": "10px"}),
                        html.Div(id="paste-status", style={"fontSize": "12px"}),
                        html.Div([
                            html.Div("How it works:", style={"fontSize": "11px", "fontWeight": "600", "color": C["mt"], "marginBottom": "4px"}),
                            html.Div("1. Generate plate layout in Tab 1", style={"fontSize": "11px", "color": C["mt"]}),
                            html.Div("2. Run assay on plate reader", style={"fontSize": "11px", "color": C["mt"]}),
                            html.Div("3. Copy 16\u00d724 grid from EnVision", style={"fontSize": "11px", "color": C["mt"]}),
                            html.Div("4. Paste here \u2192 auto-mapped to antibodies", style={"fontSize": "11px", "color": C["mt"]}),
                        ], style={"marginTop": "12px", "padding": "10px", "background": C["wb"],
                                  "borderRadius": "6px", "border": f"1px solid {C['wbd']}"}),
                    ], style={"flex": "1", "display": "flex", "flexDirection": "column"}),
                ], style={"display": "flex", "gap": "16px", "padding": "16px", "alignItems": "flex-start"}),
            ], style={**card, "marginBottom": "24px"}),

            # Download All
            html.Div([
                html.Button("\u2B07 Download All Analysis (.xlsx)", id="exp-all-btn", n_clicks=0,
                            style={**btn, "background": "#0f172a", "width": "100%", "padding": "14px",
                                   "fontSize": "14px", "letterSpacing": ".5px"}),
                dcc.Download(id="dl-all"),
            ], style={"marginBottom": "24px"}),

            # Dose-response
            html.Div([html.Div([html.Span("\U0001F4C8 Dose-Response (4PL)", style={"fontWeight": "600", "fontSize": "13px",
                "letterSpacing": ".5px", "color": C["mt"]}), html.Span(style={"flex": "1"}),
                html.Button("Export EC50", id="exp-ec50-btn", n_clicks=0, style=btno), dcc.Download(id="dl-ec50")],
                style={**chdr, "justifyContent": "space-between"}),
                dcc.Graph(id="dose-fig", figure=go.Figure(), config={"displayModeBar": True}),
            ], style={**card, "marginBottom": "24px"}),
            # EC50
            html.Div([html.Div("EC50 Summary", style=chdr),
                dag.AgGrid(id="ec50-grid", columnDefs=[
                    {"field": "Protein", "width": 150, "cellStyle": {"fontWeight": "600"}},
                    {"field": "EC50 (nM)", "width": 120, "valueFormatter": FMT_AUTO, "cellStyle": {"fontFamily": MF, "fontWeight": "700", "color": "#3b82f6"}},
                    {"field": "Top (%)", "width": 85, "valueFormatter": FMT_DEC1, "cellStyle": {"fontFamily": MF}},
                    {"field": "Bottom (%)", "width": 95, "valueFormatter": FMT_DEC1, "cellStyle": {"fontFamily": MF}},
                    {"field": "Hill Slope", "width": 95, "valueFormatter": FMT_DEC3, "cellStyle": {"fontFamily": MF}},
                    {"field": "R\u00b2", "width": 75, "valueFormatter": FMT_DEC4, "cellStyle": {"fontFamily": MF}},
                    {"field": "Max Kill (%)", "width": 100, "valueFormatter": FMT_DEC1, "cellStyle": {"fontFamily": MF, "fontWeight": "600", "color": "#ef4444"}},
                    {"field": "Avg CV%", "width": 85, "valueFormatter": FMT_DEC1, "cellStyle": {"fontFamily": MF}},
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
            # Raw reps (for viability data)
            html.Div([html.Div([html.Span("Raw Replicate Data", style={"fontWeight": "600", "fontSize": "13px",
                "letterSpacing": ".5px", "color": C["mt"]}), html.Span(style={"flex": "1"}),
                html.Button("Export", id="exp-raw-btn", n_clicks=0, style=btno), dcc.Download(id="dl-raw")],
                style={**chdr, "justifyContent": "space-between"}),
                html.Div([html.Label("Filter:", style={"fontSize": "12px", "fontWeight": "600", "color": C["mt"], "marginRight": "8px"}),
                    dcc.Dropdown(id="raw-filter", placeholder="All", clearable=True, style={"flex": "1", "fontSize": "13px"})],
                    style={"display": "flex", "alignItems": "center", "padding": "10px 20px", "borderBottom": f"1px solid {C['bd']}"}),
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


# ═══════════════════════ CALLBACKS ═══════════════════════

# ── Transfer ratio label ──
@callback(Output("xfer-label", "children"), Input("conc-factor", "value"))
def xfer_lbl(cf): return f"1 vol + {(cf or 5) - 1} vol medium"

# ── Tab 2: Upload Antibodies ──
@callback(Output("ab-store", "data"), Output("upload-status", "children"), Output("ab-summary", "children"),
          Input("reagent-upload", "contents"), State("reagent-upload", "filename"), prevent_initial_call=True)
def load_ab(contents, fn):
    if not contents: return no_update, no_update, no_update
    _, cs = contents.split(","); dec = base64.b64decode(cs)
    try:
        df = pd.read_csv(io.StringIO(dec.decode("utf-8"))) if fn.endswith(".csv") else pd.read_excel(io.BytesIO(dec))
        def fc(cols, cands):
            for ca in cands:
                for dc in cols:
                    if ca.lower().replace(" ", "") in dc.lower().replace(" ", ""): return dc
            return None
        cid = fc(df.columns, ["AB_ID", "antibody", "name", "sample"])
        cmw = fc(df.columns, ["MW", "molecular_weight", "kDa"])
        cco = fc(df.columns, ["Concentration", "conc", "mg/mL", "stock"])
        cct = fc(df.columns, ["Control", "control_type", "type", "ctrl"])
        if not cid or not cco:
            return no_update, html.Span(f"Missing columns. Found: {list(df.columns)}", style={"color": C["dg"]}), no_update
        ab_list, seen = [], {}
        for i, row in df.iterrows():
            if i >= 24: break
            nm = str(row[cid])
            if nm in seen: seen[nm] += 1; nm = f"{nm}_{seen[nm]}"
            else: seen[nm] = 1
            co = float(row[cco]); mw = float(row[cmw]) if cmw and pd.notna(row.get(cmw)) else 150
            ct = str(row[cct]).strip() if cct and pd.notna(row.get(cct)) else "Positive"
            ab_list.append({"name": nm, "conc_mg": co, "mw": mw, "color": AB_COLORS[i % len(AB_COLORS)],
                            "col_start": (i % ABS_PER_PLATE) * COLS_PER_AB + 1,
                            "plate": 1 + i // ABS_PER_PLATE, "idx": i, "ctrl_type": ct})
        n = len(ab_list); np_ = sum(1 for a in ab_list if a["ctrl_type"] == "Positive")
        st = html.Span([html.Strong(f"{n} antibodies", style={"color": C["ac"]}),
            html.Span(f" ({np_} pos, {n - np_} neg) from {fn}")])
        rows = [html.Tr([
            html.Td(f"P{a['plate']}", style={"padding": "2px 6px", "fontFamily": MF, "fontSize": "10px"}),
            html.Td(html.Span("", style={"width": "8px", "height": "8px", "borderRadius": "50%",
                "background": a["color"], "display": "inline-block"}), style={"padding": "2px 4px"}),
            html.Td(a["name"], style={"padding": "2px 6px", "fontSize": "11px", "fontWeight": "500"}),
            html.Td(f"{a['conc_mg']} mg/mL", style={"padding": "2px 6px", "fontFamily": MF, "fontSize": "10px"}),
            html.Td(a["ctrl_type"], style={"padding": "2px 6px", "fontSize": "10px",
                "color": C["dg"] if a["ctrl_type"] == "Negative" else C["ok"]}),
        ], style={"background": C["wb"] if a["ctrl_type"] == "Negative" else "transparent"}) for a in ab_list]
        tbl = html.Table([html.Tbody(rows)], style={"width": "100%", "borderCollapse": "collapse", "background": C["sa"]})
        return ab_list, st, tbl
    except Exception as e:
        return no_update, html.Span(f"Error: {e}", style={"color": C["dg"]}), no_update

# ── Calc plate layout + stock prep + concentration series ──
@callback(Output("plate-sub", "children"), Output("prep-filter", "options"), Output("prep-filter", "value"),
          Output("protocol-note", "children"), Output("prep-store", "data"), Output("grid-store", "data"),
          Output("conc-grid", "rowData"), Output("conc-store", "data"),
          Input("calc-btn", "n_clicks"), State("ab-store", "data"),
          State("start-conc", "value"), State("dil-factor", "value"), State("n-dil", "value"),
          State("block-vol", "value"), State("stock-vol", "value"), State("conc-factor", "value"),
          prevent_initial_call=True)
def calc_prep(n, ab_list, start, df_, npts, bv, sv, cf):
    _n = no_update
    if not ab_list: return _n, _n, _n, _n, _n, _n, _n, _n
    start = max(start or 100, 0.01); df_ = max(df_ or 2, 2); npts = npts or 7
    bv = max(bv or 500, 10); sv = max(sv or 200, 10); cf = max(cf or 5, 2)

    # Generate concentration series
    twv = round(bv / (1 - 1 / df_), 2)
    concs = [start / (df_ ** i) for i in range(npts)] + [0]
    conc_rows = [{"Point": i + 1, "Conc (nM)": round(c, 6),
                  "Log(nM)": round(math.log10(c), 5) if c > 0 else -3,
                  "Working Vol (\u00b5L)": twv} for i, c in enumerate(concs)]

    # Stock plate prep
    tv = round(sv / (df_ - 1), 2); pva = sv + tv
    prep, gd = [], []
    for ab in ab_list:
        mw, sc = ab["mw"], ab["conc_mg"]
        for ci, wn in enumerate(concs):
            is_bg = (wn <= 0); sn = wn * cf if not is_bg else 0; sm = nM_mg(sn, mw) if not is_bg else 0
            bl = f"{ROW_LABELS[ci * 2]}-{ROW_LABELS[ci * 2 + 1]}" if ci * 2 + 1 < 16 else "BKG"
            cs_ = ab["col_start"]; well = f"{ROW_LABELS[min(ci * 2, 15)]}{cs_}"
            if is_bg: av, mv, xv, note = 0, sv, 0, "Medium only"
            elif ci == 0:
                av = (sm * pva) / sc if sc > 0 else 0; mv = pva - av; xv = tv
                note = f"From stock; xfer {tv}\u00b5L \u2192 next"
                if av > pva: note = f"EXCEEDS {av:.1f}\u00b5L"; av, mv = pva, 0
            else:
                av, mv = 0, sv; prev = ROW_LABELS[(ci - 1) * 2]
                xv = tv if ci < len(concs) - 2 else 0
                note = f"Add {tv}\u00b5L from {prev}-block" + (", xfer out" if xv else f", discard {tv}\u00b5L")
            prep.append({"Well": well, "Antibody": ab["name"], "Block": bl, "Plate": ab["plate"],
                "Stock (nM)": round(sn, 4) if not is_bg else 0, "Ab (\u00b5L)": round(av, 2),
                "Medium (\u00b5L)": round(max(mv, 0), 2), "Xfer (\u00b5L)": round(xv, 2),
                "Total (\u00b5L)": round(pva, 2) if ci == 0 else sv, "Note": note})
            for dr in range(2):
                for dc in range(2):
                    ri = min(ci * 2 + dr, 15)
                    gd.append({"row": ri, "col": cs_ + dc, "work_nM": wn, "ab_name": ab["name"],
                                "color": ab["color"], "is_bg": is_bg, "plate": ab["plate"],
                                "ctrl": ab.get("ctrl_type", "Positive")})
    ab_names = list(dict.fromkeys(r["Antibody"] for r in prep))
    n_concs = len(concs)
    blk_opts = [{"label": f"Block {ROW_LABELS[i*2]}-{ROW_LABELS[i*2+1]} \u2014 all Ab", "value": f"BLK:{i}"}
                for i in range(min(n_concs, 8))]
    ab_opts = [{"label": nm, "value": f"AB:{nm}"} for nm in ab_names]
    dd = [{"label": "\u2501 By Block", "value": "", "disabled": True}] + blk_opts + \
         [{"label": "\u2501 By Antibody", "value": "", "disabled": True}] + ab_opts
    sub = f"12 Ab/plate \u00d7 2 plates | 2\u00d72 quad | {df_}x dilution"
    prot = html.Div([
        html.Strong("Protocol:"), html.Br(),
        html.Span(f"1. Row A-B block: Ab + medium \u2192 {pva}\u00b5L. Stock = {cf}x."), html.Br(),
        html.Span(f"2. Serial dilute: xfer {tv}\u00b5L between blocks through to last dilution."), html.Br(),
        html.Span(f"3. Stamp to assay plate: {sv // cf}\u00b5L stock + {sv // cf * (cf - 1)}\u00b5L medium (1:{cf})."),
    ], style={"background": C["wb"], "border": f"1px solid {C['wbd']}", "borderRadius": "8px",
              "padding": "14px 20px", "fontSize": "13px", "lineHeight": "1.8", "color": C["wt"]})
    # Default to Block A-B
    return sub, dd, "BLK:0", prot, prep, gd, conc_rows, concs

# ── Tab 2: Plate viz ──
@callback(Output("plate-fig", "figure"), Output("plate-leg", "children"),
          Input("plate-tabs", "value"), Input("grid-store", "data"), State("ab-store", "data"))
def render_plate(tab, gd, ab_list):
    if not gd: return go.Figure(), []
    pn = 1 if tab == "p1" else 2; wells = [w for w in gd if w["plate"] == pn]
    pabs = [a for a in (ab_list or []) if a["plate"] == pn]; filled = {(w["row"], w["col"]) for w in wells}
    fig = go.Figure()
    for r in range(16):
        for c in range(1, 25):
            if (r, c) not in filled:
                fig.add_trace(go.Scatter(x=[c], y=[15 - r], mode="markers",
                    marker=dict(size=18, color="#f1f5f9", line=dict(color="#e2e8f0", width=1)),
                    hovertext=f"<b>{ROW_LABELS[r]}{c}</b><br>Empty", hoverinfo="text", showlegend=False))
    drawn = set()
    for w in sorted(wells, key=lambda x: (x["row"], x["col"])):
        r, c, bg = w["row"], w["col"], w["is_bg"]; base = w["color"]
        a = 0.12 if bg else max(0.15, 1.0 - r / 16)
        wc, bc = hex_rgba(base, a), hex_rgba(base, 0.3 if bg else min(1, a + 0.2))
        txt = "Bkg" if bg else fmt_nM(w["work_nM"])
        tc = C["mt"] if bg else ("white" if a > 0.45 else C["tx"])
        show = w["ab_name"] not in drawn; drawn.add(w["ab_name"])
        fig.add_trace(go.Scatter(x=[c], y=[15 - r], mode="markers+text",
            marker=dict(size=18, color=wc, line=dict(color=bc, width=1)),
            text=[txt], textposition="middle center", textfont=dict(size=5, family=MF, color=tc),
            hovertext=f"<b>{ROW_LABELS[r]}{c}</b><br>{w['ab_name']}<br>{w['work_nM']:.4g} nM" if not bg
                      else f"<b>{ROW_LABELS[r]}{c}</b><br>Bkg",
            hoverinfo="text", hoverlabel=dict(bgcolor="#0f172a", font=dict(size=12, color="white")),
            name=w["ab_name"], showlegend=show, legendgroup=w["ab_name"]))
    fig.update_layout(
        xaxis=dict(range=[0.2, 24.8], tickvals=list(range(1, 25)), ticktext=[str(i) for i in range(1, 25)],
                   side="top", fixedrange=True, tickfont=dict(family=MF, size=9, color="#64748b"), showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.8, 15.8], tickvals=list(range(16)), ticktext=list(reversed(ROW_LABELS)),
                   fixedrange=True, tickfont=dict(family=MF, size=9, color="#64748b"), showgrid=False, zeroline=False),
        margin=dict(l=40, r=20, t=40, b=10), plot_bgcolor="white", paper_bgcolor="white", hovermode="closest", height=560,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, font=dict(size=9)),
        shapes=[dict(type="rect", x0=0.3, x1=24.7, y0=-0.7, y1=15.7, line=dict(color="#e2e8f0", width=1))])
    for i in range(1, 8):
        fig.add_shape(type="line", x0=0.3, x1=24.7, y0=15 - i * 2 + 0.5, y1=15 - i * 2 + 0.5,
                      line=dict(color="#cbd5e1", width=0.5, dash="dot"))
    leg = [html.Div([
        html.Div(style={"width": "40px", "height": "12px", "borderRadius": "3px",
            "background": f"linear-gradient(90deg,{a['color']},{hex_rgba(a['color'], 0.15)})", "display": "inline-block"}),
        html.Span(f"C{a['col_start']}-{a['col_start']+1}: {a['name']}", style={"fontSize": "10px", "fontWeight": "500"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "4px"}) for a in pabs]
    return fig, leg

# ── Tab 2: Prep filter ──
@callback(Output("prep-grid", "rowData"), Input("prep-filter", "value"), State("prep-store", "data"), prevent_initial_call=True)
def filt_prep(sel, data):
    if not data or not sel: return no_update
    if sel.startswith("BLK:"):
        idx = int(sel[4:]); bl = f"{ROW_LABELS[idx * 2]}-{ROW_LABELS[idx * 2 + 1]}"
        return [r for r in data if r["Block"] == bl]
    if sel.startswith("AB:"): return [r for r in data if r["Antibody"] == sel[3:]]
    return no_update

# ── Tab 3: Show/hide cell line sections based on dropdown ──
@callback(
    *[Output(f"cc-section-{i}", "style") for i in range(5)],
    Output("cc-print-section", "style"),
    Input("cc-num-lines", "value"))
def toggle_cc_sections(n):
    n = n or 1
    styles = []
    for i in range(5):
        styles.append({} if i < n else {"display": "none"})
    print_style = {} if n >= 2 else {"display": "none"}
    return *styles, print_style

# ── Tab 3: Cancer Cell Seeding — shared computation ──
def _cc_compute(stock, cpw, wv, wells, dead):
    stock=stock or 1e6; cpw=cpw or 5000; wv=wv or 40; wells=wells or 500; dead=dead or 5000
    req = cpw / (wv / 1000)
    minv = wells * wv; prep = minv + dead
    add_s = req * prep / stock; add_m = prep - add_s
    tot = cpw * wells; dil = stock / req if req > 0 else 0
    return dict(req_conc=req, prep_vol=prep, add_stock=add_s, add_medium=add_m,
                total_cells=tot, dilution=dil, min_vol=minv,
                stock=stock, cpw=cpw, wv=wv, wells=wells, dead=dead)

def _cc_html(d):
    sS = {"fontSize":"11px","fontWeight":"700","color":C["ok"],"letterSpacing":"1px","marginBottom":"4px","textTransform":"uppercase"}
    sN = {"fontSize":"18px","fontWeight":"600","fontFamily":MF,"color":C["ac"]}
    sL = {"fontSize":"12px","color":C["mt"],"marginBottom":"2px"}
    sF = {"fontSize":"11px","color":C["mt"],"fontFamily":MF}
    sR = {"display":"flex","gap":"16px","alignItems":"flex-end","padding":"12px 0"}
    sB = {"borderBottom":f"1px solid {C['bd']}"}
    warns = []
    if d["add_stock"] > d["prep_vol"]: warns.append(f"Stock density too low! Need {d['add_stock']:,.0f} \u00b5L stock.")
    if d["req_conc"] > d["stock"]: warns.append(f"Stock ({d['stock']:,.0f}/mL) < required ({d['req_conc']:,.0f}/mL).")
    r = html.Div([
        html.Div([
            html.Div([html.Div("Step 1 \u2014 Required Concentration",style=sS),
                      html.Div("Cell concentration in working suspension",style={"fontSize":"13px","color":C["tx"]})],style={"flex":"2"}),
            html.Div([html.Div("Required Conc",style=sL), html.Div(f"{d['req_conc']:,.0f} Cells/mL",style=sN),
                      html.Div(f"= {d['cpw']:,} Cells \u00f7 {d['wv']} \u00b5L",style=sF)],style={"flex":"1.5","textAlign":"right"}),
        ],style={**sR,**sB}),
        html.Div([
            html.Div([html.Div("Step 2 \u2014 Total Working Volume",style=sS),
                      html.Div("Volume to prepare including dead volume",style={"fontSize":"13px","color":C["tx"]})],style={"flex":"2"}),
            html.Div([html.Div("Working Volume",style=sL), html.Div(f"{d['prep_vol']:,.0f} \u00b5L",style=sN),
                      html.Div(f"= ({d['wells']:,} \u00d7 {d['wv']}) + {d['dead']:,.0f} dead",style=sF)],style={"flex":"1.5","textAlign":"right"}),
        ],style={**sR,**sB}),
        html.Div([
            html.Div([html.Div("Step 3 \u2014 Mix Stock Cells + Medium",style=sS),
                      html.Div(f"Dilute {d['dilution']:,.1f}\u00d7 to reach {d['req_conc']:,.0f} Cells/mL",style={"fontSize":"13px","color":C["tx"]})],style={"flex":"1.2"}),
            html.Div([html.Div([
                html.Div([html.Div("Add Stock",style=sL), html.Div(f"{d['add_stock']:,.1f} \u00b5L",style={**sN,"color":"#3b82f6"})],style={"textAlign":"center","flex":"1"}),
                html.Div("+",style={"fontSize":"20px","color":C["mt"],"padding":"0 4px","alignSelf":"center"}),
                html.Div([html.Div("Add Medium",style=sL), html.Div(f"{max(d['add_medium'],0):,.1f} \u00b5L",style={**sN,"color":"#f59e0b"})],style={"textAlign":"center","flex":"1"}),
                html.Div("=",style={"fontSize":"20px","color":C["mt"],"padding":"0 4px","alignSelf":"center"}),
                html.Div([html.Div("Total",style=sL), html.Div(f"{d['prep_vol']:,.0f} \u00b5L",style={**sN,"color":C["tx"]})],style={"textAlign":"center","flex":"1"}),
            ],style={"display":"flex","alignItems":"flex-end"})],style={"flex":"2"}),
        ],style={**sR,**sB}),
        html.Div([
            html.Div([html.Div("Step 4 \u2014 Dispense",style=sS),
                      html.Div(f"Pipette {d['wv']} \u00b5L/well into {d['wells']:,} wells",style={"fontSize":"13px","color":C["tx"]})],style={"flex":"2"}),
            html.Div([html.Div("Total Cells Seeded",style=sL), html.Div(f"{d['total_cells']:,.0f} Cells",style=sN)],style={"flex":"1.5","textAlign":"right"}),
        ],style=sR),
        *[html.Div(w,style={"background":"#fef2f2","color":"#991b1b","padding":"10px 14px",
            "borderRadius":"6px","fontSize":"12px","fontWeight":"600","marginTop":"10px"}) for w in warns],
    ])
    cr = {"flex":"1","textAlign":"center","padding":"10px","borderRadius":"8px"}
    s = html.Div([html.Div([
        html.Div([html.Div("Stock",style={"fontSize":"10px","color":C["mt"]}),
                  html.Div(f"{d['stock']:,.0f}/mL",style={"fontSize":"13px","fontWeight":"600","fontFamily":MF})],style={**cr,"background":C["sa"]}),
        html.Div([html.Div("Required",style={"fontSize":"10px","color":C["mt"]}),
                  html.Div(f"{d['req_conc']:,.0f}/mL",style={"fontSize":"13px","fontWeight":"600","fontFamily":MF})],style={**cr,"background":C["sa"]}),
        html.Div([html.Div("Dilution",style={"fontSize":"10px","color":C["mt"]}),
                  html.Div(f"{d['dilution']:,.1f}\u00d7",style={"fontSize":"13px","fontWeight":"600","fontFamily":MF})],style={**cr,"background":C["sa"]}),
        html.Div([html.Div("Add Stock",style={"fontSize":"10px","color":C["mt"]}),
                  html.Div(f"{d['add_stock']:,.1f} \u00b5L",style={"fontSize":"13px","fontWeight":"600","fontFamily":MF,"color":"#3b82f6"})],style={**cr,"background":C["al"]}),
        html.Div([html.Div("Add Medium",style={"fontSize":"10px","color":C["mt"]}),
                  html.Div(f"{max(d['add_medium'],0):,.1f} \u00b5L",style={"fontSize":"13px","fontWeight":"600","fontFamily":MF,"color":"#f59e0b"})],style={**cr,"background":C["sa"]}),
    ],style={"display":"flex","gap":"8px"})],style={**card,"padding":"12px"})
    return r, s

def _mk_cc(idx):
    @callback(Output(f"cc-result-{idx}","children"), Output(f"cc-summary-{idx}","children"),
              Output("cc-all-store","data",allow_duplicate=True),
              Input(f"cc-stock-{idx}","value"), Input(f"cc-cpw-{idx}","value"),
              Input(f"cc-wv-{idx}","value"), Input(f"cc-wells-{idx}","value"), Input(f"cc-dead-{idx}","value"),
              State(f"cc-name-{idx}","value"), State("cc-all-store","data"), prevent_initial_call=True)
    def _calc(stock,cpw,wv,wells,dead,name,all_d):
        d = _cc_compute(stock,cpw,wv,wells,dead)
        r,s = _cc_html(d)
        all_d = all_d or {}
        all_d[str(idx)] = {**d, "name": name or f"Cell Line {idx+1}"}
        return r, s, all_d
for _i in range(5): _mk_cc(_i)

# ── Print Summary ──
@callback(Output("cc-print-content","children"), Input("cc-all-store","data"),
          State("cc-num-lines","value"))
def cc_print(all_d, n):
    if not all_d: return no_update
    n = n or 1
    hs = {"padding":"8px 12px","background":C["sa"],"borderBottom":f"2px solid {C['bd']}",
          "fontSize":"12px","fontWeight":"600","color":C["mt"],"textAlign":"left"}
    header = html.Tr([html.Th(h,style=hs) for h in
        ["Cell Line","Stock (Cells/mL)","Cells/Well","\u00b5L/Well","Wells",
         "Req'd Conc (Cells/mL)","Dilution","Add Stock (\u00b5L)","Add Medium (\u00b5L)",
         "Total Vol (\u00b5L)","Total Cells"]])
    rows = []
    cs = {"padding":"8px 12px","fontSize":"12px","fontFamily":MF,"borderBottom":f"1px solid {C['bd']}"}
    for k in sorted(all_d.keys()):
        i = int(k)
        if i >= n: continue
        d = all_d[k]; nm = d.get("name", f"Cell Line {i+1}")
        rows.append(html.Tr([
            html.Td(nm,style={**cs,"fontWeight":"600","fontFamily":FF,"borderLeft":f"4px solid {AB_COLORS[i]}"}),
            html.Td(f"{d['stock']:,.0f}",style=cs), html.Td(f"{d['cpw']:,}",style=cs),
            html.Td(f"{d['wv']}",style=cs), html.Td(f"{d['wells']:,}",style=cs),
            html.Td(f"{d['req_conc']:,.0f}",style={**cs,"fontWeight":"600"}),
            html.Td(f"{d['dilution']:,.1f}\u00d7",style=cs),
            html.Td(f"{d['add_stock']:,.1f}",style={**cs,"fontWeight":"700","color":"#3b82f6"}),
            html.Td(f"{max(d['add_medium'],0):,.1f}",style={**cs,"color":"#f59e0b"}),
            html.Td(f"{d['prep_vol']:,.0f}",style=cs),
            html.Td(f"{d['total_cells']:,.0f}",style=cs),
        ]))
    return html.Div([
        html.Div([html.Div("Cancer Cell Preparation \u2014 All Cell Lines",
                            style={"fontSize":"16px","fontWeight":"600","marginBottom":"4px"}),
                  html.Div("Print or export this summary for your lab notebook",
                            style={"fontSize":"12px","color":C["mt"]})],
                 style={"marginBottom":"16px","paddingBottom":"12px","borderBottom":f"2px solid {C['tx']}"}),
        html.Table([html.Thead(header),html.Tbody(rows)],
                   style={"width":"100%","borderCollapse":"collapse","marginBottom":"16px"}),
        html.Div("Units: Density in Cells/mL | Volumes in \u00b5L",
                 style={"fontSize":"11px","color":C["mt"]}),
    ])

# ── Print button ──
app.clientside_callback("function(){window.print();return window.dash_clientside.no_update;}",
    Output("cc-print-btn","n_clicks"), Input("cc-print-btn","n_clicks"), prevent_initial_call=True)

# ── Export Cell Calc xlsx ──
@callback(Output("dl-cellcalc","data"), Input("cc-export-btn","n_clicks"),
          State("cc-all-store","data"), State("cc-num-lines","value"), prevent_initial_call=True)
def cc_export(nc, all_d, n):
    if not all_d: return no_update
    n = n or 1; rows = []
    for k in sorted(all_d.keys()):
        i = int(k)
        if i >= n: continue
        d = all_d[k]; nm = d.get("name", f"Cell Line {i+1}")
        rows.append({"Cell Line":nm, "Stock (Cells/mL)":d["stock"], "Cells/Well":d["cpw"],
            "\u00b5L/Well":d["wv"], "Total Wells":d["wells"],
            "Req'd Conc (Cells/mL)":round(d["req_conc"]), "Dilution":round(d["dilution"],1),
            "Add Stock (\u00b5L)":round(d["add_stock"],1), "Add Medium (\u00b5L)":round(max(d["add_medium"],0),1),
            "Total Vol (\u00b5L)":round(d["prep_vol"]), "Dead Vol (\u00b5L)":d["dead"],
            "Total Cells":d["total_cells"]})
    return dcc.send_data_frame(pd.DataFrame(rows).to_excel,
        f"cell_prep_{datetime.now():%Y%m%d_%H%M%S}.xlsx", index=False)

# ── Tab 4: Parse pasted plate reader grid ──
def _parse_plate_grid(text):
    """Parse pasted 16×24 grid text → 2D array of floats (16 rows × 24 cols)."""
    lines = [l for l in text.strip().split("\n") if l.strip()]
    grid = []
    for line in lines:
        cells = line.replace(",", "").split("\t") if "\t" in line else line.split()
        nums = []
        for c in cells:
            c = c.strip()
            if not c: continue
            # Skip row labels (A-P)
            if len(c) == 1 and c.upper() in "ABCDEFGHIJKLMNOP": continue
            try:
                nums.append(float(c))
            except ValueError:
                continue
        if nums:
            grid.append(nums)
    return grid

@callback(
    Output("data-status", "children", allow_duplicate=True),
    Output("mean-grid", "columnDefs", allow_duplicate=True), Output("mean-grid", "rowData", allow_duplicate=True),
    Output("norm-grid", "columnDefs", allow_duplicate=True), Output("norm-grid", "rowData", allow_duplicate=True),
    Output("dose-fig", "figure", allow_duplicate=True), Output("ec50-grid", "rowData", allow_duplicate=True),
    Output("analysis-store", "data", allow_duplicate=True),
    Output("raw-grid", "rowData", allow_duplicate=True), Output("raw-filter", "options", allow_duplicate=True),
    Output("sim-store", "data", allow_duplicate=True),
    Output("paste-status", "children"),
    Input("parse-plate-btn", "n_clicks"),
    State("paste-grid", "value"), State("paste-plate", "value"),
    State("grid-store", "data"), State("ab-store", "data"), State("conc-store", "data"),
    prevent_initial_call=True)
def parse_plate(n, text, plate_num, grid_data, ab_list, concs):
    _n = no_update
    err = lambda msg: (html.Span(msg, style={"color": C["dg"]}), _n, _n, _n, _n, _n, _n, _n, _n, _n, _n,
                       html.Span(msg, style={"color": C["dg"]}))
    if not text or not text.strip():
        return err("Please paste plate reader data first.")
    if not grid_data:
        return err("Generate plate layout in \U0001F9EA Dilution Block Preparation tab first.")

    plate_num = plate_num or 1
    grid = _parse_plate_grid(text)
    if len(grid) < 2:
        return err(f"Could not parse grid. Found {len(grid)} rows. Expected 16 rows of data.")
    n_rows = len(grid)
    n_cols = max(len(r) for r in grid) if grid else 0
    # Pad short rows
    for r in grid:
        while len(r) < n_cols: r.append(0)

    # Build well→(ab_name, conc, color, ctrl) mapping from grid_store for selected plate
    well_map = {}
    for w in grid_data:
        if w["plate"] == plate_num:
            well_map[(w["row"], w["col"])] = w

    if not well_map:
        return err(f"No layout data for Plate {plate_num}. Generate layout first.")

    # Collect RLU values grouped by (ab_name, conc)
    from collections import defaultdict
    grouped = defaultdict(list)  # (ab_name, conc) → [values]
    ab_info = {}  # ab_name → {color, ctrl}
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
        return err(f"No wells matched. Grid: {n_rows}\u00d7{n_cols}, layout wells: {len(well_map)}.")

    # Build ordered lists
    ab_names = list(dict.fromkeys(w["ab_name"] for w in sorted(well_map.values(), key=lambda x: x["col"])))
    conc_list = sorted(set(w["work_nM"] for w in well_map.values()), reverse=True)

    # Build raw dict for process_rlu_data: (protein_idx, conc_idx) → [reps]
    raw_rlu = {}
    for pi, ab in enumerate(ab_names):
        for ci, c in enumerate(conc_list):
            reps = grouped.get((ab, c), [])
            if reps:
                raw_rlu[(pi, ci)] = reps

    if not raw_rlu:
        return err("Could not map pasted values to plate layout.")

    # Use RLU processing
    mc, mr, nc, nr, fig, ec = process_rlu_data(raw_rlu, conc_list, ab_names)

    # Also build viability-style raw table for the raw grid
    sim_data = []
    for ab in ab_names:
        info = ab_info.get(ab, {"ctrl": "Positive"})
        for c in conc_list:
            reps = grouped.get((ab, c), [])
            for ri, v in enumerate(reps):
                sim_data.append({"Antibody": ab, "Conc_nM": c, "Replicate": ri + 1,
                                 "Viability_%": v, "Control": info["ctrl"]})
    raw_tbl = build_raw_table(sim_data) if sim_data else []
    fopts = [{"label": a, "value": a} for a in ab_names]

    st = html.Span(["\u2705 ", html.Strong(f"Plate {plate_num}: {len(ab_names)} antibodies", style={"color": C["ok"]}),
                     f" \u00d7 {len(conc_list)} concs | {matched} wells parsed from {n_rows}\u00d7{n_cols} grid"])
    ps = html.Span(["\u2705 ", html.Strong(f"{matched} wells mapped", style={"color": C["ok"]}),
                     f" to {len(ab_names)} antibodies"])
    return st, mc, mr, nc, nr, fig, ec, ec, raw_tbl, fopts, sim_data, ps

# ── Tab 4: Simulate RLU ──
@callback(Output("data-status", "children"), Output("mean-grid", "columnDefs"), Output("mean-grid", "rowData"),
          Output("norm-grid", "columnDefs"), Output("norm-grid", "rowData"),
          Output("dose-fig", "figure"), Output("ec50-grid", "rowData"), Output("analysis-store", "data"),
          Input("sim-rlu-btn", "n_clicks"), State("conc-store", "data"), prevent_initial_call=True)
def sim_rlu(n, concs):
    _n = no_update
    if not concs:
        return html.Span("Generate Dilution Series first.", style={"color": C["dg"]}), _n, _n, _n, _n, _n, _n, _n
    names = [f"Protein {chr(65 + i)}" for i in range(MAX_PROTEINS)]
    raw = sim_plate_data(concs, MAX_PROTEINS)
    mc, mr, nc, nr, fig, ec = process_rlu_data(raw, concs, names)
    st = html.Span(["\u2705 ", html.Strong(f"Simulated {MAX_PROTEINS} proteins", style={"color": C["ok"]}),
                     f" \u00d7 {len(concs)} concs \u00d7 4 reps"])
    return st, mc, mr, nc, nr, fig, ec, ec

# ── Tab 4: Simulate Viability (uses ab-store from Tab 2) ──
@callback(Output("data-status", "children", allow_duplicate=True),
          Output("mean-grid", "columnDefs", allow_duplicate=True), Output("mean-grid", "rowData", allow_duplicate=True),
          Output("norm-grid", "columnDefs", allow_duplicate=True), Output("norm-grid", "rowData", allow_duplicate=True),
          Output("dose-fig", "figure", allow_duplicate=True), Output("ec50-grid", "rowData", allow_duplicate=True),
          Output("analysis-store", "data", allow_duplicate=True),
          Output("raw-grid", "rowData"), Output("raw-filter", "options"), Output("sim-store", "data"),
          Input("sim-via-btn", "n_clicks"), State("ab-store", "data"), State("conc-store", "data"),
          prevent_initial_call=True)
def sim_via(n, ab_list, concs):
    _n = no_update
    if not ab_list:
        return html.Span("Load antibodies in Reagent Prep tab first.", style={"color": C["dg"]}), _n, _n, _n, _n, _n, _n, _n, _n, _n, _n
    if not concs: concs = [10000 / (4 ** i) for i in range(7)] + [0]
    sim = sim_ab_response(ab_list, concs)
    raw_tbl = build_raw_table(sim)
    analysis = run_4pl_analysis(ab_list, sim, concs)
    fig = make_dose_fig(ab_list, analysis)
    fopts = [{"label": a["name"], "value": a["name"]} for a in ab_list]
    ec_tbl = [{k: v for k, v in a.items() if not k.startswith("_")} for a in analysis]
    st = html.Span(["\u2705 ", html.Strong(f"{len(ab_list)} antibodies", style={"color": C["ok"]}),
                     f" \u00d7 {len(concs)} concs \u00d7 4 reps = {len(sim)} points"])
    # Empty mean/norm grids for this mode (data is in raw-grid instead)
    return st, [], [], [], [], fig, ec_tbl, analysis, raw_tbl, fopts, sim

# ── Tab 4: Upload data ──
@callback(Output("data-status", "children", allow_duplicate=True),
          Output("mean-grid", "columnDefs", allow_duplicate=True), Output("mean-grid", "rowData", allow_duplicate=True),
          Output("norm-grid", "columnDefs", allow_duplicate=True), Output("norm-grid", "rowData", allow_duplicate=True),
          Output("dose-fig", "figure", allow_duplicate=True), Output("ec50-grid", "rowData", allow_duplicate=True),
          Output("analysis-store", "data", allow_duplicate=True),
          Output("raw-grid", "rowData", allow_duplicate=True), Output("raw-filter", "options", allow_duplicate=True),
          Output("sim-store", "data", allow_duplicate=True),
          Input("data-upload", "contents"), State("data-upload", "filename"), prevent_initial_call=True)
def upload_data(contents, fn):
    _n = no_update
    if not contents: return _n, _n, _n, _n, _n, _n, _n, _n, _n, _n, _n
    _, cs = contents.split(","); dec = base64.b64decode(cs)
    try:
        df = pd.read_csv(io.StringIO(dec.decode("utf-8"))) if fn.endswith(".csv") else pd.read_excel(io.BytesIO(dec))
        def fc(cols, cands):
            for ca in cands:
                for dc in cols:
                    if ca.lower().replace(" ", "") in dc.lower().replace(" ", ""): return dc
            return None
        cp = fc(df.columns, ["Protein", "Antibody", "AB_ID", "name", "sample"])
        cc = fc(df.columns, ["Concentration", "conc", "Conc_nM", "nM"])
        rcs = sorted([c for c in df.columns if "rep" in c.lower() or c.startswith("Rep")])
        cct = fc(df.columns, ["Control", "control_type", "type", "ctrl"])
        if not cp or not cc or len(rcs) < 2:
            return html.Span(f"Missing columns. Found: {list(df.columns)}", style={"color": C["dg"]}), _n, _n, _n, _n, _n, _n, _n, _n, _n, _n
        # Detect if RLU or viability data (RLU values typically > 1000)
        sample_vals = df[rcs[0]].dropna().head(20)
        is_rlu = sample_vals.mean() > 500
        names = list(dict.fromkeys(df[cp].astype(str)))
        concs = sorted(df[cc].unique(), reverse=True)
        if is_rlu:
            raw = {}
            for pi, nm in enumerate(names):
                for ci, c in enumerate(concs):
                    mask = (df[cp].astype(str) == nm) & (df[cc] == c)
                    sub = df.loc[mask, rcs]
                    if not sub.empty:
                        raw[(pi, ci)] = [float(v) for v in sub.iloc[0].values if pd.notna(v)]
            mc, mr, nc, nr, fig, ec = process_rlu_data(raw, list(concs), names[:MAX_PROTEINS])
            st = html.Span(["\u2705 ", html.Strong(f"{len(names)} proteins (RLU)", style={"color": C["ok"]}), f" from {fn}"])
            return st, mc, mr, nc, nr, fig, ec, ec, [], [], []
        else:
            # Viability data
            sim_data = []
            for _, row in df.iterrows():
                nm = str(row[cp]); c = float(row[cc])
                ct = str(row[cct]).strip() if cct and pd.notna(row.get(cct)) else "Positive"
                for ri, rc in enumerate(rcs):
                    if pd.notna(row[rc]):
                        sim_data.append({"Antibody": nm, "Conc_nM": c, "Replicate": ri + 1,
                                         "Viability_%": float(row[rc]), "Control": ct})
            ab_list = [{"name": nm, "color": AB_COLORS[i % len(AB_COLORS)],
                        "ctrl_type": next((r["Control"] for r in sim_data if r["Antibody"] == nm), "Positive")}
                       for i, nm in enumerate(names)]
            raw_tbl = build_raw_table(sim_data)
            analysis = run_4pl_analysis(ab_list, sim_data, list(concs))
            fig = make_dose_fig(ab_list, analysis)
            fopts = [{"label": a["name"], "value": a["name"]} for a in ab_list]
            ec_tbl = [{k: v for k, v in a.items() if not k.startswith("_")} for a in analysis]
            st = html.Span(["\u2705 ", html.Strong(f"{len(ab_list)} antibodies (viability)", style={"color": C["ok"]}), f" from {fn}"])
            return st, [], [], [], [], fig, ec_tbl, analysis, raw_tbl, fopts, sim_data
    except Exception as e:
        return html.Span(f"Error: {e}", style={"color": C["dg"]}), _n, _n, _n, _n, _n, _n, _n, _n, _n, _n

# ── Tab 4: Raw filter ──
@callback(Output("raw-grid", "rowData", allow_duplicate=True), Input("raw-filter", "value"),
          State("sim-store", "data"), prevent_initial_call=True)
def filt_raw(sel, sim):
    if not sim: return no_update
    return build_raw_table(sim, sel if sel else None)

# ── Exports ──
@callback(Output("dl-prep", "data"), Input("exp-prep-btn", "n_clicks"), State("prep-store", "data"), prevent_initial_call=True)
def ep(n, d):
    return dcc.send_data_frame(pd.DataFrame(d).to_excel, f"prep_{datetime.now():%Y%m%d_%H%M%S}.xlsx", index=False) if d else no_update
@callback(Output("dl-mean", "data"), Input("exp-mean-btn", "n_clicks"), State("mean-grid", "rowData"), prevent_initial_call=True)
def em(n, d):
    return dcc.send_data_frame(pd.DataFrame(d).to_excel, f"mean_{datetime.now():%Y%m%d_%H%M%S}.xlsx", index=False) if d else no_update
@callback(Output("dl-norm", "data"), Input("exp-norm-btn", "n_clicks"), State("norm-grid", "rowData"), prevent_initial_call=True)
def en(n, d):
    return dcc.send_data_frame(pd.DataFrame(d).to_excel, f"norm_{datetime.now():%Y%m%d_%H%M%S}.xlsx", index=False) if d else no_update
@callback(Output("dl-ec50", "data"), Input("exp-ec50-btn", "n_clicks"), State("ec50-grid", "rowData"), prevent_initial_call=True)
def ee(n, d):
    return dcc.send_data_frame(pd.DataFrame(d).to_excel, f"ec50_{datetime.now():%Y%m%d_%H%M%S}.xlsx", index=False) if d else no_update
@callback(Output("dl-raw", "data"), Input("exp-raw-btn", "n_clicks"), State("sim-store", "data"), prevent_initial_call=True)
def er(n, d):
    return dcc.send_data_frame(pd.DataFrame(d).to_excel, f"raw_{datetime.now():%Y%m%d_%H%M%S}.xlsx", index=False) if d else no_update

@callback(Output("dl-all", "data"), Input("exp-all-btn", "n_clicks"),
          State("mean-grid", "rowData"), State("norm-grid", "rowData"),
          State("ec50-grid", "rowData"), State("sim-store", "data"),
          State("prep-store", "data"), prevent_initial_call=True)
def exp_all(n, mean_d, norm_d, ec50_d, raw_d, prep_d):
    if not any([mean_d, norm_d, ec50_d, raw_d]): return no_update
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        if ec50_d: pd.DataFrame(ec50_d).to_excel(w, sheet_name="EC50 Summary", index=False)
        if mean_d: pd.DataFrame(mean_d).to_excel(w, sheet_name="Mean & SD", index=False)
        if norm_d: pd.DataFrame(norm_d).to_excel(w, sheet_name="Normalized %", index=False)
        if raw_d: pd.DataFrame(raw_d).to_excel(w, sheet_name="Raw Data", index=False)
        if prep_d: pd.DataFrame(prep_d).to_excel(w, sheet_name="Stock Prep", index=False)
    buf.seek(0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dcc.send_bytes(buf.getvalue(), f"full_analysis_{ts}.xlsx")

if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  Potency Assay \u2014 384-Well (Unified)")
    print("  http://127.0.0.1:8050")
    print("=" * 56 + "\n")
    app.run(debug=True, port=8050)