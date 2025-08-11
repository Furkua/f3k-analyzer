# f3k_dashboard_plus.py
# Streamlit dashboard for F3K glider session analysis (UPGRADED)
# New features:
# - ZIP upload (auto-merge many CSVs)
# - Session metadata (tags, wind) with export/import
# - Per-session summary (throws, best/avg/median max height, total airtime, avg flight duration)
# - Per-throw extra metrics: climb rate near launch (max vario in first 2s), climb-to-peak average
# - Filters: min peak altitude, min/max duration
# - Histogram of throw max heights
# - Download combined (throws + metadata) CSV
# - Optional PNG export of current chart (requires kaleido)

import io
import json
import zipfile
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="F3K Session Analyzer — Plus", layout="wide")

# ---------------------------- Utilities ---------------------------- #

ALT_KEYS = ["Alt(m)", "Alt", "Altitude", "BaroAlt", "Height", "Alti"]
SA_KEYS  = ["SA", "Acc", "Accel", "AccMag", "A", "Ax", "Ay", "Az"]
TIME_KEYS = ["Time", "Timestamp", "Datetime", "UTC", "T"]

def find_column(df, keys):
    lower = {c.lower(): c for c in df.columns}
    for k in keys:
        for c in df.columns:
            if c.lower() == k.lower():
                return c
        for lc, orig in lower.items():
            if k.lower() in lc:
                return orig
    return None

def parse_time_seconds(s):
    try:
        t = pd.to_datetime(s, format="%H:%M:%S.%f", errors="coerce")
    except Exception:
        t = pd.to_datetime(s, errors="coerce")
    if t.isna().all():
        try:
            v = pd.to_numeric(s, errors="coerce")
            v = v - np.nanmin(v)
            return v.fillna(0).astype(float).values
        except Exception:
            return np.arange(len(s), dtype=float)
    return (t - t.iloc[0]).dt.total_seconds().fillna(0).values

def detect_first_start(alt, sa, t):
    alt = pd.Series(pd.to_numeric(alt, errors="coerce")).interpolate(limit_direction="both").values
    sa  = pd.Series(pd.to_numeric(sa,  errors="coerce")).interpolate(limit_direction="both").values
    dt = np.diff(t)
    dt_med = np.median(dt[dt > 0]) if len(dt) else 0.02
    if not np.isfinite(dt_med) or dt_med <= 0:
        dt_med = 0.02
    vs = np.gradient(alt, dt_med)
    n = len(alt)
    base_n = int(max(50, min(0.05 * n, 2000)))
    base_vs = vs[:base_n]
    base_sa = sa[:base_n]
    vs_thr = np.mean(base_vs) + 3*np.std(base_vs) if np.std(base_vs) > 0 else np.mean(base_vs) + 0.3
    sa_thr = np.mean(base_sa) + 3*np.std(base_sa) if np.std(base_sa) > 0 else np.mean(base_sa) + 0.3
    idx_vs = np.argmax(vs > vs_thr)
    idx_sa = np.argmax(sa > sa_thr)
    cands = []
    if idx_vs > 0 and vs[idx_vs] > vs_thr: cands.append(int(idx_vs))
    if idx_sa > 0 and sa[idx_sa] > sa_thr: cands.append(int(idx_sa))
    start_idx = max(0, min(cands) - 5) if cands else 0
    return int(start_idx), float(dt_med), vs

def detect_throws(alt, sa, t, ground_alt=2.0, min_gap_s=5.0, min_flight_s=4.0):
    alt = pd.Series(pd.to_numeric(alt, errors="coerce")).interpolate(limit_direction="both").values
    sa  = pd.Series(pd.to_numeric(sa,  errors="coerce")).interpolate(limit_direction="both").values
    t = np.asarray(t, dtype=float)
    alt_s = pd.Series(alt).rolling(5, center=True, min_periods=1).median().values

    dt = np.diff(t)
    dt_med = np.median(dt[dt > 0]) if len(dt) else 0.02
    if not np.isfinite(dt_med) or dt_med <= 0:
        dt_med = 0.02
    vs = np.gradient(alt_s, dt_med)  # m/s approximate

    ground_mask = alt_s <= (ground_alt + 1.0)
    if ground_mask.any():
        base_sa = sa[ground_mask]
        base_vs = vs[ground_mask]
    else:
        base_sa = sa[: max(100, int(0.1*len(sa)))]
        base_vs = vs[: max(100, int(0.1*len(vs)))]
    sa_thr = float(np.nanmean(base_sa) + 3*np.nanstd(base_sa)) if np.nanstd(base_sa) > 0 else float(np.nanmean(base_sa) + 0.5)
    vs_thr = float(np.nanmean(base_vs) + 3*np.nanstd(base_vs)) if np.nanstd(base_vs) > 0 else float(np.nanmean(base_vs) + 0.5)

    throws = []
    i = 0
    n = len(alt_s)
    while i < n:
        if (sa[i] > sa_thr or vs[i] > vs_thr) and alt_s[i] <= (ground_alt + 1.0):
            start_idx = i
            # Peak search up to 8s
            search_end = min(n-1, start_idx + int(8.0/dt_med))
            peak_idx = start_idx
            for j in range(start_idx, search_end):
                if alt_s[j] >= alt_s[peak_idx]:
                    peak_idx = j
            # End: return to ground up to 90s
            end_search_end = min(n-1, peak_idx + int(90.0/dt_med))
            end_idx = None
            for j in range(peak_idx, end_search_end):
                if alt_s[j] <= ground_alt:
                    stay = min(n-1, j + int(0.5/dt_med))
                    if alt_s[j:stay].max() <= (ground_alt + 0.5):
                        end_idx = j
                        break
            if end_idx is None:
                end_idx = end_search_end

            if t[end_idx] - t[start_idx] >= min_flight_s:
                # Extra metrics
                t_start = t[start_idx]
                t_peak  = t[peak_idx]
                # Max climb in first 2s after start
                win_end = min(n-1, start_idx + int(2.0/dt_med))
                max_climb_2s = float(np.nanmax(vs[start_idx:win_end+1])) if win_end > start_idx else float("nan")
                # Average climb to peak
                duration_to_peak = max(t_peak - t_start, 1e-6)
                avg_climb_to_peak = float( (alt_s[peak_idx] - alt_s[start_idx]) / duration_to_peak )
                throws.append({
                    "start_idx": start_idx,
                    "peak_idx": peak_idx,
                    "end_idx": end_idx,
                    "max_climb_2s": max_climb_2s,
                    "avg_climb_to_peak": avg_climb_to_peak
                })
                i = end_idx + int(min_gap_s/dt_med)
            else:
                i += int(1.0/dt_med)
        else:
            i += 1
    return throws, vs

def read_csv_bytes(name: str, content: bytes):
    df = pd.read_csv(io.BytesIO(content))
    alt_col = find_column(df, ALT_KEYS)
    sa_col = find_column(df, SA_KEYS)
    time_col = find_column(df, TIME_KEYS)
    if alt_col is None or sa_col is None or time_col is None:
        raise ValueError(f"{name}: Missing columns. Need Time + Alt + SA. Found: {list(df.columns)}")
    t = parse_time_seconds(df[time_col])
    alt = df[alt_col].values
    sa  = df[sa_col].values
    return {"name": name, "df": df, "t": t, "alt": alt, "sa": sa}

# ---------------------------- Sidebar: Uploads ---------------------------- #

st.sidebar.header("Upload sessions")
uploads = st.sidebar.file_uploader(
    "Drop multiple CSVs *or* a ZIP with many CSVs",
    type=["csv", "zip"],
    accept_multiple_files=True
)

st.sidebar.header("Detection settings")
sync_to_first_launch = st.sidebar.checkbox("Synchronize sessions to first launch", True)
ground_alt = st.sidebar.number_input("Ground altitude (m)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
min_gap_s = st.sidebar.number_input("Min gap between throws (s)", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
min_flight_s = st.sidebar.number_input("Minimum flight duration (s)", min_value=1.0, max_value=30.0, value=4.0, step=0.5)
show_peak_labels = st.sidebar.checkbox("Show peak labels on chart", True)

st.sidebar.header("Throw filters")
min_peak_alt = st.sidebar.number_input("Min peak altitude (m)", min_value=0.0, value=0.0, step=1.0)
min_duration = st.sidebar.number_input("Min duration (s)", min_value=0.0, value=0.0, step=0.5)
max_duration = st.sidebar.number_input("Max duration (s, 0=ignore)", min_value=0.0, value=0.0, step=0.5)

st.sidebar.header("Chart export")
enable_png_export = st.sidebar.checkbox("Enable PNG export (requires kaleido)", False)

# ---------------------------- Main ---------------------------- #

st.title("F3K Session Analyzer — Plus")

if not uploads:
    st.info("Upload CSV files or a ZIP in the sidebar to begin.")
    st.stop()

sessions_raw = []
errors = []

for f in uploads:
    name = getattr(f, "name", "file")
    data = f.read()
    if name.lower().endswith(".csv"):
        try:
            sessions_raw.append(read_csv_bytes(name, data))
        except Exception as e:
            errors.append(str(e))
    elif name.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for zname in zf.namelist():
                    if not zname.lower().endswith(".csv"):
                        continue
                    content = zf.read(zname)
                    try:
                        sessions_raw.append(read_csv_bytes(zname, content))
                    except Exception as e:
                        errors.append(str(e))
        except Exception as e:
            errors.append(f"{name}: {e}")
    else:
        errors.append(f"{name}: unsupported file type")

if errors:
    st.warning("Some files could not be parsed:\n\n" + "\n".join(errors))

if not sessions_raw:
    st.stop()

# Determine t0 across sessions
starts = []
for s in sessions_raw:
    start_idx, dt_med, vs = detect_first_start(s["alt"], s["sa"], s["t"])
    s["first_start_idx"] = start_idx
    s["dt_med"] = dt_med
    s["vs"] = vs
    starts.append(s["t"][start_idx])
t0_global = min(starts) if sync_to_first_launch else 0.0

# Session metadata store
if "meta" not in st.session_state:
    st.session_state.meta = {}  # name -> dict

# Build processed sessions and stats
processed = []
rows = []

for s in sessions_raw:
    name = s["name"]
    t_al = s["t"] - (s["t"][s["first_start_idx"]] if sync_to_first_launch else t0_global)
    alt = pd.Series(pd.to_numeric(s["alt"], errors="coerce")).interpolate(limit_direction="both").values
    sa  = pd.Series(pd.to_numeric(s["sa"],  errors="coerce")).interpolate(limit_direction="both").values
    throws, vs_series = detect_throws(alt, sa, t_al, ground_alt, min_gap_s, min_flight_s)

    # Ensure metadata exists
    meta = st.session_state.meta.get(name, {"tags": "", "wind_speed": "", "wind_dir": "", "notes": ""})
    st.session_state.meta[name] = meta

    for k, th in enumerate(throws, start=1):
        t_start = float(t_al[th["start_idx"]])
        t_peak  = float(t_al[th["peak_idx"]])
        t_end   = float(t_al[th["end_idx"]])
        max_alt = float(alt[th["peak_idx"]])
        dur = t_end - t_start
        rows.append({
            "session": name,
            "throw": k,
            "t_start_s": round(t_start, 2),
            "t_peak_s": round(t_peak, 2),
            "t_end_s": round(t_end, 2),
            "duration_s": round(dur, 2),
            "max_alt_m": round(max_alt, 2),
            "max_climb_2s_mps": round(th["max_climb_2s"], 2),
            "avg_climb_to_peak_mps": round(th["avg_climb_to_peak"], 2),
            "tags": meta.get("tags", ""),
            "wind_speed": meta.get("wind_speed", ""),
            "wind_dir": meta.get("wind_dir", ""),
            "notes": meta.get("notes", ""),
        })

    processed.append({"name": name, "t": t_al, "alt": alt, "throws": throws})

stats_df = pd.DataFrame(rows)

# Apply throw filters
if not stats_df.empty:
    filt = (stats_df["max_alt_m"] >= min_peak_alt) & (stats_df["duration_s"] >= min_duration)
    if max_duration > 0:
        filt &= stats_df["duration_s"] <= max_duration
    stats_df = stats_df[filt].reset_index(drop=True)

# Per-session summary
summary = []
for name in sorted({p["name"] for p in processed}):
    sdf = stats_df[stats_df["session"] == name]
    if sdf.empty:
        summary.append({"session": name, "throws": 0, "best_max_m": None, "avg_max_m": None, "median_max_m": None, "total_airtime_s": 0, "avg_duration_s": None})
        continue
    summary.append({
        "session": name,
        "throws": int(sdf["throw"].count()),
        "best_max_m": float(sdf["max_alt_m"].max()),
        "avg_max_m": float(sdf["max_alt_m"].mean()),
        "median_max_m": float(sdf["max_alt_m"].median()),
        "total_airtime_s": float(sdf["duration_s"].sum()),
        "avg_duration_s": float(sdf["duration_s"].mean()),
    })
summary_df = pd.DataFrame(summary)

# ---------------------------- Layout ---------------------------- #
top, = st.columns([1])
with top:
    st.subheader("Session metadata")
    # Editable metadata per session
    for name in sorted({p["name"] for p in processed}):
        with st.expander(name, expanded=False):
            m = st.session_state.meta[name]
            col1, col2, col3 = st.columns(3)
            m["tags"] = col1.text_input("Tags (comma-separated)", value=m.get("tags",""), key=f"tags_{name}")
            m["wind_speed"] = col2.text_input("Wind speed (m/s or km/h)", value=m.get("wind_speed",""), key=f"ws_{name}")
            m["wind_dir"] = col3.text_input("Wind direction (deg/cardinal)", value=m.get("wind_dir",""), key=f"wd_{name}")
            m["notes"] = st.text_area("Notes", value=m.get("notes",""), key=f"notes_{name}")
            st.session_state.meta[name] = m
    colA, colB = st.columns(2)
    with colA:
        st.download_button("Export metadata (JSON)",
                           data=json.dumps(st.session_state.meta, indent=2).encode("utf-8"),
                           file_name="f3k_metadata.json",
                           mime="application/json")
    with colB:
        meta_json = st.file_uploader("Import metadata JSON", type=["json"], accept_multiple_files=False, key="meta_import")
        if meta_json is not None:
            try:
                st.session_state.meta.update(json.loads(meta_json.read().decode("utf-8")))
                st.success("Metadata imported.")
            except Exception as e:
                st.error(f"Import failed: {e}")

left, right = st.columns([3,2])

with left:
    st.subheader("Interactive altitude plot")
    session_names = [p["name"] for p in processed]
    selected_sessions = st.multiselect("Select sessions", options=session_names, default=session_names, key="sess_select")

    fig = go.Figure()
    for p in processed:
        if p["name"] not in selected_sessions:
            continue
        fig.add_trace(go.Scatter(x=p["t"], y=p["alt"], mode="lines", name=p["name"]))
        if show_peak_labels and p["throws"]:
            peak_x = [p["t"][th["peak_idx"]] for th in p["throws"]]
            peak_y = [p["alt"][th["peak_idx"]] for th in p["throws"]]
            peak_text = [f'#{i+1} {int(round(y))}m' for i, y in enumerate(peak_y)]
            fig.add_trace(go.Scatter(x=peak_x, y=peak_y, mode="markers+text", text=peak_text,
                                     textposition="top center", name=f"{p['name']} peaks", showlegend=False))
    fig.update_layout(
        xaxis_title="Time since first launch start (s)" if sync_to_first_launch else "Time (s)",
        yaxis_title="Altitude (m)",
        hovermode="x unified",
        legend_title="Sessions",
        margin=dict(l=20, r=20, t=40, b=20),
        height=520
    )
    st.plotly_chart(fig, use_container_width=True)

    if enable_png_export:
        try:
            import kaleido  # noqa: F401
            png = fig.to_image(format="png", scale=2)
            st.download_button("Download current chart as PNG", data=png, file_name="f3k_chart.png", mime="image/png")
        except Exception as e:
            st.info("PNG export requires the 'kaleido' package. Install it with: pip install -U kaleido")

with right:
    st.subheader("Per-session summary")
    st.dataframe(summary_df, use_container_width=True, height=240)
    st.subheader("Best throws")
    if not stats_df.empty:
        best = stats_df.loc[stats_df.groupby("session")["max_alt_m"].idxmax()].sort_values("max_alt_m", ascending=False)
        st.dataframe(best, use_container_width=True, height=240)
    else:
        st.info("No throws detected with current thresholds/filters.")

st.subheader("All throws (filters applied)")
st.dataframe(stats_df, use_container_width=True, height=360)

st.subheader("Distribution of peak heights")
if not stats_df.empty:
    hist = px.histogram(stats_df, x="max_alt_m", nbins=20, title="Histogram of Throw Peak Altitudes (m)")
    hist.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=360)
    st.plotly_chart(hist, use_container_width=True)
else:
    st.info("Nothing to plot.")

# Download combined CSV
csv_all = stats_df.to_csv(index=False).encode("utf-8")
st.download_button("Download combined throws + metadata (CSV)", data=csv_all, file_name="f3k_throws_with_meta.csv", mime="text/csv")

st.caption("Tip: Use ZIP upload to quickly add a batch of session CSVs. Adjust detection thresholds in the sidebar if launches are missed or false positives appear.")
