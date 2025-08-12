# f3k_dashboard_combo_predict.py
# Combo app with layouts ("Classic" vs "Wide + X-Zoom") and predictive helpers:
# - Throw Strength Prediction (next-throw peak height, per session)
# - Session Fatigue Modeling (trend of peak height vs time/throw index)
# - Best Possible Throw Estimate (upper bound based on history + residuals)
# - Optimal Launch Timing in Tasks (highlight time windows with top conditions)
#
# Works with multiple CSVs or a ZIP of CSVs.
# No extra ML libs required (uses numpy/pandas for simple regressions).

import io
import json
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logo_url = "https://raw.githubusercontent.com/Furkua/f3k-analyzer/main/logo.png"

with st.sidebar:
    st.image(logo_url, use_container_width=True)

st.set_page_config(page_title="F3K Session Analyzer — Predict", layout="wide")

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
    vs = np.gradient(alt_s, dt_med)  # m/s approx

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
            search_end = min(n-1, start_idx + int(8.0/dt_med))
            peak_idx = start_idx
            for j in range(start_idx, search_end):
                if alt_s[j] >= alt_s[peak_idx]:
                    peak_idx = j
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
                # extra metrics for prediction
                t_start = t[start_idx]; t_peak = t[peak_idx]
                win_end = min(n-1, start_idx + int(2.0/dt_med))
                max_climb_2s = float(np.nanmax(vs[start_idx:win_end+1])) if win_end > start_idx else float("nan")
                duration_to_peak = max(t_peak - t_start, 1e-6)
                avg_climb_to_peak = float((alt_s[peak_idx] - alt_s[start_idx]) / duration_to_peak)
                throws.append({
                    "start_idx": start_idx, "peak_idx": peak_idx, "end_idx": end_idx,
                    "max_climb_2s": max_climb_2s, "avg_climb_to_peak": avg_climb_to_peak
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

# ---------------------------- Sidebar ---------------------------- #

st.sidebar.header("Upload sessions")
uploads = st.sidebar.file_uploader("Drop multiple CSVs or a ZIP", type=["csv","zip"], accept_multiple_files=True)

st.sidebar.header("Detection settings")
sync_to_first_launch = st.sidebar.checkbox("Synchronize sessions to first launch", True)
ground_alt = st.sidebar.number_input("Ground altitude (m)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
min_gap_s = st.sidebar.number_input("Min gap between throws (s)", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
min_flight_s = st.sidebar.number_input("Minimum flight duration (s)", min_value=1.0, max_value=30.0, value=4.0, step=0.5)
show_peak_labels = st.sidebar.checkbox("Show peak labels on chart", True)

st.sidebar.header("Layout")
layout_mode = st.sidebar.selectbox("Choose layout", ["Classic", "Wide + X-Zoom (Hero Plot)"], index=1)

# ---------------------------- Load sessions ---------------------------- #

if not uploads:
    st.title("F3K Session Analyzer — Predict")
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
                    if zname.lower().endswith(".csv"):
                        try:
                            sessions_raw.append(read_csv_bytes(zname, zf.read(zname)))
                        except Exception as e:
                            errors.append(str(e))
        except Exception as e:
            errors.append(f"{name}: {e}")
if errors:
    st.warning("Some files could not be parsed:\n\n" + "\n".join(errors))

# Determine global start
starts = []
for s in sessions_raw:
    idx, dt, vs = detect_first_start(s["alt"], s["sa"], s["t"])
    s["first_start_idx"] = idx
    s["dt_med"] = dt
    starts.append(s["t"][idx])
t0_global = min(starts) if sync_to_first_launch else 0.0

# Process sessions & assemble throw-level dataset
processed = []
rows = []
for s in sessions_raw:
    name = s["name"]
    t_al = s["t"] - (s["t"][s["first_start_idx"]] if sync_to_first_launch else t0_global)
    alt = pd.Series(pd.to_numeric(s["alt"], errors="coerce")).interpolate(limit_direction="both").values
    sa  = pd.Series(pd.to_numeric(s["sa"],  errors="coerce")).interpolate(limit_direction="both").values
    throws, vs = detect_throws(alt, sa, t_al, ground_alt, min_gap_s, min_flight_s)

    # build per-throw rows
    for k, th in enumerate(throws, start=1):
        t_start = float(t_al[th["start_idx"]])
        t_peak  = float(t_al[th["peak_idx"]])
        t_end   = float(t_al[th["end_idx"]])
        max_alt = float(alt[th["peak_idx"]])
        rows.append({
            "session": name,
            "throw": k,
            "t_start_s": round(t_start, 2),
            "t_peak_s": round(t_peak, 2),
            "t_end_s": round(t_end, 2),
            "elapsed_s": round(t_start, 2),
            "duration_s": round(t_end - t_start, 2),
            "max_alt_m": round(max_alt, 2),
            "max_climb_2s_mps": round(th["max_climb_2s"], 2),
            "avg_climb_to_peak_mps": round(th["avg_climb_to_peak"], 2),
        })

    processed.append({"name": name, "t": t_al, "alt": alt, "throws": throws})

stats_df = pd.DataFrame(rows).sort_values(["session","throw"]).reset_index(drop=True)

# ---------------------------- Predictive helpers ---------------------------- #

def ridge_fit_predict(X, y, X_next, alpha=1.0):
    # Add bias term
    Xb = np.c_[np.ones(len(X)), X]
    XtX = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
    beta = np.linalg.pinv(XtX) @ (Xb.T @ y)
    Xn = np.r_[1.0, X_next]
    yhat = float(Xn @ beta)
    # residual std
    y_pred_all = Xb @ beta
    resid = y - y_pred_all
    sigma = float(np.sqrt(max(np.mean(resid**2), 1e-6)))
    return yhat, sigma, beta

def make_predictions(df):
    # For each session, predict next throw height from features
    cards = []
    vrects = {}  # session -> list of (start,end) "good windows"
    if df.empty:
        return cards, vrects
    for session, sdf in df.groupby("session"):
        sdf = sdf.sort_values("throw").reset_index(drop=True)
        if len(sdf) < 3:
            continue
        # features for each throw
        X = sdf[["max_climb_2s_mps", "avg_climb_to_peak_mps", "duration_s"]].values
        # add previous height as a feature (lag 1)
        prev_height = np.r_[np.nan, sdf["max_alt_m"].values[:-1]]
        sdf["prev_height"] = prev_height
        X = np.c_[X, sdf["prev_height"].fillna(method="bfill").values]
        y = sdf["max_alt_m"].values

        # Train on throws except last, predict for "next" using last row features
        X_train = X[:-1]; y_train = y[:-1]
        X_next = X[-1]  # next-throw features approximated from last known
        yhat, sigma, beta = ridge_fit_predict(X_train, y_train, X_next, alpha=1.0)

        # Fatigue modeling: linear trend vs throw index and vs elapsed time
        idx = np.arange(len(sdf))
        # simple linear fit y = a + b*x
        A = np.c_[np.ones_like(idx), idx]
        b_idx = np.linalg.lstsq(A, y, rcond=None)[0]
        slope_per_throw = float(b_idx[1])  # meters per throw

        # vs elapsed time
        t = sdf["elapsed_s"].values
        At = np.c_[np.ones_like(t), t]
        b_t = np.linalg.lstsq(At, y, rcond=None)[0]
        slope_per_min = float(b_t[1] * 60.0)  # meters per minute

        # Best possible throw: upper bound = min(personal_best * 1.02, mean + 2*sigma global)
        personal_best = float(np.max(y))
        upper_bound = float(min(personal_best * 1.02, np.mean(y) + 2.0*sigma))

        # Optimal timing windows: find rolling windows where height & duration are in top quantile
        roll = sdf.rolling(window=3, min_periods=2).mean(numeric_only=True)
        good = (roll["max_alt_m"] >= np.nanpercentile(sdf["max_alt_m"], 75)) & \
               (roll["duration_s"] >= np.nanpercentile(sdf["duration_s"], 75))
        good = good.fillna(False).values
        windows = []
        start = None
        for i, ok in enumerate(good):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                windows.append((sdf.loc[start,"t_start_s"], sdf.loc[i-1,"t_end_s"]))
                start = None
        if start is not None:
            windows.append((sdf.loc[start,"t_start_s"], sdf.loc[len(sdf)-1,"t_end_s"]))
        vrects[session] = windows

        cards.append({
            "session": session,
            "pred_next_m": round(yhat, 1),
            "uncertainty_m": round(sigma, 1),
            "fatigue_slope_per_throw": round(slope_per_throw, 2),
            "fatigue_slope_per_min": round(slope_per_min, 2),
            "best_possible_m": round(upper_bound, 1),
            "personal_best_m": round(personal_best, 1),
            "good_windows": windows
        })
    return cards, vrects

pred_cards, good_windows = make_predictions(stats_df)

# ---------------------------- Layouts ---------------------------- #

def build_plot(selected_sessions, lock_y=False, add_rangeslider=False, vrects=None):
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
        # Add optimal timing vrects
        if vrects and p["name"] in vrects:
            for (xs, xe) in vrects[p["name"]]:
                fig.add_vrect(x0=xs, x1=xe, fillcolor="LightGreen", opacity=0.25, line_width=0)

    layout_kwargs = dict(
        xaxis_title="Time since first launch start (s)" if sync_to_first_launch else "Time (s)",
        yaxis_title="Altitude (m)",
        hovermode="x unified",
        legend_title="Sessions",
        margin=dict(l=10, r=10, t=10, b=0),
        height=600
    )
    if lock_y:
        layout_kwargs["yaxis"] = dict(fixedrange=True)
    if add_rangeslider:
        layout_kwargs["xaxis"] = dict(rangeslider=dict(visible=True))
    fig.update_layout(**layout_kwargs)
    return fig

st.title("F3K Session Analyzer — Predict")

if st.sidebar.selectbox("Prediction panels", ["Show"], index=0) == "Show":
    if pred_cards:
        st.markdown("### Predictive insights")
        cols = st.columns(min(3, len(pred_cards)))
        for i, card in enumerate(pred_cards):
            with cols[i % len(cols)]:
                st.markdown(f"**{card['session']}**")
                st.write(f"Next throw: **{card['pred_next_m']} m** ± {card['uncertainty_m']}")
                slope = card['fatigue_slope_per_throw']
                trend = "⬇️" if slope < -0.2 else ("⬆️" if slope > 0.2 else "➡️")
                st.write(f"Fatigue trend per throw: **{slope} m** {trend}")
                st.write(f"Best possible (now): **{card['best_possible_m']} m** (PB {card['personal_best_m']} m)")
                if card["good_windows"]:
                    st.write("Optimal timing windows (s):")
                    st.write(", ".join([f"{int(a)}–{int(b)}" for a,b in card["good_windows"]]))
                else:
                    st.write("No standout timing windows detected.")

# ----- Layout toggle -----
layout_mode = st.sidebar.selectbox("Layout", ["Classic", "Wide + X-Zoom (Hero Plot)"], index=1, key="layout_mode")

if layout_mode == "Wide + X-Zoom (Hero Plot)":
    # Hero plot at the top (full width, tall) as per your red rectangle
    st.subheader("Interactive altitude plot")
    session_names = [p["name"] for p in processed]
    selected_sessions = st.multiselect("Select sessions", options=session_names, default=session_names, key="sess_select_wide")
    hero_fig = build_plot(selected_sessions, lock_y=True, add_rangeslider=True, vrects=good_windows)
    st.plotly_chart(
        hero_fig, use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False, "doubleClick": "reset",
                "modeBarButtonsToAdd": ["zoom2d","pan2d","autoScale2d","resetScale2d"]}
    )

    # Under the hero plot, show summaries side-by-side
    left, right = st.columns([3,2], gap="large")
    with left:
        st.subheader("All throws")
        st.dataframe(stats_df, use_container_width=True, height=360)
    with right:
        st.subheader("Per-session summary")
        if stats_df.empty:
            st.info("No throws detected yet.")
        else:
            summary = (
                stats_df.groupby("session")
                .agg(throws=("throw","count"),
                     best_max_m=("max_alt_m","max"),
                     avg_max_m=("max_alt_m","mean"),
                     median_max_m=("max_alt_m","median"),
                     total_airtime_s=("duration_s","sum"))
                .reset_index()
                .sort_values("session")
            )
            st.dataframe(summary, use_container_width=True, height=360)

else:
    # Classic layout with more analysis panels
    st.subheader("Interactive altitude plot")
    session_names = [p["name"] for p in processed]
    selected_sessions = st.multiselect("Select sessions", options=session_names, default=session_names, key="sess_select_classic")
    fig = build_plot(selected_sessions, lock_y=False, add_rangeslider=False, vrects=good_windows)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "doubleClick": "reset"})

    left, right = st.columns([3,2])
    with left:
        st.subheader("Per-session summary")
        if stats_df.empty:
            st.info("No throws detected.")
        else:
            summary = (
                stats_df.groupby("session")
                .agg(throws=("throw","count"),
                     best_max_m=("max_alt_m","max"),
                     avg_max_m=("max_alt_m","mean"),
                     median_max_m=("max_alt_m","median"),
                     total_airtime_s=("duration_s","sum"),
                     avg_duration_s=("duration_s","mean"))
                .reset_index()
                .sort_values("session")
            )
            st.dataframe(summary, use_container_width=True, height=360)

    with right:
        st.subheader("Best throws")
        if not stats_df.empty:
            best = stats_df.loc[stats_df.groupby("session")["max_alt_m"].idxmax()].sort_values("max_alt_m", ascending=False)
            st.dataframe(best, use_container_width=True, height=360)

    st.subheader("All throws")
    st.dataframe(stats_df.sort_values(["session","throw"]), use_container_width=True, height=360)

    st.subheader("Distribution of peak heights")
    if not stats_df.empty:
        hist = px.histogram(stats_df, x="max_alt_m", nbins=20, title="Histogram of Throw Peak Altitudes (m)")
        hist.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=360)
        st.plotly_chart(hist, use_container_width=True)
    else:
        st.info("Nothing to plot yet.")
