# ===== Imports & Setup =====
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).parent.resolve()
DATA_DIR = (ROOT / "data") if (ROOT / "data").exists() else Path(os.getenv("DATA_DIR", ROOT))
OUT_DIR = ROOT / "out"
OUT_DIR.mkdir(exist_ok=True)

RED = "#ff3b3b"; DARK = "#0e0e0e"; GRID = "#333333"; TXT = "#ffffff"
plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK, "savefig.facecolor": DARK,
    "axes.edgecolor": GRID, "axes.labelcolor": TXT, "text.color": TXT,
    "xtick.color": TXT, "ytick.color": TXT, "grid.color": GRID,
    "axes.grid": True, "grid.linestyle": ":", "font.size": 11,
})
reds = LinearSegmentedColormap.from_list("reds_rev", ["#ffefef", "#ff9f9f", "#ff5a5a", "#b30d0d"])

FILE_MAP = {
    "circuits": "circuits.csv",
    "constructors": "constructors.csv",
    "constructor_results": "constructor_results.csv",
    "constructor_standings": "constructor_standings.csv",
    "drivers": "drivers.csv",
    "driver_standings": "driver_standings.csv",
    "lap_times": "lap_times.csv",
    "pit_stops": "pit_stops.csv",
    "qualifying": "qualifying.csv",
    "races": "races.csv",
    "results": "results.csv",
    "seasons": "seasons.csv",
    "sprint_results": "sprint_results.csv",
    "status": "status.csv",
}

def load_csv(name: str) -> Optional[pd.DataFrame]:
    p = DATA_DIR / FILE_MAP.get(name, "")
    return pd.read_csv(p) if p.exists() else None

circuits = load_csv("circuits")
constructors = load_csv("constructors")
constructor_results = load_csv("constructor_results")
constructor_standings = load_csv("constructor_standings")
drivers = load_csv("drivers")
driver_standings = load_csv("driver_standings")
lap_times = load_csv("lap_times")
pit_stops = load_csv("pit_stops")
qualifying = load_csv("qualifying")
races = load_csv("races")
results = load_csv("results")
seasons = load_csv("seasons")
sprint_results = load_csv("sprint_results")
status = load_csv("status")

if drivers is not None and "driverId" in drivers.columns:
    drivers["driver"] = drivers[["forename","surname"]].fillna("").agg(" ".join, axis=1).str.strip()
if constructors is not None and "name" in constructors.columns:
    constructors = constructors.rename(columns={"name":"constructor"})

def get_year_bounds() -> Tuple[int,int]:
    if races is not None and "year" in races.columns:
        yrs = races["year"].dropna().astype(int); return int(yrs.min()), int(yrs.max())
    if seasons is not None and "year" in seasons.columns:
        yrs = seasons["year"].dropna().astype(int); return int(yrs.min()), int(yrs.max())
    return 1950, 2024
YEAR_MIN, YEAR_MAX = get_year_bounds()

def filter_year_span(df: pd.DataFrame, y0: int = YEAR_MIN, y1: int = YEAR_MAX) -> pd.DataFrame:
    if df is None or df.empty: return df
    if "raceId" in df.columns and races is not None and "raceId" in races.columns:
        df = df.merge(races[["raceId","year"]], on="raceId", how="left")
    if "year" in df.columns: df = df[(df["year"] >= y0) & (df["year"] <= y1)]
    return df

def savefig(fig, name: str):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=160, bbox_inches="tight"); plt.close(fig)

def note_figure(title: str, note: str):
    fig, ax = plt.subplots(figsize=(6,3.2)); ax.axis("off"); ax.set_title(title, loc="left", fontsize=13, fontweight="bold"); ax.text(0.01, 0.5, note, transform=ax.transAxes)
    return fig

# ===== Key Metrics (Cards) =====
def top_constructor_by_points(y0=YEAR_MIN, y1=YEAR_MAX):
    if constructor_standings is None or races is None: return None, None
    cs = filter_year_span(constructor_standings.copy(), y0, y1)
    if not {"constructorId","points","year"}.issubset(cs.columns): return None, None
    agg = cs.groupby(["constructorId","year"], as_index=False)["points"].max()
    agg = agg.merge(constructors[["constructorId","constructor"]], on="constructorId", how="left")
    tot = agg.groupby("constructor", as_index=False)["points"].sum().sort_values("points", ascending=False)
    if tot.empty: return None, None
    r = tot.iloc[0]; return r["constructor"], int(round(r["points"]))

def top_driver_by_podium(y0=YEAR_MIN, y1=YEAR_MAX):
    if results is None or races is None: return None, None
    res = filter_year_span(results[["raceId","driverId","position"]].copy(), y0, y1)
    res["pos_num"] = pd.to_numeric(res["position"], errors="coerce")
    pods = res[res["pos_num"].isin([1,2,3])].groupby("driverId").size().reset_index(name="podiums")
    if pods.empty: return None, None
    pods = pods.merge(drivers[["driverId","driver"]], on="driverId", how="left").sort_values("podiums", ascending=False)
    r = pods.iloc[0]; return r.get("driver"), int(r.get("podiums"))

def avg_pit_stops(y0=YEAR_MIN, y1=YEAR_MAX):
    if pit_stops is None or races is None: return None
    ps = filter_year_span(pit_stops[["raceId"]].copy(), y0, y1)
    per_race = ps.groupby("raceId").size().reset_index(name="stops")
    return int(round(per_race["stops"].mean())) if not per_race.empty else None

def total_laps(y0=YEAR_MIN, y1=YEAR_MAX):
    if lap_times is None or races is None: return None
    lt = filter_year_span(lap_times[["raceId","lap"]].copy(), y0, y1); return int(lt.shape[0]) if lt is not None and not lt.empty else None

def highest_speed(y0=YEAR_MIN, y1=YEAR_MAX):
    if results is None: return None, None
    cols = [c for c in results.columns if c.lower() in ("fastestlapspeed","fastestlap_speed","speed")]
    if not cols: return None, None
    col = cols[0]; df = results[["raceId","driverId",col]].copy()
    df[col] = pd.to_numeric(df[col], errors="coerce"); df = df.dropna(subset=[col]); df = filter_year_span(df, y0, y1)
    if df.empty: return None, None
    i = df[col].idxmax(); best = df.loc[i]
    dname = drivers.loc[drivers["driverId"] == best["driverId"], "driver"].iloc[0] if drivers is not None and not drivers.empty else None
    return round(float(best[col]), 1), dname

def plot_metric_cards(y0=YEAR_MIN, y1=YEAR_MAX):
    tc_name, tc_pts = top_constructor_by_points(y0, y1)
    td_name, td_pods = top_driver_by_podium(y0, y1)
    avg_stops = avg_pit_stops(y0, y1)
    max_speed, speed_driver = highest_speed(y0, y1)
    laps = total_laps(y0, y1)
    fig, ax = plt.subplots(figsize=(4,8)); ax.axis("off")
    def block(y, title, value, subtitle=None):
        ax.text(0.02, y, title, fontsize=9); ax.text(0.02, y-0.04, f"{value}", fontsize=18, color=RED, fontweight="bold")
        if subtitle: ax.text(0.02, y-0.09, subtitle, fontsize=10); ax.add_patch(plt.Rectangle((0.01, y-0.13), 0.95, 0.12, transform=ax.transAxes, facecolor="none", edgecolor=GRID, linewidth=1))
    block(0.95, "Top Constructor by Points", f"{tc_pts:,}" if tc_pts is not None else "—", tc_name or "—")
    block(0.77, "Top Driver by Podium", f"{td_pods:,}" if td_pods is not None else "—", td_name or "—")
    block(0.59, "Avg Pit Stops", f"{avg_stops:,}" if avg_stops is not None else "—")
    block(0.41, "Highest Driver Speed", f"{max_speed} km/h" if max_speed is not None else "—", speed_driver or "—")
    block(0.23, "Total Laps", f"{laps:,}" if laps is not None else "—")
    ax.set_title("Key Metrics", loc="left", fontsize=14, fontweight="bold"); plt.tight_layout(); savefig(fig, "metric_cards")

# ===== Top 10 Drivers by Points (Pie) =====
def top10_drivers_points(y0=YEAR_MIN, y1=YEAR_MAX):
    if driver_standings is None or races is None: return pd.DataFrame(columns=["driver","points"])
    ds = filter_year_span(driver_standings[["raceId","driverId","points"]], y0, y1)
    agg = ds.groupby(["driverId","year"], as_index=False)["points"].max().merge(drivers[["driverId","driver"]], on="driverId", how="left")
    return agg.groupby("driver", as_index=False)["points"].sum().sort_values("points", ascending=False).head(10)

def plot_top10_drivers_pie(y0=YEAR_MIN, y1=YEAR_MAX):
    df = top10_drivers_points(y0, y1)
    if df.empty: savefig(note_figure("Top 10 Drivers by Points", "Need driver_standings + races."), "top10_drivers_pie"); return
    fig, ax = plt.subplots(figsize=(6,4)); ax.set_title("Top 10 Drivers by Points", loc="left", fontsize=13, fontweight="bold")
    wedges, _ = ax.pie(df["points"].values, startangle=90); ax.legend(wedges, df["driver"].values, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    savefig(fig, "top10_drivers_pie")

# ===== Avg. Finish Position by Driver (Heatmap) =====
# ===== Avg. Finish Position by Team (Heatmap) =====
def avg_finish_heatmaps(y0=YEAR_MIN, y1=YEAR_MAX):
    if results is None or races is None: return pd.DataFrame(), pd.DataFrame()
    res = filter_year_span(results[["raceId","driverId","constructorId","position"]], y0, y1); res["pos_num"] = pd.to_numeric(res["position"], errors="coerce"); res = res.dropna(subset=["pos_num"])
    d = res.groupby(["driverId","year"], as_index=False)["pos_num"].mean().merge(drivers[["driverId","driver"]], on="driverId", how="left")
    t = res.groupby(["constructorId","year"], as_index=False)["pos_num"].mean().merge(constructors[["constructorId","constructor"]], on="constructorId", how="left")
    return d, t

def plot_avg_finish_heatmaps(y0=YEAR_MIN, y1=YEAR_MAX):
    d, t = avg_finish_heatmaps(y0, y1)
    if d.empty: savefig(note_figure("Avg. Finish Position by Driver", "Need results + races + drivers."), "avg_finish_by_driver")
    else:
        piv = d.pivot_table(index="driver", columns="year", values="pos_num", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(7.5,5)); im = ax.imshow(piv.values, aspect="auto", cmap=reds, interpolation="nearest")
        ax.set_title("Avg. Finish Position by Driver", loc="left", fontsize=13, fontweight="bold")
        ax.set_yticks(np.arange(piv.shape[0])); ax.set_yticklabels(piv.index)
        ax.set_xticks(np.arange(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
        fig.colorbar(im, ax=ax, shrink=0.85).set_label("Avg Pos"); plt.tight_layout(); savefig(fig, "avg_finish_by_driver")
    if t.empty: savefig(note_figure("Avg. Finish Position by Team", "Need results + races + constructors."), "avg_finish_by_team")
    else:
        piv = t.pivot_table(index="constructor", columns="year", values="pos_num", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(7.5,5)); im = ax.imshow(piv.values, aspect="auto", cmap=reds, interpolation="nearest")
        ax.set_title("Avg. Finish Position by Team", loc="left", fontsize=13, fontweight="bold")
        ax.set_yticks(np.arange(piv.shape[0])); ax.set_yticklabels(piv.index)
        ax.set_xticks(np.arange(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
        fig.colorbar(im, ax=ax, shrink=0.85).set_label("Avg Pos"); plt.tight_layout(); savefig(fig, "avg_finish_by_team")

# ===== DNFs by Constructor (Scatter) =====
def dnfs_by_constructor(y0=YEAR_MIN, y1=YEAR_MAX):
    if results is None or status is None or races is None: return pd.DataFrame(columns=["constructor","dnfs"])
    res = filter_year_span(results[["raceId","constructorId","statusId"]], y0, y1).merge(status[["statusId","status"]], on="statusId", how="left")
    dnf_mask = res["status"].fillna("").str.contains(r"Retired|Accident|Collision|Engine|Transmission|Electrical|Hydraulics|Wheel|Suspension|Overheating|Exhaust|Brakes|Driveshaft|Fuel|Clutch|DNF|Did not finish|Did not start|Did not qualify|Withdrew|Disqualified", case=False, regex=True)
    d = res[dnf_mask].groupby("constructorId").size().reset_index(name="dnfs").merge(constructors[["constructorId","constructor"]], on="constructorId", how="left")
    return d.sort_values("dnfs", ascending=False)

def plot_dnfs_by_constructor(y0=YEAR_MIN, y1=YEAR_MAX):
    df = dnfs_by_constructor(y0, y1)
    if df.empty: savefig(note_figure("DNFs by Constructor", "Need results + status + races."), "dnfs_by_constructor"); return
    df = df.sort_values("dnfs", ascending=True).reset_index(drop=True); df["cum"] = df["dnfs"].cumsum()
    fig, ax = plt.subplots(figsize=(7,4)); ax.scatter(df["cum"], df["dnfs"], s=35, c=RED, alpha=0.9)
    for _, r in df.iterrows(): ax.text(r["cum"], r["dnfs"]+0.5, r["constructor"], ha="center", va="bottom", fontsize=8)
    ax.set_title("DNFs by Constructor", loc="left", fontsize=13, fontweight="bold"); ax.set_xlabel("Cumulative DNF Count (sorted)"); ax.set_ylabel("DNFs"); savefig(fig, "dnfs_by_constructor")

# ===== Podium Wins by Constructor (Bar) =====
def podiums_by_constructor(y0=YEAR_MIN, y1=YEAR_MAX):
    if results is None or races is None: return pd.DataFrame(columns=["constructor","podiums"])
    res = filter_year_span(results[["raceId","constructorId","position"]], y0, y1); res["pos_num"] = pd.to_numeric(res["position"], errors="coerce")
    df = res[res["pos_num"].isin([1,2,3])].groupby("constructorId").size().reset_index(name="podiums").merge(constructors[["constructorId","constructor"]], on="constructorId", how="left")
    return df.sort_values("podiums", ascending=False)

def plot_podiums_by_constructor(y0=YEAR_MIN, y1=YEAR_MAX):
    df = podiums_by_constructor(y0, y1)
    if df.empty: savefig(note_figure("Podium Wins by Constructor", "Need results + races."), "podiums_by_constructor"); return
    fig, ax = plt.subplots(figsize=(7,4)); ax.bar(df["constructor"], df["podiums"], color=RED, alpha=0.9)
    ax.set_title("Podium Wins by Constructor", loc="left", fontsize=13, fontweight="bold"); ax.set_xticklabels(df["constructor"], rotation=45, ha="right"); ax.set_ylabel("Podiums"); savefig(fig, "podiums_by_constructor")

# ===== Leading Points by Country (Barh) =====
def leading_points_by_country(y0=YEAR_MIN, y1=YEAR_MAX):
    if driver_standings is None or races is None or drivers is None: return pd.DataFrame(columns=["nationality","points"])
    ds = filter_year_span(driver_standings[["raceId","driverId","points"]], y0, y1)
    d = ds.groupby(["driverId","year"], as_index=False)["points"].max().merge(drivers[["driverId","nationality"]], on="driverId", how="left")
    return d.groupby("nationality", as_index=False)["points"].sum().sort_values("points", ascending=False)

def plot_leading_points_by_country(y0=YEAR_MIN, y1=YEAR_MAX, topn=15):
    df = leading_points_by_country(y0, y1)
    if df.empty: savefig(note_figure("Leading Pts. by Country", "Need driver_standings + races + drivers."), "leading_points_by_country"); return
    df = df.head(topn); fig, ax = plt.subplots(figsize=(5,6)); ax.barh(df["nationality"], df["points"], color=RED, alpha=0.9)
    ax.set_title("Leading Pts. by Country", loc="left", fontsize=13, fontweight="bold"); ax.invert_yaxis(); ax.xaxis.set_major_locator(MaxNLocator(6)); savefig(fig, "leading_points_by_country")

# ===== Circuit Map (Lon/Lat Scatter) =====
def circuits_usage_map(y0=YEAR_MIN, y1=YEAR_MAX):
    if circuits is None or races is None: return pd.DataFrame(columns=["circuit","lat","lng","events","country"])
    rr = races[(races["year"] >= y0) & (races["year"] <= y1)]; counts = rr.groupby("circuitId").size().reset_index(name="events")
    c = circuits.rename(columns={"name":"circuit"}); latcol = "lat" if "lat" in c.columns else ("latitude" if "latitude" in c.columns else None); lngcol = "lng" if "lng" in c.columns else ("longitude" if "longitude" in c.columns else None)
    if not latcol or not lngcol: return pd.DataFrame(columns=["circuit","lat","lng","events","country"])
    merged = counts.merge(c, on="circuitId", how="left").rename(columns={latcol:"lat", lngcol:"lng"})
    return merged[["circuitId","circuit","lat","lng","country","events"]]

def plot_circuits_scatter(y0=YEAR_MIN, y1=YEAR_MAX):
    df = circuits_usage_map(y0, y1)
    if df.empty: savefig(note_figure("Circuit Map", "Need circuits + races."), "circuit_map"); return
    fig, ax = plt.subplots(figsize=(8,4.2))
    s = 10 + 5*(df["events"]-df["events"].min())/(df["events"].max()-df["events"].min()+1e-9)
    ax.scatter(df["lng"], df["lat"], s=s*10, c=RED, alpha=0.8)
    ax.set_title("Circuit Map (bubble size = number of events)", loc="left", fontsize=13, fontweight="bold"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); savefig(fig, "circuit_map")

# ===== Fastest Lap by Season (Line) =====
def fastest_lap_by_season(y0=YEAR_MIN, y1=YEAR_MAX):
    if results is not None and "fastestLapTime" in results.columns:
        r = filter_year_span(results[["raceId","fastestLapTime"]].copy(), y0, y1)
        def to_seconds(t):
            if pd.isna(t): return np.nan
            t = str(t); 
            if ":" not in t: return pd.to_numeric(t, errors="coerce")
            ps = t.split(":")
            try:
                if len(ps)==2: m,s=ps; return float(m)*60+float(s)
                if len(ps)==3: h,m,s=ps; return float(h)*3600+float(m)*60+float(s)
            except: return np.nan
            return np.nan
        r["sec"] = r["fastestLapTime"].apply(to_seconds); r = r.dropna(subset=["sec"])
        if not r.empty: return r.groupby("year", as_index=False)["sec"].min().rename(columns={"sec":"fastest"})
    if lap_times is None: return pd.DataFrame(columns=["year","fastest"])
    lt = filter_year_span(lap_times[["raceId","time"]].copy(), y0, y1)
    def to_seconds_lap(t):
        if pd.isna(t): return np.nan
        t = str(t); ps = t.split(":")
        try:
            if len(ps)==2: m,s=ps; return float(m)*60+float(s)
            if len(ps)==3: h,m,s=ps; return float(h)*3600+float(m)*60+float(s)
            return float(t)
        except: return np.nan
    lt["sec"] = lt["time"].apply(to_seconds_lap)
    return lt.groupby("year", as_index=False)["sec"].min().rename(columns={"sec":"fastest"})

def plot_fastest_by_season(y0=YEAR_MIN, y1=YEAR_MAX):
    df = fastest_lap_by_season(y0, y1)
    if df.empty: savefig(note_figure("Fastest Lap by Season", "Need results.fastestLapTime or lap_times + races."), "fastest_by_season"); return
    df = df.sort_values("year"); fig, ax = plt.subplots(figsize=(7,3.8)); ax.plot(df["year"], df["fastest"], linewidth=2, color=RED)
    ax.set_title("Fastest Lap by Season", loc="left", fontsize=13, fontweight="bold"); ax.set_xlabel("Year"); ax.set_ylabel("Fastest Lap (s)"); savefig(fig, "fastest_by_season")

# ===== Run All =====
def main(y0: int = YEAR_MIN, y1: int = YEAR_MAX):
    plot_metric_cards(y0, y1)
    plot_top10_drivers_pie(y0, y1)
    plot_avg_finish_heatmaps(y0, y1)
    plot_dnfs_by_constructor(y0, y1)
    plot_podiums_by_constructor(y0, y1)
    plot_leading_points_by_country(y0, y1)
    plot_circuits_scatter(y0, y1)
    plot_fastest_by_season(y0, y1)

if __name__ == "__main__":
    main()
