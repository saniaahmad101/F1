
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

# Optional FastF1
try:
    import fastf1
    import fastf1.plotting as f1plot
    FASTF1_AVAILABLE = True
except Exception:
    FASTF1_AVAILABLE = False

# ===== Paths & Theme =====
ROOT = Path(__file__).parent.resolve()
DATA_DIR = (ROOT / "data") if (ROOT / "data").exists() else Path(os.getenv("DATA_DIR", ROOT))
OUT_DIR  = ROOT / "out"
FIG_DIR  = OUT_DIR / "figs"
TAB_DIR  = OUT_DIR / "tables"
NOTE_DIR = OUT_DIR / "notes"
for d in (OUT_DIR, FIG_DIR, TAB_DIR, NOTE_DIR):
    d.mkdir(parents=True, exist_ok=True)

DARK = "#0e0e0e"; GRID = "#333333"; TXT = "#ffffff"; ACCENT = "#ff3b3b"
plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK, "savefig.facecolor": DARK,
    "axes.edgecolor": GRID, "axes.labelcolor": TXT, "text.color": TXT,
    "xtick.color": TXT, "ytick.color": TXT, "grid.color": GRID,
    "axes.grid": True, "grid.linestyle": ":", "font.size": 11
})
REDS = LinearSegmentedColormap.from_list("reds_rev", ["#ffefef", "#ff9f9f", "#ff5a5a", "#b30d0d"])

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

# ===== I/O helpers =====
def load_csv(name: str) -> Optional[pd.DataFrame]:
    p = DATA_DIR / FILE_MAP.get(name, "")
    return pd.read_csv(p) if p.exists() else None

def write_table(df: pd.DataFrame, name: str):
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)

def write_note(name: str, text: str):
    (NOTE_DIR / f"{name}.txt").write_text(text.strip() + "\n", encoding="utf-8")

def savefig(fig, name: str):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def note_figure(title: str, note: str, filename: str):
    fig, ax = plt.subplots(figsize=(6,3.2))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.text(0.01, 0.5, note, transform=ax.transAxes)
    savefig(fig, filename)

# ===== Load base tables =====
circuits              = load_csv("circuits")
constructors          = load_csv("constructors")
constructor_results   = load_csv("constructor_results")
constructor_standings = load_csv("constructor_standings")
drivers               = load_csv("drivers")
driver_standings      = load_csv("driver_standings")
lap_times             = load_csv("lap_times")
pit_stops             = load_csv("pit_stops")
qualifying            = load_csv("qualifying")
races                 = load_csv("races")
results               = load_csv("results")
seasons               = load_csv("seasons")
sprint_results        = load_csv("sprint_results")
status_tbl            = load_csv("status")

# Basic enrichments / column normalizations
if drivers is not None and {"forename","surname"}.issubset(drivers.columns):
    drivers["driver"] = drivers[["forename","surname"]].fillna("").agg(" ".join, axis=1).str.strip()

if constructors is not None and "name" in constructors.columns:
    constructors = constructors.rename(columns={"name":"constructor"})

# Make sure races has a 'year' column (some dumps use 'season')
if races is not None:
    if "year" not in races.columns and "season" in races.columns:
        races = races.rename(columns={"season": "year"})

# ===== Year utilities =====
def get_year_bounds() -> Tuple[int,int]:
    if races is not None and "year" in races.columns:
        yrs = pd.to_numeric(races["year"], errors="coerce").dropna().astype(int)
        if not yrs.empty:
            return int(yrs.min()), int(yrs.max())
    if seasons is not None and "year" in seasons.columns:
        yrs = pd.to_numeric(seasons["year"], errors="coerce").dropna().astype(int)
        if not yrs.empty:
            return int(yrs.min()), int(yrs.max())
    return 1950, 2024

YEAR_MIN, YEAR_MAX = get_year_bounds()

def filter_year_span(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "raceId" in df.columns and races is not None and "raceId" in races.columns:
        df = df.merge(races[["raceId","year"]], on="raceId", how="left")
    if "year" in df.columns:
        df = df[(df["year"] >= y0) & (df["year"] <= y1)]
    return df

# ===== Dashboard-equivalent DATA =====
def data_top_constructor_points(y0: int, y1: int) -> pd.DataFrame:
    if constructor_standings is None or races is None or constructors is None:
        write_note("top_constructor_by_points", "Need constructor_standings + races + constructors.")
        return pd.DataFrame(columns=["constructor","points_total"])
    cs = filter_year_span(constructor_standings[["raceId","constructorId","points"]].copy(), y0, y1)
    if cs.empty:
        write_note("top_constructor_by_points", "No constructor standings in span.")
        return pd.DataFrame(columns=["constructor","points_total"])
    year_max = cs.groupby(["constructorId","year"], as_index=False)["points"].max()
    year_max = year_max.merge(constructors[["constructorId","constructor"]], on="constructorId", how="left")
    out = (year_max.groupby("constructor", as_index=False)["points"]
                  .sum()
                  .rename(columns={"points":"points_total"})
                  .sort_values("points_total", ascending=False))
    write_table(out, "top_constructor_points")
    return out

def data_top_driver_podiums(y0: int, y1: int) -> pd.DataFrame:
    if results is None or races is None or drivers is None:
        write_note("top_driver_podiums", "Need results + races + drivers.")
        return pd.DataFrame(columns=["driver","podiums"])
    res = filter_year_span(results[["raceId","driverId","position"]].copy(), y0, y1)
    res["pos_num"] = pd.to_numeric(res["position"], errors="coerce")
    pods = (res[res["pos_num"].isin([1,2,3])]
            .groupby("driverId").size().reset_index(name="podiums")
            .merge(drivers[["driverId","driver"]], on="driverId", how="left")
            .sort_values("podiums", ascending=False))
    write_table(pods[["driver","podiums"]], "top_driver_podiums")
    return pods[["driver","podiums"]]

def data_avg_pit_stops_per_race(y0: int, y1: int) -> pd.DataFrame:
    if pit_stops is None or races is None:
        write_note("avg_pit_stops", "Need pit_stops + races.")
        return pd.DataFrame(columns=["avg_stops_per_race"])
    ps = filter_year_span(pit_stops[["raceId"]].copy(), y0, y1)
    per_race = ps.groupby("raceId").size().reset_index(name="stops")
    avgv = float(per_race["stops"].mean()) if not per_race.empty else np.nan
    out = pd.DataFrame({"avg_stops_per_race":[avgv]})
    write_table(out, "avg_pit_stops_per_race")
    return out

def data_total_laps(y0: int, y1: int) -> pd.DataFrame:
    if lap_times is None or races is None:
        write_note("total_laps", "Need lap_times + races.")
        return pd.DataFrame(columns=["total_laps"])
    lt = filter_year_span(lap_times[["raceId","lap"]].copy(), y0, y1)
    total = int(lt.shape[0]) if lt is not None and not lt.empty else 0
    out = pd.DataFrame({"total_laps":[total]})
    write_table(out, "total_laps")
    return out

def data_highest_speed(y0: int, y1: int) -> pd.DataFrame:
    if results is None or drivers is None:
        write_note("highest_speed", "Need results + drivers.")
        return pd.DataFrame(columns=["speed","driver"])
    cols = [c for c in results.columns if c.lower() in ("fastestlapspeed","fastestlap_speed","speed")]
    if not cols:
        write_note("highest_speed", "No speed column found in results.")
        return pd.DataFrame(columns=["speed","driver"])
    col = cols[0]
    df = results[["raceId","driverId",col]].copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    df = filter_year_span(df, y0, y1)
    if df.empty:
        write_note("highest_speed", "No speed data in span.")
        return pd.DataFrame(columns=["speed","driver"])
    i = df[col].idxmax()
    best = df.loc[i]
    dname = drivers.loc[drivers["driverId"] == best["driverId"], "driver"]
    driver_name = dname.iloc[0] if not dname.empty else None
    out = pd.DataFrame({"speed":[round(float(best[col]), 3)], "driver":[driver_name]})
    write_table(out, "highest_speed")
    return out

def data_top10_driver_points(y0: int, y1: int) -> pd.DataFrame:
    if driver_standings is None or races is None or drivers is None:
        write_note("top10_driver_points", "Need driver_standings + races + drivers.")
        return pd.DataFrame(columns=["driver","points_total"])
    ds = filter_year_span(driver_standings[["raceId","driverId","points"]], y0, y1)
    agg_year = ds.groupby(["driverId","year"], as_index=False)["points"].max()
    merged = agg_year.merge(drivers[["driverId","driver"]], on="driverId", how="left")
    out = (merged.groupby("driver", as_index=False)["points"]
           .sum()
           .rename(columns={"points":"points_total"})
           .sort_values("points_total", ascending=False)
           .head(10))
    write_table(out, "top10_driver_points")
    return out

def data_avg_finish_positions(y0: int, y1: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results is None or races is None:
        write_note("avg_finish_positions", "Need results + races.")
        return pd.DataFrame(), pd.DataFrame()
    res = filter_year_span(results[["raceId","driverId","constructorId","position"]], y0, y1)
    res["pos_num"] = pd.to_numeric(res["position"], errors="coerce")
    res = res.dropna(subset=["pos_num"])

    by_driver = res.groupby(["driverId","year"], as_index=False)["pos_num"].mean()
    if drivers is not None:
        by_driver = by_driver.merge(drivers[["driverId","driver"]], on="driverId", how="left")
    by_driver = by_driver.rename(columns={"pos_num":"avg_finish"}).sort_values(["year","avg_finish"])

    by_team = res.groupby(["constructorId","year"], as_index=False)["pos_num"].mean()
    if constructors is not None:
        by_team = by_team.merge(constructors[["constructorId","constructor"]], on="constructorId", how="left")
    by_team = by_team.rename(columns={"pos_num":"avg_finish"}).sort_values(["year","avg_finish"])

    write_table(by_driver, "avg_finish_by_driver")
    write_table(by_team, "avg_finish_by_constructor")
    return by_driver, by_team

def data_dnfs_by_constructor(y0: int, y1: int) -> pd.DataFrame:
    if results is None or status_tbl is None or races is None or constructors is None:
        write_note("dnfs_by_constructor", "Need results + status + races + constructors.")
        return pd.DataFrame(columns=["constructor","dnfs"])
    res = filter_year_span(results[["raceId","constructorId","statusId"]], y0, y1)
    res = res.merge(status_tbl[["statusId","status"]], on="statusId", how="left")
    patt = r"Retired|Accident|Collision|Engine|Transmission|Electrical|Hydraulics|Wheel|Suspension|Overheating|Exhaust|Brakes|Driveshaft|Fuel|Clutch|DNF|Did not finish|Did not start|Did not qualify|Withdrew|Disqualified"
    dnf = res["status"].fillna("").str.contains(patt, case=False, regex=True)
    d = (res[dnf]
         .groupby("constructorId").size().reset_index(name="dnfs")
         .merge(constructors[["constructorId","constructor"]], on="constructorId", how="left")
         .sort_values("dnfs", ascending=False))
    write_table(d[["constructor","dnfs"]], "dnfs_by_constructor")
    return d[["constructor","dnfs"]]

def data_points_by_country(y0: int, y1: int) -> pd.DataFrame:
    if driver_standings is None or drivers is None or races is None:
        write_note("points_by_country", "Need driver_standings + drivers + races.")
        return pd.DataFrame(columns=["nationality","points_total"])
    ds = filter_year_span(driver_standings[["raceId","driverId","points"]], y0, y1)
    yearmax = ds.groupby(["driverId","year"], as_index=False)["points"].max()
    merged = yearmax.merge(drivers[["driverId","nationality"]], on="driverId", how="left")
    out = (merged.groupby("nationality", as_index=False)["points"]
           .sum()
           .rename(columns={"points":"points_total"})
           .sort_values("points_total", ascending=False))
    write_table(out, "points_by_country")
    return out

def data_fastest_lap_by_season(y0: int, y1: int) -> pd.DataFrame:
    def to_sec_str(t) -> float:
        if pd.isna(t): return np.nan
        s = str(t)
        if ":" not in s:
            return pd.to_numeric(s, errors="coerce")
        parts = s.split(":")
        try:
            if len(parts) == 2:
                m, sec = parts; return float(m)*60 + float(sec)
            if len(parts) == 3:
                h, m, sec = parts; return float(h)*3600 + float(m)*60 + float(sec)
        except Exception:
            return np.nan
        return np.nan

    if results is not None and "fastestLapTime" in results.columns:
        r = filter_year_span(results[["raceId","fastestLapTime"]].copy(), y0, y1)
        r["sec"] = r["fastestLapTime"].apply(to_sec_str)
        r = r.dropna(subset=["sec"])
        if not r.empty:
            out = r.groupby("year", as_index=False)["sec"].min().rename(columns={"sec":"fastest_s"})
            write_table(out, "fastest_lap_by_season")
            return out

    if lap_times is None:
        write_note("fastest_lap_by_season", "Need results.fastestLapTime OR lap_times.time + races.")
        return pd.DataFrame(columns=["year","fastest_s"])

    lt = filter_year_span(lap_times[["raceId","time"]].copy(), y0, y1)
    def to_sec_any(t):
        if pd.isna(t): return np.nan
        s = str(t); parts = s.split(":")
        try:
            if len(parts) == 2: m, sec = parts; return float(m)*60 + float(sec)
            if len(parts) == 3: h, m, sec = parts; return float(h)*3600 + float(m)*60 + float(sec)
            return float(s)
        except Exception:
            return np.nan
    lt["sec"] = lt["time"].apply(to_sec_any)
    out = lt.groupby("year", as_index=False)["sec"].min().rename(columns={"sec":"fastest_s"})
    write_table(out, "fastest_lap_by_season")
    return out

# ===== Extra analyses (warning-free) =====
def data_constructor_consistency(y0: int, y1: int) -> pd.DataFrame:
    """
    Constructor consistency from results:
    - avg_finish, finish_std, podium_rate, dnf_rate, starts
    """
    if results is None or races is None or constructors is None or status_tbl is None:
        write_note("constructor_consistency", "Need results + races + constructors + status.")
        return pd.DataFrame()

    res = filter_year_span(results[["raceId","constructorId","position","statusId"]], y0, y1)
    res = res.merge(constructors[["constructorId","constructor"]], on="constructorId", how="left")
    res = res.merge(status_tbl[["statusId","status"]], on="statusId", how="left")
    res["pos_num"] = pd.to_numeric(res["position"], errors="coerce")

    patt = r"Retired|Accident|Collision|Engine|Transmission|Electrical|Hydraulics|Wheel|Suspension|Overheating|Exhaust|Brakes|Driveshaft|Fuel|Clutch|DNF|Did not finish|Did not start|Did not qualify|Withdrew|Disqualified"
    res["dnf_flag"] = res["status"].fillna("").str.contains(patt, case=False, regex=True)
    res["podium_flag"] = res["pos_num"].isin([1,2,3])

    grp = res.groupby("constructor", dropna=False)
    out = pd.DataFrame({
        "avg_finish": grp["pos_num"].mean(),
        "finish_std": grp["pos_num"].std(ddof=0),
        "podium_rate": grp["podium_flag"].mean(),
        "dnf_rate": grp["dnf_flag"].mean(),
        "starts": grp.size()
    }).reset_index().sort_values(["avg_finish","dnf_rate","finish_std"], ascending=[True, True, True])

    write_table(out, f"constructor_consistency_{y0}_{y1}")
    return out

def data_driver_teammate_delta(y0: int, y1: int) -> pd.DataFrame:
    """
    Driver teammate comparison: delta vs team median finish in each race.
    Positive delta = finished better than team median (lower position number).
    """
    if results is None or races is None or drivers is None:
        write_note("driver_teammate_delta", "Need results + races + drivers.")
        return pd.DataFrame()

    df = filter_year_span(results[["raceId","driverId","constructorId","position"]].copy(), y0, y1)
    df["pos_num"] = pd.to_numeric(df["position"], errors="coerce")
    df = df.dropna(subset=["pos_num"])

    # Median per race/team, then per-driver delta (no groupby.apply warnings)
    team_median = df.groupby(["raceId","constructorId"])["pos_num"].transform("median")
    df["delta_vs_team"] = team_median - df["pos_num"]
    df = df.merge(drivers[["driverId","driver"]], on="driverId", how="left")

    agg = (df.groupby("driver", as_index=False)
             .agg(avg_delta=("delta_vs_team","mean"),
                  med_delta=("delta_vs_team","median"),
                  races=("raceId","nunique"))
             .sort_values(["avg_delta","med_delta"], ascending=[False, False]))
    write_table(agg, f"driver_teammate_delta_{y0}_{y1}")
    return agg

def data_pit_stop_stats(y0: int, y1: int) -> pd.DataFrame:
    """
    Pit stop distribution per constructor:
    - team_avg_stops_per_race, car_median_stops_per_race
    """
    if pit_stops is None or races is None or results is None or constructors is None:
        write_note("pit_stop_stats", "Need pit_stops + races + results + constructors.")
        return pd.DataFrame()

    stops = filter_year_span(pit_stops[["raceId","driverId"]].copy(), y0, y1)
    car_race_stops = stops.groupby(["raceId","driverId"]).size().reset_index(name="stops")
    res = filter_year_span(results[["raceId","driverId","constructorId"]].copy(), y0, y1)
    joined = car_race_stops.merge(res, on=["raceId","driverId"], how="right")
    joined["stops"] = joined["stops"].fillna(0)

    team_race_avg = (joined.groupby(["constructorId","raceId"], as_index=False)["stops"].mean()
                            .groupby("constructorId", as_index=False)["stops"].mean()
                            .rename(columns={"stops":"team_avg_stops_per_race"}))
    car_median = (joined.groupby(["constructorId","raceId"], as_index=False)["stops"].median()
                         .groupby("constructorId", as_index=False)["stops"].median()
                         .rename(columns={"stops":"car_median_stops_per_race"}))

    out = (team_race_avg.merge(car_median, on="constructorId", how="outer")
                        .merge(constructors[["constructorId","constructor"]], on="constructorId", how="left")
                        .sort_values(["team_avg_stops_per_race","car_median_stops_per_race"]))
    write_table(out[["constructor","team_avg_stops_per_race","car_median_stops_per_race"]], f"pit_stop_stats_{y0}_{y1}")
    return out[["constructor","team_avg_stops_per_race","car_median_stops_per_race"]]

def data_qualifying_delta_to_field_best(y0: int, y1: int) -> pd.DataFrame:
    """
    Rough proxy: driver's best Q time vs field-best for that race (no sector data).
    """
    if qualifying is None or races is None or drivers is None:
        write_note("qualifying_ideal", "Need qualifying + races + drivers.")
        return pd.DataFrame()

    q = filter_year_span(qualifying.copy(), y0, y1)
    q_times = ["q1","q2","q3"]
    present = [c for c in q_times if c in q.columns]
    if not present:
        write_note("qualifying_ideal", "No q1/q2/q3 columns present.")
        return pd.DataFrame()

    def to_sec_any(t):
        if pd.isna(t): return np.nan
        s = str(t); parts = s.split(":")
        try:
            if len(parts) == 2: m, sec = parts; return float(m)*60 + float(sec)
            if len(parts) == 3: h, m, sec = parts; return float(h)*3600 + float(m)*60 + float(sec)
            return float(s)
        except Exception:
            return np.nan

    for c in present:
        q[c+"_s"] = q[c].apply(to_sec_any)
    q["best_actual_s"] = q[[c+"_s" for c in present]].min(axis=1, skipna=True)

    field = q.groupby("raceId", as_index=False)["best_actual_s"].min().rename(columns={"best_actual_s":"field_best_s"})
    out = q.merge(field, on="raceId", how="left")
    out["delta_to_field_best_s"] = out["best_actual_s"] - out["field_best_s"]
    out = out.merge(drivers[["driverId","driver"]], on="driverId", how="left")
    keep = ["raceId","driverId","driver","best_actual_s","field_best_s","delta_to_field_best_s"]
    out = out[keep].sort_values(["raceId","delta_to_field_best_s"])
    write_table(out, f"qualifying_delta_to_field_best_{y0}_{y1}")
    return out

def data_constructor_yearly_trend(y0: int, y1: int) -> pd.DataFrame:
    """
    Yearly constructor points (max per season) + simple CAGR-like trend.
    Robust: always merges year from races, then filters.
    """
    if constructor_standings is None or constructors is None or races is None:
        write_note("constructor_yearly_trend", "Need constructor_standings + constructors + races.")
        return pd.DataFrame()

    cs = constructor_standings[["raceId","constructorId","points"]].copy()
    if "raceId" in cs.columns and races is not None and "raceId" in races.columns:
        cs = cs.merge(races[["raceId","year"]], on="raceId", how="left")
    cs = filter_year_span(cs, y0, y1)

    if cs.empty or "year" not in cs.columns:
        write_note("constructor_yearly_trend", "No 'year' column available after merge/filter.")
        return pd.DataFrame()

    year_max = cs.groupby(["constructorId","year"], as_index=False)["points"].max()
    year_max = year_max.merge(constructors[["constructorId","constructor"]], on="constructorId", how="left")

    def small_cagr(g: pd.DataFrame) -> float:
        g = g.sort_values("year")
        if g.shape[0] <= 1:
            return np.nan
        start = g["points"].iloc[0]
        end   = g["points"].iloc[-1]
        n     = g.shape[0] - 1
        if not np.isfinite(start) or not np.isfinite(end) or start <= 0 or n <= 0:
            return np.nan
        try:
            return (end/start)**(1.0/n) - 1.0
        except Exception:
            return np.nan

    trend = (year_max.groupby("constructor", dropna=False)
                      .apply(lambda g: pd.Series({"cagr_like": small_cagr(g)}))
                      .reset_index())
    out = year_max.merge(trend, on="constructor", how="left").sort_values(["constructor","year"])
    write_table(out, f"constructor_yearly_trend_{y0}_{y1}")
    return out

# ===== Reference plots (sanity only) =====
def plot_reference_tables(y0: int, y1: int):
    t10 = data_top10_driver_points(y0, y1)
    if not t10.empty:
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(t10["driver"], t10["points_total"], color=ACCENT, alpha=0.9)
        ax.set_title(f"Top-10 Drivers by Points ({y0}-{y1})", loc="left", fontweight="bold")
        ax.set_xticklabels(t10["driver"], rotation=45, ha="right")
        ax.set_ylabel("Points")
        savefig(fig, "top10_driver_points")

    dnfs = data_dnfs_by_constructor(y0, y1)
    if not dnfs.empty:
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(dnfs["constructor"], dnfs["dnfs"], color=ACCENT, alpha=0.9)
        ax.set_title(f"DNFs by Constructor ({y0}-{y1})", loc="left", fontweight="bold")
        ax.set_xticklabels(dnfs["constructor"], rotation=45, ha="right")
        ax.set_ylabel("DNFs")
        savefig(fig, "dnfs_by_constructor")

    pc = data_points_by_country(y0, y1)
    if not pc.empty:
        pc_top = pc.head(15)
        fig, ax = plt.subplots(figsize=(5.5,6))
        ax.barh(pc_top["nationality"], pc_top["points_total"], color=ACCENT, alpha=0.9)
        ax.invert_yaxis()
        ax.set_title(f"Points by Nationality (Top 15, {y0}-{y1})", loc="left", fontweight="bold")
        ax.xaxis.set_major_locator(MaxNLocator(6))
        savefig(fig, "points_by_country_top15")

    fl = data_fastest_lap_by_season(y0, y1)
    if not fl.empty:
        fl = fl.sort_values("year")
        fig, ax = plt.subplots(figsize=(7,3.8))
        ax.plot(fl["year"], fl["fastest_s"], linewidth=2, color=ACCENT)
        ax.set_title(f"Fastest Lap by Season ({y0}-{y1})", loc="left", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Fastest Lap (s)")
        savefig(fig, "fastest_by_season")

    cc = data_constructor_consistency(y0, y1)
    if not cc.empty:
        fig, ax = plt.subplots(figsize=(6.5,4.5))
        ax.scatter(cc["avg_finish"], cc["dnf_rate"], s=(10 + 2*cc["starts"].fillna(0)), c=ACCENT, alpha=0.85)
        for _, r in cc.iterrows():
            ax.text(r["avg_finish"], r["dnf_rate"]+0.005, r["constructor"], fontsize=8, ha="center")
        ax.set_xlabel("Avg Finish (lower is better)")
        ax.set_ylabel("DNF Rate")
        ax.set_title(f"Constructor Consistency ({y0}-{y1})", loc="left", fontweight="bold")
        savefig(fig, "constructor_consistency_scatter")

# ===== FastF1 enrichment =====
def setup_fastf1(cache_dir: Optional[Path] = None):
    if not FASTF1_AVAILABLE:
        print("[fastf1] Not installed; skip FastF1 enrichment.")
        return False
    cdir = cache_dir or Path(os.getenv("FASTF1_CACHE", "") or (ROOT / "fastf1_cache"))
    cdir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cdir))
    try:
        f1plot.setup_mpl(misc_mpl_mods=False)
    except Exception:
        pass
    return True

def ff1_session_summary(year: int, gp_name: str = "Bahrain Grand Prix", session: str = "R"):
    try:
        ses = fastf1.get_session(year, gp_name, session)
        ses.load()
    except Exception as e:
        write_note(f"fastf1_{year}_{gp_name}_{session}", f"Failed to load FastF1 session: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    laps = ses.laps
    if laps is None or laps.empty:
        write_note(f"fastf1_{year}_{gp_name}_{session}", "No laps dataframe.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    stints = laps[["Driver","Stint","Compound","LapNumber","PitOutTime","PitInTime"]].copy()
    stint_agg = (stints.groupby(["Driver","Stint","Compound"], as_index=False)
                       .agg(laps=("LapNumber","count")))
    write_table(stint_agg, f"ff1_{year}_stints_{gp_name.replace(' ','_')}")

    def td_to_s(td):
        try: return td.total_seconds()
        except Exception: return np.nan

    clean = laps[(laps["LapTime"].notna()) &
                 (~laps["PitInTime"].notna()) &
                 (~laps["PitOutTime"].notna())].copy()
    clean["lap_s"] = clean["LapTime"].apply(td_to_s)
    clean = clean.dropna(subset=["lap_s","Compound"])

    pace = (clean.groupby(["Driver","Compound"], as_index=False)["lap_s"]
                 .median()
                 .rename(columns={"lap_s":"median_lap_s"})
                 .sort_values(["Compound","median_lap_s"]))
    write_table(pace, f"ff1_{year}_median_pace_by_compound_{gp_name.replace(' ','_')}")

    clean = clean.sort_values(["Driver","Stint","LapNumber"])
    clean["stint_rel_idx"] = clean.groupby(["Driver","Stint"]).cumcount() + 1

    def fit_slope(group: pd.DataFrame) -> float:
        x = group["stint_rel_idx"].to_numpy(dtype=float)
        y = group["lap_s"].to_numpy(dtype=float)
        if x.size < 3: return np.nan
        X = np.vstack([np.ones_like(x), x]).T
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return float(beta[1])
        except Exception:
            return np.nan

    deg = (clean.groupby(["Driver","Compound"])
                 .apply(lambda g: fit_slope(g))
                 .reset_index(name="deg_s_per_lap"))
    write_table(deg, f"ff1_{year}_degradation_{gp_name.replace(' ','_')}")

    try:
        comp_order = pace.groupby("Compound", as_index=False)["median_lap_s"].median().sort_values("median_lap_s")["Compound"]
        fig, ax = plt.subplots(figsize=(7,4))
        data_plot = (pace.set_index("Compound").loc[comp_order].reset_index())
        ax.bar(data_plot["Compound"], data_plot["median_lap_s"], color=ACCENT, alpha=0.9)
        ax.set_title(f"{year} {gp_name} – Median Lap by Compound", loc="left", fontweight="bold")
        ax.set_ylabel("Lap Time (s)")
        savefig(fig, f"ff1_{year}_{gp_name.replace(' ','_')}_median_lap_compound")
    except Exception:
        pass

    return stint_agg, pace, deg

# ===== Orchestration =====
def run_ergast_block(y0: int, y1: int, with_plots: bool = True, extras: str = "none"):
    print(f"[ergast] Analyzing {y0}-{y1} from: {DATA_DIR}")
    tc  = data_top_constructor_points(y0, y1)
    td  = data_top_driver_podiums(y0, y1)
    ap  = data_avg_pit_stops_per_race(y0, y1)
    tl  = data_total_laps(y0, y1)
    hs  = data_highest_speed(y0, y1)
    t10 = data_top10_driver_points(y0, y1)
    d_drv, d_team = data_avg_finish_positions(y0, y1)
    dnfs = data_dnfs_by_constructor(y0, y1)
    pcn  = data_points_by_country(y0, y1)
    fl   = data_fastest_lap_by_season(y0, y1)

    if extras in ("all","consistency","teammate","pits","qual","trend"):
        if extras in ("all","consistency"): _ = data_constructor_consistency(y0, y1)
        if extras in ("all","teammate"):    _ = data_driver_teammate_delta(y0, y1)
        if extras in ("all","pits"):        _ = data_pit_stop_stats(y0, y1)
        if extras in ("all","qual"):        _ = data_qualifying_delta_to_field_best(y0, y1)
        if extras in ("all","trend"):       _ = data_constructor_yearly_trend(y0, y1)

    if with_plots:
        plot_reference_tables(y0, y1)

    def head_row(df: pd.DataFrame) -> Dict:
        return df.iloc[0].to_dict() if isinstance(df, pd.DataFrame) and not df.empty else {}

    print("Top constructor:", head_row(tc))
    print("Top driver by podiums:", head_row(td))
    print("Highest recorded speed:", head_row(hs))
    if not ap.empty:
        try:
            print("Avg pit stops per race:", round(float(ap.iloc[0,0]), 2))
        except Exception:
            pass

def run_fastf1_block(years: Iterable[int], gp_name: str, session: str = "R"):
    if not setup_fastf1():
        return
    for yr in years:
        print(f"[fastf1] {yr} {gp_name} {session} …")
        stint, pace, deg = ff1_session_summary(yr, gp_name=gp_name, session=session)
        if not pace.empty:
            best = (pace.sort_values(["Driver","median_lap_s"])
                         .groupby("Driver", as_index=False).first())
            write_table(best, f"ff1_{yr}_best_compound_per_driver_{gp_name.replace(' ','_')}")
            print(f"  Saved FastF1 tables for {yr} {gp_name}.")

# ===== CLI =====
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="F1 analysis pipeline (CSV + optional FastF1).")
    p.add_argument("--y0", type=int, default=YEAR_MIN, help=f"Start year (default {YEAR_MIN})")
    p.add_argument("--y1", type=int, default=YEAR_MAX, help=f"End year (default {YEAR_MAX})")
    p.add_argument("--no-plots", action="store_true", help="Skip reference plots.")
    p.add_argument("--use-fastf1", action="store_true", help="Fetch FastF1 data for a session.")
    p.add_argument("--ff1-years", type=str, default="", help="Comma list of years for FastF1 (e.g., 2022,2023). Defaults to y1 only.")
    p.add_argument("--ff1-gp", type=str, default="Bahrain Grand Prix", help='GP name for FastF1, e.g., "Bahrain Grand Prix".')
    p.add_argument("--ff1-session", type=str, default="R", help='Session: "R" (race), "Q", "FP1", etc.')
    p.add_argument("--extras", type=str, default="none",
                   choices=["none","consistency","teammate","pits","qual","trend","all"],
                   help="Run additional intermediate analyses.")
    return p

def main():
    args = build_parser().parse_args()
    y0, y1 = int(args.y0), int(args.y1)
    run_ergast_block(y0=y0, y1=y1, with_plots=(not args.no_plots), extras=args.extras)

    if args.use_fastf1 and FASTF1_AVAILABLE:
        if args.ff1_years.strip():
            years = [int(x.strip()) for x in args.ff1_years.split(",") if x.strip()]
        else:
            years = [y1]
        run_fastf1_block(years=years, gp_name=args.ff1_gp, session=args.ff1_session)
    elif args.use_fastf1 and not FASTF1_AVAILABLE:
        print("fastf1 not installed; run: pip install fastf1")

if __name__ == "__main__":
    main()
