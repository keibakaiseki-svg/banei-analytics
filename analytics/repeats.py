"""
同一馬の連続出走ペアを抽出し、タイム変化・着順変化の要因を分析する。

主要関数:
- consecutive_pairs : 連続2走ペアの DataFrame を返す
- time_change_decomposition : タイム変化の説明変数別影響度
- jockey_change_impact : 騎手交代有無での比較
- body_weight_impact : 馬体重変化帯別の比較
- past_performance_predictor : 過去N走平均と次走着順の相関
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from persist.parquet_writer import DEFAULT_PARQUET_ROOT


def open_connection(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in ("races", "entries", "payouts"):
        glob = str(parquet_root / t / "**/*.parquet")
        con.execute(
            f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
        )
    return con


def appearance_distribution(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """馬ごとの出走回数分布。連続ペア数の見積もりに使う。"""
    return con.execute(
        """
        WITH counts AS (
            SELECT e.horse_id, COUNT(*) AS appearances
            FROM entries e WHERE e.horse_id IS NOT NULL AND e.finish_pos IS NOT NULL
            GROUP BY e.horse_id
        )
        SELECT appearances, COUNT(*) AS horses
        FROM counts GROUP BY appearances ORDER BY appearances
        """
    ).fetchdf()


def consecutive_pairs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """連続する2走を1行にまとめたペア DataFrame。
    完走したエントリのみ。"""
    return con.execute(
        """
        WITH ranked AS (
            SELECT
                e.horse_id, e.horse_name, r.race_date, r.race_id,
                r.race_class, r.track_water_pct,
                e.jockey_id, e.jockey_name,
                e.body_weight_kg, e.body_weight_diff_kg,
                e.load_weight_kg,
                e.finish_pos, e.finish_time_sec, e.popularity,
                ROW_NUMBER() OVER (PARTITION BY e.horse_id ORDER BY r.race_date, r.race_no) AS run_no
            FROM entries e
            JOIN races r USING (race_id)
            WHERE e.finish_pos IS NOT NULL AND e.horse_id IS NOT NULL
              AND e.finish_status IN ('finished', 'dead_heat', 'demoted')
        )
        SELECT
            a.horse_id, a.horse_name,
            a.race_date AS prev_date, b.race_date AS curr_date,
            DATE_DIFF('day', a.race_date::DATE, b.race_date::DATE) AS days_between,
            a.race_class AS prev_class, b.race_class AS curr_class,
            (a.race_class != b.race_class) AS class_changed,
            a.track_water_pct AS prev_water, b.track_water_pct AS curr_water,
            (b.track_water_pct - a.track_water_pct) AS water_change,
            a.jockey_name AS prev_jockey, b.jockey_name AS curr_jockey,
            (a.jockey_id != b.jockey_id) AS jockey_changed,
            a.body_weight_kg AS prev_body, b.body_weight_kg AS curr_body,
            (b.body_weight_kg - a.body_weight_kg) AS body_diff,
            a.load_weight_kg AS prev_load, b.load_weight_kg AS curr_load,
            (b.load_weight_kg - a.load_weight_kg) AS load_change,
            a.finish_time_sec AS prev_time, b.finish_time_sec AS curr_time,
            (b.finish_time_sec - a.finish_time_sec) AS time_change,
            a.finish_pos AS prev_pos, b.finish_pos AS curr_pos,
            (b.finish_pos - a.finish_pos) AS pos_change,
            a.popularity AS prev_pop, b.popularity AS curr_pop
        FROM ranked a
        JOIN ranked b ON a.horse_id = b.horse_id AND b.run_no = a.run_no + 1
        """
    ).fetchdf()


def correlation_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    """タイム変化を目的変数として、変化変数とのピアソン相関を返す。"""
    if pairs.empty:
        return pd.DataFrame()
    cols = ["time_change", "body_diff", "water_change", "load_change", "days_between", "pos_change"]
    sub = pairs[cols].dropna()
    if len(sub) < 3:
        return pd.DataFrame({"warning": [f"sample size {len(sub)} too small"]})
    return sub.corr()[["time_change"]].rename(columns={"time_change": "corr_with_time_change"})


def jockey_change_impact(pairs: pd.DataFrame) -> pd.DataFrame:
    """騎手交代の有無別: タイム変化・着順変化のサマリ。"""
    if pairs.empty:
        return pd.DataFrame()
    sub = pairs.dropna(subset=["time_change", "pos_change", "jockey_changed"])
    if len(sub) < 3:
        return pd.DataFrame({"warning": [f"sample size {len(sub)} too small"]})
    return sub.groupby("jockey_changed").agg(
        n=("time_change", "count"),
        time_change_mean=("time_change", "mean"),
        time_change_median=("time_change", "median"),
        time_change_std=("time_change", "std"),
        pos_change_mean=("pos_change", "mean"),
        pos_change_median=("pos_change", "median"),
    ).round(2)


def body_weight_band_impact(pairs: pd.DataFrame) -> pd.DataFrame:
    """馬体重変化帯別: タイム・着順変化のサマリ。"""
    if pairs.empty:
        return pd.DataFrame()
    sub = pairs.dropna(subset=["time_change", "pos_change", "body_diff"]).copy()
    if len(sub) < 5:
        return pd.DataFrame({"warning": [f"sample size {len(sub)} too small"]})

    def band(x):
        if x <= -15: return "A_減15kg以上"
        if x <= -5:  return "B_減5〜15kg"
        if x <  5:   return "C_±5kg以内"
        if x <  15:  return "D_増5〜15kg"
        return "E_増15kg以上"

    sub["band"] = sub["body_diff"].apply(band)
    return sub.groupby("band").agg(
        n=("time_change", "count"),
        time_change_mean=("time_change", "mean"),
        time_change_median=("time_change", "median"),
        pos_change_mean=("pos_change", "mean"),
        pos_change_median=("pos_change", "median"),
        top3_rate=("curr_pos", lambda s: (s <= 3).mean()),
    ).round(3)


def water_change_impact(pairs: pd.DataFrame) -> pd.DataFrame:
    """馬場水分量の変化幅別: タイム変化(net)のサマリ。"""
    if pairs.empty:
        return pd.DataFrame()
    sub = pairs.dropna(subset=["time_change", "water_change"]).copy()
    if len(sub) < 5:
        return pd.DataFrame({"warning": [f"sample size {len(sub)} too small"]})

    def band(x):
        if x <= -1.0: return "A_乾く方向-1.0以上"
        if x <  0:    return "B_少し乾く"
        if x == 0:    return "C_変化なし"
        if x <  1.0:  return "D_少し湿る"
        return "E_湿る方向+1.0以上"

    sub["band"] = sub["water_change"].apply(band)
    return sub.groupby("band").agg(
        n=("time_change", "count"),
        time_change_mean=("time_change", "mean"),
        time_change_median=("time_change", "median"),
    ).round(2)


def past_n_predictor(con: duckdb.DuckDBPyConnection, n: int = 3) -> pd.DataFrame:
    """過去N走の平均着順 × 当該レース着順の相関 (順序数値として)。"""
    df = con.execute(
        f"""
        WITH ranked AS (
            SELECT
                e.horse_id, r.race_date, r.race_no, e.finish_pos,
                ROW_NUMBER() OVER (PARTITION BY e.horse_id ORDER BY r.race_date, r.race_no) AS run_no
            FROM entries e
            JOIN races r USING (race_id)
            WHERE e.finish_pos IS NOT NULL AND e.horse_id IS NOT NULL
              AND e.finish_status IN ('finished', 'dead_heat', 'demoted')
        ),
        with_past AS (
            SELECT
                c.horse_id, c.finish_pos AS curr_pos,
                AVG(p.finish_pos) AS past_avg_pos,
                COUNT(*) AS past_count
            FROM ranked c
            JOIN ranked p
              ON c.horse_id = p.horse_id
             AND p.run_no >= c.run_no - {n}
             AND p.run_no <  c.run_no
            GROUP BY c.horse_id, c.run_no, c.finish_pos
            HAVING COUNT(*) = {n}
        )
        SELECT * FROM with_past
        """
    ).fetchdf()
    if df.empty or len(df) < 5:
        return pd.DataFrame({"warning": [f"sample size {len(df)} too small for past_{n}"]})
    corr = df[["curr_pos", "past_avg_pos"]].corr().iloc[0, 1]
    summary = pd.DataFrame({
        "metric": [f"past_{n}_avg_pos vs curr_pos correlation"],
        "sample_size": [len(df)],
        "pearson_r": [round(corr, 3)],
    })
    return summary


def full_report(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> dict:
    """全分析を一括実行して dict にまとめる。"""
    con = open_connection(parquet_root)
    pairs = consecutive_pairs(con)
    return {
        "appearance_distribution": appearance_distribution(con),
        "pairs_count": len(pairs),
        "correlation_matrix": correlation_matrix(pairs),
        "jockey_change_impact": jockey_change_impact(pairs),
        "body_weight_band_impact": body_weight_band_impact(pairs),
        "water_change_impact": water_change_impact(pairs),
        "past_3_predictor": past_n_predictor(con, 3),
        "past_5_predictor": past_n_predictor(con, 5),
    }


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 30)
    report = full_report()
    for k, v in report.items():
        print(f"=== {k} ===")
        print(v)
        print()
