"""
各馬の馬場適性 (水分量別パフォーマンス) を分析・分類する。

スピード指数 (speed_figure) を水分量帯別に集計し、以下に分類:

  - universal_good     : 全bandで平均SFが+3秒以上 (絶対能力高い)
  - universal_poor     : 全bandで平均SFが-3秒以下
  - dry_specialist     : dry band の SF が他bandより+5秒以上高い
  - wet_specialist     : wet band の SF が他bandより+5秒以上高い
  - moist_specialist   : moist band 特化
  - weak_in_dry        : dry band で-5秒以下
  - weak_in_wet        : wet band で-5秒以下
  - mixed              : 上記いずれにも該当しない

水分量帯定義 (3分割):
  - dry   : track_water_pct < 2.0
  - normal: 2.0 <= track_water_pct < 3.0
  - wet   : track_water_pct >= 3.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from persist.parquet_writer import DEFAULT_PARQUET_ROOT


def open_connection(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in ("races", "entries"):
        glob = str(parquet_root / t / "**/*.parquet")
        con.execute(
            f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
        )
    return con


def _build_speed_figures(con: duckdb.DuckDBPyConnection) -> None:
    """(class × water_band) 別基準タイム + 日次バリアント + スピード指数を一時テーブルに格納。"""
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE races_w AS
        SELECT *,
            CASE
                WHEN track_water_pct < 2.0 THEN 'dry'
                WHEN track_water_pct < 3.0 THEN 'normal'
                ELSE 'wet'
            END AS water_band
        FROM races
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE baselines AS
        SELECT r.race_class, r.water_band, MEDIAN(e.finish_time_sec) AS expected_time
        FROM races_w r JOIN entries e USING (race_id)
        WHERE e.finish_pos IS NOT NULL AND e.finish_time_sec IS NOT NULL AND r.race_class IS NOT NULL
        GROUP BY r.race_class, r.water_band
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE variants AS
        SELECT r.race_date, MEDIAN(e.finish_time_sec - b.expected_time) AS variant_sec
        FROM races_w r
        JOIN entries e USING (race_id)
        JOIN baselines b ON b.race_class = r.race_class AND b.water_band = r.water_band
        WHERE e.finish_pos IS NOT NULL AND e.finish_time_sec IS NOT NULL
        GROUP BY r.race_date
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE entry_sf AS
        SELECT
            e.horse_id, e.horse_name, e.horse_no,
            r.race_date, r.race_class, r.water_band, r.track_water_pct,
            e.load_weight_kg, e.body_weight_kg,
            e.finish_pos, e.finish_time_sec,
            ROUND(b.expected_time - (e.finish_time_sec - v.variant_sec), 2) AS speed_figure
        FROM races_w r
        JOIN entries e USING (race_id)
        LEFT JOIN baselines b ON b.race_class = r.race_class AND b.water_band = r.water_band
        LEFT JOIN variants v ON v.race_date = r.race_date
        WHERE e.finish_pos IS NOT NULL
          AND e.finish_time_sec IS NOT NULL
          AND e.horse_id IS NOT NULL
        """
    )


def horse_band_profile(
    con: duckdb.DuckDBPyConnection, min_total_runs: int = 10
) -> pd.DataFrame:
    """馬ごと×水分量帯のスピード指数プロファイル。"""
    return con.execute(
        f"""
        WITH overall AS (
            SELECT horse_id, COUNT(*) AS total_runs,
                   ROUND(AVG(speed_figure), 2) AS overall_mean_sf
            FROM entry_sf WHERE speed_figure IS NOT NULL
            GROUP BY horse_id
        ),
        by_band AS (
            SELECT horse_id, water_band, COUNT(*) AS n,
                   ROUND(AVG(speed_figure), 2) AS mean_sf,
                   ROUND(STDDEV(speed_figure), 2) AS std_sf
            FROM entry_sf WHERE speed_figure IS NOT NULL
            GROUP BY horse_id, water_band
        )
        SELECT b.horse_id,
               (SELECT ANY_VALUE(horse_name) FROM entry_sf WHERE entry_sf.horse_id = b.horse_id) AS horse_name,
               o.total_runs,
               o.overall_mean_sf,
               b.water_band,
               b.n,
               b.mean_sf,
               b.std_sf,
               ROUND(b.mean_sf - o.overall_mean_sf, 2) AS delta_vs_overall
        FROM by_band b
        JOIN overall o USING (horse_id)
        WHERE o.total_runs >= {min_total_runs}
        ORDER BY b.horse_id, b.water_band
        """
    ).fetchdf()


def classify_horses(
    profile_df: pd.DataFrame,
    min_band_runs: int = 5,
    universal_threshold: float = 3.0,
    specialist_threshold: float = 5.0,
) -> pd.DataFrame:
    """プロファイルから各馬を分類。"""
    rows = []
    for horse_id, g in profile_df.groupby("horse_id"):
        horse_name = g["horse_name"].iloc[0]
        overall = g["overall_mean_sf"].iloc[0]
        total = g["total_runs"].iloc[0]

        # 各 band 別 mean_sf と n
        band_stats = {row["water_band"]: (row["n"], row["mean_sf"], row["delta_vs_overall"])
                      for _, row in g.iterrows()}

        # 十分なサンプルがある band のみ評価対象
        valid_bands = {b: s for b, s in band_stats.items() if s[0] >= min_band_runs}
        if len(valid_bands) < 2:
            classification = "insufficient_band_samples"
        else:
            # 普遍性チェック (全ての band で平均 mean_sf が高い/低い)
            mean_sfs = [s[1] for s in valid_bands.values()]
            if all(m > universal_threshold for m in mean_sfs):
                classification = "universal_good"
            elif all(m < -universal_threshold for m in mean_sfs):
                classification = "universal_poor"
            else:
                # band 特化チェック
                classification = "mixed"
                # dry specialist : dry band が他より +specialist_threshold 以上
                bands = list(valid_bands.keys())
                for target in ["dry", "wet", "normal"]:
                    if target not in valid_bands:
                        continue
                    target_mean = valid_bands[target][1]
                    others = [valid_bands[b][1] for b in valid_bands if b != target]
                    if all(target_mean - o >= specialist_threshold for o in others):
                        classification = f"{target}_specialist"
                        break
                # 苦手 band チェック
                if classification == "mixed":
                    for target in ["dry", "wet", "normal"]:
                        if target not in valid_bands:
                            continue
                        target_mean = valid_bands[target][1]
                        others = [valid_bands[b][1] for b in valid_bands if b != target]
                        if all(o - target_mean >= specialist_threshold for o in others):
                            classification = f"weak_in_{target}"
                            break

        rows.append({
            "horse_id": horse_id,
            "horse_name": horse_name,
            "total_runs": total,
            "overall_mean_sf": overall,
            "n_dry": band_stats.get("dry", (0, None, None))[0],
            "sf_dry": band_stats.get("dry", (0, None, None))[1],
            "n_normal": band_stats.get("normal", (0, None, None))[0],
            "sf_normal": band_stats.get("normal", (0, None, None))[1],
            "n_wet": band_stats.get("wet", (0, None, None))[0],
            "sf_wet": band_stats.get("wet", (0, None, None))[1],
            "classification": classification,
        })
    return pd.DataFrame(rows)


def summary(class_df: pd.DataFrame) -> pd.DataFrame:
    """分類別の馬数サマリ。"""
    return (
        class_df.groupby("classification")
        .agg(n_horses=("horse_id", "count"), avg_overall_sf=("overall_mean_sf", "mean"))
        .round(2)
        .sort_values("n_horses", ascending=False)
        .reset_index()
    )


def materialize(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """馬場適性プロファイルを Parquet に永続化。"""
    con = open_connection(parquet_root)
    _build_speed_figures(con)
    profile = horse_band_profile(con)
    classification = classify_horses(profile)
    profile_path = parquet_root / "horse_water_profile.parquet"
    class_path = parquet_root / "horse_water_classification.parquet"
    profile.to_parquet(profile_path, index=False, compression="zstd")
    classification.to_parquet(class_path, index=False, compression="zstd")
    return profile, classification


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 50)

    profile, classification = materialize()
    print(f"=== 馬数 (10走以上の馬のみ) ===")
    print(f"  {classification['horse_id'].nunique():,} 馬")
    print()
    print("=== 分類別サマリ ===")
    print(summary(classification).to_string(index=False))

    print()
    print("=== A. universal_good サンプル (TOP 5・絶対能力上位) ===")
    print(classification[classification["classification"] == "universal_good"]
          .sort_values("overall_mean_sf", ascending=False).head(5).to_string(index=False))

    print()
    print("=== C. dry_specialist サンプル (上位5・軽馬場巧者) ===")
    print(classification[classification["classification"] == "dry_specialist"]
          .sort_values("sf_dry", ascending=False).head(5).to_string(index=False))

    print()
    print("=== D. wet_specialist サンプル (上位5・重馬場巧者) ===")
    print(classification[classification["classification"] == "wet_specialist"]
          .sort_values("sf_wet", ascending=False).head(5).to_string(index=False))

    print()
    print("=== B. universal_poor サンプル (5・走らない馬) ===")
    print(classification[classification["classification"] == "universal_poor"]
          .sort_values("overall_mean_sf").head(5).to_string(index=False))

    print()
    print("=== E. weak_in_wet サンプル (5・重馬場苦手) ===")
    weak = classification[classification["classification"] == "weak_in_wet"]
    if not weak.empty:
        print(weak.sort_values("sf_wet").head(5).to_string(index=False))
    else:
        print("該当なし")
