"""
日次トラックバリアントと馬個別スピード指数の計算。

設計:
1. 各レースに「期待タイム」を計算
   = 同じ (クラス, 水分量帯) の長期中央値
2. 各レース日に「日次バリアント」を計算
   = 当日全レースの (観測タイム - 期待タイム) の中央値
3. 馬個別のスピード指数を計算
   = 期待タイム - (観測タイム - 日次バリアント)
   = 「補正済み実タイム」が期待タイムからどれだけ速いか (正の値が高評価)

Walk-Forward 互換:
  各時点で、その時点までの過去データのみから期待タイムを計算する設計。
  バックテスト時はカットオフ日を指定して fit_baselines() を呼ぶ。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from persist.parquet_writer import DEFAULT_PARQUET_ROOT


def _water_band(pct: float) -> str:
    """水分量帯を5段階に区分。"""
    if pct < 1.0:
        return "0_dry"
    if pct < 2.0:
        return "1_normal_dry"
    if pct < 3.0:
        return "2_normal_wet"
    if pct < 4.0:
        return "3_moist"
    return "4_wet"


def _attach_water_band(con: duckdb.DuckDBPyConnection) -> None:
    """races ビューに water_band 列を追加した派生ビュー races_w を作成。"""
    con.execute(
        """
        CREATE OR REPLACE VIEW races_w AS
        SELECT *,
            CASE
                WHEN track_water_pct < 1.0 THEN '0_dry'
                WHEN track_water_pct < 2.0 THEN '1_normal_dry'
                WHEN track_water_pct < 3.0 THEN '2_normal_wet'
                WHEN track_water_pct < 4.0 THEN '3_moist'
                ELSE '4_wet'
            END AS water_band
        FROM races
        """
    )


def open_connection(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in ("races", "entries", "payouts"):
        glob = str(parquet_root / t / "**/*.parquet")
        con.execute(
            f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
        )
    _attach_water_band(con)
    return con


def fit_baselines(
    con: duckdb.DuckDBPyConnection,
    cutoff_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    (race_class, water_band) ごとの長期基準タイムを返す。
    完走馬の finish_time の中央値を基準とする (外れ値に頑健)。
    """
    where_clause = f"AND r.race_date < '{cutoff_date}'" if cutoff_date else ""
    return con.execute(
        f"""
        SELECT
            r.race_class,
            r.water_band,
            COUNT(*) AS n,
            ROUND(MEDIAN(e.finish_time_sec), 2) AS median_time_sec,
            ROUND(AVG(e.finish_time_sec), 2) AS mean_time_sec,
            ROUND(STDDEV_POP(e.finish_time_sec), 2) AS std_time_sec
        FROM races_w r
        JOIN entries e USING (race_id)
        WHERE e.finish_pos IS NOT NULL
          AND e.finish_time_sec IS NOT NULL
          AND r.race_class IS NOT NULL
          {where_clause}
        GROUP BY r.race_class, r.water_band
        ORDER BY r.race_class, r.water_band
        """
    ).fetchdf()


def fit_daily_variants(
    con: duckdb.DuckDBPyConnection,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    """
    各レース日の「日次バリアント」を算出。
    = 当日全エントリの (観測タイム - 期待タイム) の中央値
    """
    # baselines を一時テーブルに展開
    con.register("baselines", baselines)
    df = con.execute(
        """
        WITH joined AS (
            SELECT
                r.race_date,
                r.race_class,
                r.water_band,
                e.finish_time_sec,
                b.median_time_sec AS expected_time
            FROM races_w r
            JOIN entries e USING (race_id)
            JOIN baselines b
              ON b.race_class = r.race_class
             AND b.water_band = r.water_band
            WHERE e.finish_pos IS NOT NULL
              AND e.finish_time_sec IS NOT NULL
        )
        SELECT
            race_date,
            COUNT(*) AS n_runs,
            ROUND(MEDIAN(finish_time_sec - expected_time), 2) AS variant_sec
        FROM joined
        GROUP BY race_date
        ORDER BY race_date
        """
    ).fetchdf()
    con.unregister("baselines")
    return df


def compute_speed_figures(
    con: duckdb.DuckDBPyConnection,
    baselines: pd.DataFrame,
    variants: pd.DataFrame,
) -> pd.DataFrame:
    """
    各エントリの スピード指数 を算出。
        speed_figure = expected_time - (observed_time - day_variant)
    正の値が大きいほど高評価 (基準より速かった)。
    """
    con.register("baselines", baselines)
    con.register("variants", variants[["race_date", "variant_sec"]])
    df = con.execute(
        """
        SELECT
            e.race_id,
            e.horse_id,
            e.horse_no,
            r.race_date,
            r.race_class,
            r.water_band,
            r.track_water_pct,
            e.finish_pos,
            e.finish_time_sec,
            b.median_time_sec AS expected_time,
            v.variant_sec     AS day_variant,
            ROUND(
              b.median_time_sec - (e.finish_time_sec - v.variant_sec),
              2
            ) AS speed_figure
        FROM races_w r
        JOIN entries e USING (race_id)
        JOIN baselines b
          ON b.race_class = r.race_class
         AND b.water_band = r.water_band
        JOIN variants v ON v.race_date = r.race_date
        WHERE e.finish_pos IS NOT NULL
          AND e.finish_time_sec IS NOT NULL
        """
    ).fetchdf()
    con.unregister("baselines")
    con.unregister("variants")
    return df


def speed_figure_predictive_power(figures: pd.DataFrame) -> pd.DataFrame:
    """
    過去N走の平均スピード指数と当該レース着順の相関を計算。
    """
    fig = figures.dropna(subset=["speed_figure", "horse_id"]).copy()
    fig = fig.sort_values(["horse_id", "race_date", "race_id"]).reset_index(drop=True)
    fig["run_no"] = fig.groupby("horse_id").cumcount()

    results = []
    for n in [1, 3, 5, 10]:
        # 各馬の過去N走スピード指数の平均を、現走の予測子として算出
        fig[f"past_{n}_sf"] = (
            fig.groupby("horse_id")["speed_figure"]
            .shift(1)
            .rolling(n, min_periods=n)
            .mean()
            .reset_index(level=0, drop=True)
        )
        sub = fig.dropna(subset=[f"past_{n}_sf", "finish_pos"])
        if len(sub) < 100:
            results.append({"past_n": n, "sample_size": len(sub), "pearson_r": None})
            continue
        # 現走着順 vs 過去N走平均SF (低着順=好成績、SF高=好成績、なので負相関期待)
        corr = sub[[f"past_{n}_sf", "finish_pos"]].corr().iloc[0, 1]
        results.append({"past_n": n, "sample_size": len(sub), "pearson_r": round(corr, 3)})
    return pd.DataFrame(results)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    con = open_connection()
    print("=== 1. (クラス × 水分量帯) 別の基準タイム ===")
    baselines = fit_baselines(con)
    print(baselines.to_string(index=False))

    print()
    print("=== 2. 日次バリアントの分布 ===")
    variants = fit_daily_variants(con, baselines)
    print(f"日数: {len(variants)}")
    print(f"バリアント分布: min={variants['variant_sec'].min()} median={variants['variant_sec'].median()} max={variants['variant_sec'].max()}")
    print(f"|バリアント|>=5秒の日: {(variants['variant_sec'].abs() >= 5).sum()}日")
    print(f"|バリアント|>=10秒の日: {(variants['variant_sec'].abs() >= 10).sum()}日")
    print()
    print("極端な日 (上位10):")
    print(variants.reindex(variants["variant_sec"].abs().sort_values(ascending=False).index).head(10).to_string(index=False))

    print()
    print("=== 3. スピード指数の予測力 (過去N走平均 → 現走着順) ===")
    figures = compute_speed_figures(con, baselines, variants)
    print(speed_figure_predictive_power(figures).to_string(index=False))

    print()
    print("=== 4. スピード指数 分布 ===")
    print(figures["speed_figure"].describe().round(2).to_string())
