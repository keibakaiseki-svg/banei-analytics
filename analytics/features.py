"""
予測モデル用の特徴量行列を構築する。

設計原則:
- データリーケージ防止: 各エントリの特徴量は「そのレース時点までの情報」のみで計算
- 馬個別の lag 特徴量は LAG/ROW_NUMBER で算出 (馬の自身の過去のみ参照)
- 集計系特徴量 (クラス×水分量帯の平均タイム等) は将来的に walk-forward で再計算
  今は計算コスト削減のためフルデータの値を使う (実装簡略化)
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
    return con


def build_feature_matrix(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> pd.DataFrame:
    """
    全エントリの特徴量行列を構築。各行 = 1エントリ(出走馬)。
    target_win: 1 = 1着, 0 = それ以外。
    target_top3: 1 = 1-3着, 0 = それ以外。

    Lag 特徴量:
      - prev_finish_pos_1/2/3: 過去N走前の着順
      - prev_speed_figure_1/2/3: 過去N走前のスピード指数
      - past_3_avg_pos, past_5_avg_pos
      - past_3_avg_sf (speed figure)
      - days_since_last_run
      - lifetime_starts (除く当該レース)
      - lifetime_wins (除く当該レース)
      - lifetime_top3_rate
    """
    con = open_connection(parquet_root)

    # まず (race_class, water_band) × 期待タイム(median) テーブルを作る
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE baselines AS
        SELECT
            r.race_class,
            r.water_band,
            MEDIAN(e.finish_time_sec) AS expected_time_sec
        FROM races_w r
        JOIN entries e USING (race_id)
        WHERE e.finish_pos IS NOT NULL
          AND e.finish_time_sec IS NOT NULL
          AND r.race_class IS NOT NULL
        GROUP BY r.race_class, r.water_band
        """
    )

    # 日次バリアント
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE day_variants AS
        WITH joined AS (
            SELECT
                r.race_date,
                e.finish_time_sec,
                b.expected_time_sec
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
            MEDIAN(finish_time_sec - expected_time_sec) AS day_variant_sec
        FROM joined
        GROUP BY race_date
        """
    )

    # スピード指数を含む base 行
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE enriched AS
        SELECT
            e.race_id,
            r.race_date,
            r.race_no,
            r.race_class,
            r.water_band,
            r.track_water_pct,
            r.weather,
            r.entry_count,
            e.horse_id,
            e.horse_no,
            e.post_position,
            e.horse_name,
            e.sex_age,
            SUBSTR(e.sex_age, 1, 1) AS sex,
            CAST(REGEXP_EXTRACT(e.sex_age, '[0-9]+') AS INTEGER) AS age,
            e.load_weight_kg,
            e.body_weight_kg,
            e.body_weight_diff_kg,
            e.jockey_id,
            e.jockey_name,
            e.allowance_marker,
            (e.allowance_marker IS NOT NULL) AS has_allowance,
            e.trainer_id,
            e.popularity,
            e.finish_pos,
            e.finish_time_sec,
            e.finish_status,
            b.expected_time_sec,
            v.day_variant_sec,
            CASE WHEN e.finish_time_sec IS NOT NULL AND b.expected_time_sec IS NOT NULL AND v.day_variant_sec IS NOT NULL
                THEN ROUND(b.expected_time_sec - (e.finish_time_sec - v.day_variant_sec), 2)
                ELSE NULL
            END AS speed_figure,
            CASE WHEN e.finish_pos = 1 THEN 1 ELSE 0 END AS target_win,
            CASE WHEN e.finish_pos <= 3 THEN 1 ELSE 0 END AS target_top3
        FROM races_w r
        JOIN entries e USING (race_id)
        LEFT JOIN baselines b
          ON b.race_class = r.race_class
         AND b.water_band = r.water_band
        LEFT JOIN day_variants v ON v.race_date = r.race_date
        WHERE e.horse_id IS NOT NULL
        """
    )

    # 馬のキャリア順序を付け、lag 特徴量を作る
    df = con.execute(
        """
        WITH ordered AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY horse_id ORDER BY race_date, race_no) AS run_no
            FROM enriched
        ),
        with_lag AS (
            SELECT *,
                LAG(finish_pos, 1) OVER (PARTITION BY horse_id ORDER BY run_no) AS prev_pos_1,
                LAG(finish_pos, 2) OVER (PARTITION BY horse_id ORDER BY run_no) AS prev_pos_2,
                LAG(finish_pos, 3) OVER (PARTITION BY horse_id ORDER BY run_no) AS prev_pos_3,
                LAG(speed_figure, 1) OVER (PARTITION BY horse_id ORDER BY run_no) AS prev_sf_1,
                LAG(speed_figure, 2) OVER (PARTITION BY horse_id ORDER BY run_no) AS prev_sf_2,
                LAG(speed_figure, 3) OVER (PARTITION BY horse_id ORDER BY run_no) AS prev_sf_3,
                LAG(body_weight_kg, 1) OVER (PARTITION BY horse_id ORDER BY run_no) AS prev_body_weight_kg,
                LAG(load_weight_kg, 1) OVER (PARTITION BY horse_id ORDER BY run_no) AS prev_load_weight_kg,
                LAG(jockey_id, 1) OVER (PARTITION BY horse_id ORDER BY run_no) AS prev_jockey_id,
                LAG(race_date, 1) OVER (PARTITION BY horse_id ORDER BY run_no) AS prev_race_date,
                LAG(race_class, 1) OVER (PARTITION BY horse_id ORDER BY run_no) AS prev_race_class,
                -- ローリング集計 (現走除く・過去のみ)
                AVG(finish_pos) OVER (
                    PARTITION BY horse_id ORDER BY run_no
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) AS past_3_avg_pos,
                AVG(finish_pos) OVER (
                    PARTITION BY horse_id ORDER BY run_no
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) AS past_5_avg_pos,
                AVG(speed_figure) OVER (
                    PARTITION BY horse_id ORDER BY run_no
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) AS past_3_avg_sf,
                COUNT(*) OVER (
                    PARTITION BY horse_id ORDER BY run_no
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS lifetime_starts,
                SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END) OVER (
                    PARTITION BY horse_id ORDER BY run_no
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS lifetime_wins,
                SUM(CASE WHEN finish_pos <= 3 THEN 1 ELSE 0 END) OVER (
                    PARTITION BY horse_id ORDER BY run_no
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS lifetime_top3
        FROM ordered
        )
        SELECT *,
            (jockey_id != prev_jockey_id) AS jockey_changed,
            DATE_DIFF('day', prev_race_date::DATE, race_date::DATE) AS days_since_last_run,
            (load_weight_kg - prev_load_weight_kg) AS load_change_kg,
            (body_weight_kg - prev_body_weight_kg) AS body_weight_diff_calc,
            CASE WHEN lifetime_starts > 0
                THEN lifetime_top3 * 1.0 / lifetime_starts
                ELSE NULL
            END AS lifetime_top3_rate,
            CASE WHEN lifetime_starts > 0
                THEN lifetime_wins * 1.0 / lifetime_starts
                ELSE NULL
            END AS lifetime_win_rate
        FROM with_lag
        """
    ).fetchdf()

    return df


FEATURE_COLS_NUMERIC = [
    "horse_no",
    "post_position",
    "load_weight_kg",
    "body_weight_kg",
    "body_weight_diff_kg",
    "track_water_pct",
    "entry_count",
    "age",
    "popularity",
    "prev_pos_1",
    "prev_pos_2",
    "prev_pos_3",
    "prev_sf_1",
    "prev_sf_2",
    "prev_sf_3",
    "past_3_avg_pos",
    "past_5_avg_pos",
    "past_3_avg_sf",
    "lifetime_starts",
    "lifetime_wins",
    "lifetime_top3",
    "lifetime_win_rate",
    "lifetime_top3_rate",
    "days_since_last_run",
    "load_change_kg",
]
FEATURE_COLS_BOOL = ["has_allowance", "jockey_changed"]
FEATURE_COLS_CATEGORICAL = ["race_class", "water_band", "weather", "sex"]


def feature_columns() -> dict[str, list[str]]:
    return {
        "numeric": FEATURE_COLS_NUMERIC,
        "boolean": FEATURE_COLS_BOOL,
        "categorical": FEATURE_COLS_CATEGORICAL,
    }


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("=== 特徴量行列構築中... ===")
    df = build_feature_matrix()
    print(f"shape: {df.shape}")
    print()
    print("=== サンプル行 (馬名 シオン) ===")
    sub = df[df["horse_name"] == "シオン"].head(5)
    print(sub[["race_date", "horse_no", "load_weight_kg", "body_weight_kg",
               "speed_figure", "prev_pos_1", "prev_sf_1", "past_3_avg_sf",
               "lifetime_starts", "lifetime_top3_rate", "finish_pos", "target_win"]])
    print()
    print("=== 欠損値 (主要特徴量) ===")
    for col in ["popularity", "prev_pos_1", "past_3_avg_sf", "lifetime_starts"]:
        print(f"  {col}: {df[col].isna().sum()} / {len(df)} ({df[col].isna().mean()*100:.1f}%)")
