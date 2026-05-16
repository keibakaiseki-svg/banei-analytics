"""
entries / races テーブルから馬・騎手・調教師マスタを派生生成する。

マスタはレース確定後に都度再計算可能（純粋関数なので冪等）。
Phase 2 でバックフィルが進めば自動的にデータ密度が上がる。
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from persist.parquet_writer import DEFAULT_PARQUET_ROOT


def _open_connection(parquet_root: Path) -> duckdb.DuckDBPyConnection:
    """data/parquet/* を読み込んで races/entries/payouts ビューを張った接続を返す。"""
    con = duckdb.connect()
    for t in ("races", "entries", "payouts"):
        glob = str(parquet_root / t / "**/*.parquet")
        con.execute(
            f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
        )
    return con


def build_horse_master(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> pd.DataFrame:
    """
    馬マスタを entries × races から派生。
    完走したエントリのみ集計対象（中止・失格・取消は除外）。
    """
    con = _open_connection(parquet_root)
    df = con.execute(
        """
        WITH valid AS (
            SELECT e.*, r.race_date, r.track_water_pct, r.race_class
            FROM entries e
            JOIN races r USING (race_id)
            WHERE finish_status IN ('finished', 'dead_heat', 'demoted')
              AND finish_pos IS NOT NULL
        )
        SELECT
            horse_id,
            ANY_VALUE(horse_name)                                AS horse_name,
            ANY_VALUE(SUBSTR(sex_age, 1, 1))                     AS sex,
            MIN(race_date)                                       AS first_seen_date,
            MAX(race_date)                                       AS last_seen_date,
            COUNT(*)                                             AS starts,
            SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END)      AS wins,
            SUM(CASE WHEN finish_pos <= 2 THEN 1 ELSE 0 END)     AS rentai,
            SUM(CASE WHEN finish_pos <= 3 THEN 1 ELSE 0 END)     AS top3,
            ROUND(AVG(finish_pos), 2)                            AS avg_finish_pos,
            ROUND(AVG(time_diff_from_winner_sec), 2)             AS avg_time_diff_from_winner_sec,
            ROUND(AVG(body_weight_kg), 1)                        AS avg_body_weight_kg,
            ROUND(AVG(load_weight_kg), 1)                        AS avg_load_weight_kg,
            -- 水分量別の複勝率（重い馬場・軽い馬場での適性）
            ROUND(
              SUM(CASE WHEN track_water_pct < 2.0 AND finish_pos <= 3 THEN 1 ELSE 0 END) * 1.0 /
              NULLIF(SUM(CASE WHEN track_water_pct < 2.0 THEN 1 ELSE 0 END), 0), 3
            ) AS top3_rate_dry_track,
            ROUND(
              SUM(CASE WHEN track_water_pct >= 4.0 AND finish_pos <= 3 THEN 1 ELSE 0 END) * 1.0 /
              NULLIF(SUM(CASE WHEN track_water_pct >= 4.0 THEN 1 ELSE 0 END), 0), 3
            ) AS top3_rate_wet_track
        FROM valid
        WHERE horse_id IS NOT NULL
        GROUP BY horse_id
        ORDER BY starts DESC, last_seen_date DESC
        """
    ).fetchdf()
    return df


def build_jockey_master(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> pd.DataFrame:
    """騎手マスタ。allowance_marker の出現有無も付与（性別マッピングは別途）。"""
    con = _open_connection(parquet_root)
    df = con.execute(
        """
        WITH valid AS (
            SELECT e.*, r.race_date
            FROM entries e
            JOIN races r USING (race_id)
            WHERE finish_status IN ('finished', 'dead_heat', 'demoted')
              AND finish_pos IS NOT NULL
        )
        SELECT
            jockey_id,
            ANY_VALUE(jockey_name)                                AS jockey_name,
            MIN(race_date)                                        AS first_seen_date,
            MAX(race_date)                                        AS last_seen_date,
            COUNT(*)                                              AS starts,
            SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END)       AS wins,
            SUM(CASE WHEN finish_pos <= 3 THEN 1 ELSE 0 END)      AS top3,
            ROUND(SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 3) AS win_rate,
            ROUND(SUM(CASE WHEN finish_pos <= 3 THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 3) AS top3_rate,
            -- 減量マーカー出現有無
            SUM(CASE WHEN allowance_marker IS NOT NULL THEN 1 ELSE 0 END) AS allowance_rides,
            ANY_VALUE(allowance_marker) FILTER (WHERE allowance_marker IS NOT NULL) AS observed_marker
        FROM valid
        WHERE jockey_id IS NOT NULL
        GROUP BY jockey_id
        ORDER BY starts DESC, last_seen_date DESC
        """
    ).fetchdf()
    return df


def build_trainer_master(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> pd.DataFrame:
    """調教師マスタ。"""
    con = _open_connection(parquet_root)
    df = con.execute(
        """
        WITH valid AS (
            SELECT e.*, r.race_date
            FROM entries e
            JOIN races r USING (race_id)
            WHERE finish_status IN ('finished', 'dead_heat', 'demoted')
              AND finish_pos IS NOT NULL
        )
        SELECT
            trainer_id,
            ANY_VALUE(trainer_name)                               AS trainer_name,
            MIN(race_date)                                        AS first_seen_date,
            MAX(race_date)                                        AS last_seen_date,
            COUNT(*)                                              AS starts,
            SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END)       AS wins,
            SUM(CASE WHEN finish_pos <= 3 THEN 1 ELSE 0 END)      AS top3,
            ROUND(SUM(CASE WHEN finish_pos <= 3 THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 3) AS top3_rate
        FROM valid
        WHERE trainer_id IS NOT NULL
        GROUP BY trainer_id
        ORDER BY starts DESC, last_seen_date DESC
        """
    ).fetchdf()
    return df


def materialize_masters(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> dict[str, int]:
    """マスタを Parquet に書き出す。完全再生成（差分ではない）。"""
    masters = {
        "horses": build_horse_master(parquet_root),
        "jockeys": build_jockey_master(parquet_root),
        "trainers": build_trainer_master(parquet_root),
    }
    counts: dict[str, int] = {}
    for name, df in masters.items():
        out = parquet_root / name / "master.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, compression="zstd")
        counts[name] = len(df)
    return counts


if __name__ == "__main__":
    counts = materialize_masters()
    for k, v in counts.items():
        print(f"  {k}: {v} 件")
