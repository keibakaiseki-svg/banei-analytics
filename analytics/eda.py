"""
探索分析の再利用可能クエリ集。

データ量が少ない段階でも実行可能。Phase 2 のバックフィルが進めば
そのまま同じクエリで信頼性の高い知見が得られる設計。
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
    for m in ("horses", "jockeys", "trainers"):
        path = parquet_root / m / "master.parquet"
        if path.exists():
            con.execute(f"CREATE VIEW {m} AS SELECT * FROM read_parquet('{path}')")
    return con


def dataset_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """現在のデータセット規模をテーブル別に集計。"""
    rows = []
    for t in ("races", "entries", "payouts"):
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        rows.append({"table": t, "rows": n})
    for m in ("horses", "jockeys", "trainers"):
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {m}").fetchone()[0]
            rows.append({"table": m, "rows": n})
        except Exception:
            pass
    return pd.DataFrame(rows)


def date_coverage(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            MIN(race_date) AS first_date,
            MAX(race_date) AS last_date,
            COUNT(DISTINCT race_date) AS race_days,
            COUNT(*) AS total_races
        FROM races
        """
    ).fetchdf()


def water_distribution(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """馬場水分量の分布。Phase 4のクラスタリング軸として重要。"""
    return con.execute(
        """
        SELECT
            ROUND(track_water_pct, 1) AS water_pct,
            COUNT(*) AS races
        FROM races
        WHERE track_water_pct IS NOT NULL
        GROUP BY water_pct
        ORDER BY water_pct
        """
    ).fetchdf()


def class_distribution(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            SUBSTR(race_class, 1, 2) AS class_prefix,
            COUNT(*) AS races,
            ROUND(AVG(prize_1st), 0)::INT AS avg_prize_1st
        FROM races
        WHERE race_class IS NOT NULL
        GROUP BY class_prefix
        ORDER BY class_prefix
        """
    ).fetchdf()


def field_size_distribution(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            entry_count,
            COUNT(*) AS races
        FROM races
        GROUP BY entry_count
        ORDER BY entry_count
        """
    ).fetchdf()


def post_position_win_rate(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    馬番別の勝率・複勝率。
    ばんえいの「馬番効果」を検証する基本クエリ。データが少ないと信頼区間が広いので注意。
    """
    return con.execute(
        """
        SELECT
            horse_no,
            COUNT(*) AS starts,
            SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
            ROUND(SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 3) AS win_rate,
            ROUND(SUM(CASE WHEN finish_pos <= 3 THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 3) AS top3_rate
        FROM entries
        WHERE finish_pos IS NOT NULL
        GROUP BY horse_no
        ORDER BY horse_no
        """
    ).fetchdf()


def popularity_finish_correlation(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """人気順 × 着順分布。1番人気の勝率など市場効率の指標。"""
    return con.execute(
        """
        SELECT
            popularity,
            COUNT(*) AS starts,
            SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
            ROUND(SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 3) AS win_rate,
            ROUND(SUM(CASE WHEN finish_pos <= 3 THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 3) AS top3_rate
        FROM entries
        WHERE finish_pos IS NOT NULL AND popularity IS NOT NULL
        GROUP BY popularity
        ORDER BY popularity
        """
    ).fetchdf()


def water_band_top3_rate(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """水分量帯別の人気上位馬の的中率（市場が水分量変化に追従できているか）。"""
    return con.execute(
        """
        SELECT
            CASE
                WHEN r.track_water_pct < 1.5 THEN '0_dry'
                WHEN r.track_water_pct < 2.5 THEN '1_normal'
                WHEN r.track_water_pct < 4.0 THEN '2_moist'
                ELSE '3_wet'
            END AS water_band,
            COUNT(*) AS starts,
            ROUND(AVG(CASE WHEN e.popularity = 1 AND e.finish_pos = 1 THEN 1.0
                           WHEN e.popularity = 1 THEN 0.0 ELSE NULL END), 3) AS pop1_win_rate
        FROM entries e
        JOIN races r USING (race_id)
        WHERE e.finish_pos IS NOT NULL AND r.track_water_pct IS NOT NULL
        GROUP BY water_band
        ORDER BY water_band
        """
    ).fetchdf()


def jockey_allowance_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """減量騎手出現頻度。マーカー種別ごとの集計。"""
    return con.execute(
        """
        SELECT
            COALESCE(allowance_marker, '(なし)') AS marker,
            COUNT(*) AS rides,
            COUNT(DISTINCT jockey_id) AS unique_jockeys,
            SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
            ROUND(SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 3) AS win_rate
        FROM entries
        WHERE finish_pos IS NOT NULL
        GROUP BY marker
        ORDER BY rides DESC
        """
    ).fetchdf()


def body_weight_diff_vs_finish(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """馬体重増減と着順の関係（コンディション指標の検証）。"""
    return con.execute(
        """
        SELECT
            CASE
                WHEN body_weight_diff_kg <= -10 THEN 'A_減10kg以上'
                WHEN body_weight_diff_kg <  0   THEN 'B_微減'
                WHEN body_weight_diff_kg =  0   THEN 'C_変動なし'
                WHEN body_weight_diff_kg <= 10  THEN 'D_微増'
                ELSE                                'E_増10kg超'
            END AS weight_band,
            COUNT(*) AS starts,
            ROUND(AVG(finish_pos), 2) AS avg_finish_pos,
            ROUND(SUM(CASE WHEN finish_pos <= 3 THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 3) AS top3_rate
        FROM entries
        WHERE finish_pos IS NOT NULL AND body_weight_diff_kg IS NOT NULL
        GROUP BY weight_band
        ORDER BY weight_band
        """
    ).fetchdf()


def load_weight_vs_finish(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """積載重量帯別の平均着順と複勝率。"""
    return con.execute(
        """
        SELECT
            (load_weight_kg / 50) * 50 AS load_band,
            COUNT(*) AS starts,
            ROUND(AVG(finish_pos), 2) AS avg_finish_pos,
            ROUND(SUM(CASE WHEN finish_pos <= 3 THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 3) AS top3_rate
        FROM entries
        WHERE finish_pos IS NOT NULL AND load_weight_kg IS NOT NULL
        GROUP BY load_band
        ORDER BY load_band
        """
    ).fetchdf()


def market_efficiency(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    単勝の市場効率: 人気順別の100円ベット回収率。
    払戻情報は勝ち馬のみだが、ベット側は全エントリで計算する必要があるため LEFT JOIN を使う。
    1.0 = 100%回収。胴元控除20-25%程度のため通常 0.75〜0.85 程度に収まる。
    """
    return con.execute(
        """
        WITH bets AS (
            SELECT e.popularity, e.finish_pos,
                   p.payout_yen,
                   CASE WHEN e.finish_pos = 1 THEN 1 ELSE 0 END AS won
            FROM entries e
            LEFT JOIN (
                SELECT race_id, CAST(combination AS INT) AS horse_no, payout_yen
                FROM payouts WHERE bet_type = '単勝'
            ) p ON e.race_id = p.race_id AND e.horse_no = p.horse_no
            WHERE e.popularity IS NOT NULL AND e.finish_pos IS NOT NULL
        )
        SELECT popularity,
               COUNT(*) AS bets,
               SUM(won) AS wins,
               ROUND(SUM(won) * 1.0 / COUNT(*), 3) AS win_rate,
               ROUND(
                 SUM(CASE WHEN won = 1 THEN payout_yen ELSE 0 END) * 1.0 / (COUNT(*) * 100),
                 3
               ) AS expected_return
        FROM bets
        GROUP BY popularity ORDER BY popularity
        """
    ).fetchdf()
