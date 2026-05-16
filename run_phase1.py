"""
Phase 1 パイプライン: 指定日の全レースをスクレイピング → パース → Parquet 永続化 → 検証クエリ。

使用例:
  uv run python run_phase1.py 2026-05-04
  uv run python run_phase1.py 2026-05-04 --force-refresh
"""

from __future__ import annotations

import argparse
from datetime import date as Date
from pathlib import Path

import duckdb

from persist.parquet_writer import DEFAULT_PARQUET_ROOT, write_race
from scrapers.fetcher import BABA_CODE_OBIHIRO, _build_url, fetch_day
from scrapers.race_page import parse_race_page


def run(race_date: Date, force_refresh: bool = False) -> None:
    print(f"=== Phase 1: {race_date} ===")
    fetched = fetch_day(race_date, force_refresh=force_refresh)
    print(f"取得レース数: {len(fetched)}")
    if not fetched:
        print("開催なし or 取得失敗")
        return

    totals = {"races": [0, 0], "entries": [0, 0], "payouts": [0, 0]}
    for race_no, html in fetched:
        info, entries_df, payouts_df = parse_race_page(html)
        source_url = _build_url(race_date, race_no, BABA_CODE_OBIHIRO)
        summary = write_race(info, entries_df, payouts_df, source_url=source_url)
        for table, (replaced, added) in summary.items():
            totals[table][0] += replaced
            totals[table][1] += added
        print(
            f"  R{race_no:>2}: {info.race_name or '(無名)'} | "
            f"class={info.race_class} water={info.track_water_pct} "
            f"entries={len(entries_df)} payouts={len(payouts_df)} | "
            f"replaced/added: races={summary['races']} entries={summary['entries']} payouts={summary['payouts']}"
        )

    print()
    print("=== 永続化サマリ (replaced, added) ===")
    for k, v in totals.items():
        print(f"  {k}: replaced={v[0]} added={v[1]}")

    print()
    print("=== DuckDB 検証クエリ ===")
    _verify(race_date)


def _verify(race_date: Date) -> None:
    parquet_root = DEFAULT_PARQUET_ROOT
    races_glob = str(parquet_root / "races/**/*.parquet")
    entries_glob = str(parquet_root / "entries/**/*.parquet")
    payouts_glob = str(parquet_root / "payouts/**/*.parquet")

    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW races AS SELECT * FROM read_parquet('{races_glob}', hive_partitioning=true)"
    )
    con.execute(
        f"CREATE VIEW entries AS SELECT * FROM read_parquet('{entries_glob}', hive_partitioning=true)"
    )
    con.execute(
        f"CREATE VIEW payouts AS SELECT * FROM read_parquet('{payouts_glob}', hive_partitioning=true)"
    )

    date_str = race_date.strftime("%Y-%m-%d")
    print("[races] 当日件数・水分量推移")
    print(
        con.execute(
            f"""
            SELECT race_no, race_name, race_class, weather, track_water_pct, entry_count
            FROM races
            WHERE race_date = '{date_str}'
            ORDER BY race_no
            """
        ).fetchdf()
    )

    print()
    print("[entries] レースごと出走頭数・完走数")
    print(
        con.execute(
            f"""
            SELECT race_id,
                   COUNT(*) AS entries,
                   SUM(CASE WHEN finish_pos IS NOT NULL THEN 1 ELSE 0 END) AS finished,
                   SUM(CASE WHEN allowance_marker IS NOT NULL THEN 1 ELSE 0 END) AS allowance_riders
            FROM entries
            WHERE race_id LIKE '{date_str.replace('-', '')}%'
            GROUP BY race_id
            ORDER BY race_id
            """
        ).fetchdf()
    )

    print()
    print("[payouts] 当日の単勝・三連単の払戻分布")
    print(
        con.execute(
            f"""
            SELECT bet_type,
                   COUNT(*) AS bets,
                   MIN(payout_yen) AS min_yen,
                   AVG(payout_yen)::INT AS avg_yen,
                   MAX(payout_yen) AS max_yen
            FROM payouts
            WHERE race_id LIKE '{date_str.replace('-', '')}%'
              AND bet_type IN ('単勝', '三連単')
            GROUP BY bet_type
            ORDER BY bet_type
            """
        ).fetchdf()
    )

    print()
    print("[integrity] 重複PKチェック (0であるべき)")
    for tbl, pk in [
        ("races", "race_id"),
        ("entries", "race_id || '_' || horse_no"),
        ("payouts", "race_id || '_' || bet_type || '_' || combination"),
    ]:
        dup = con.execute(
            f"SELECT COUNT(*) FROM (SELECT {pk} k FROM {tbl} GROUP BY k HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        print(f"  {tbl}: 重複={dup}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("race_date", help="YYYY-MM-DD")
    p.add_argument("--force-refresh", action="store_true")
    args = p.parse_args()
    run(Date.fromisoformat(args.race_date), force_refresh=args.force_refresh)


if __name__ == "__main__":
    main()
