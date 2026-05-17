"""
当日帯広開催の全レース最終オッズを一括取得 → Parquet に永続化。

データは data/parquet/odds/year=YYYY/month=MM/data.parquet にHive形式で保存。
PK: (race_id, horse_no, snapshot_at_utc) — 1日複数回スナップショット可能。

使用例:
  uv run python -m scripts.scrape_today_odds                 # 今日の帯広全レース
  uv run python -m scripts.scrape_today_odds --date 2026-05-17
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from datetime import date as Date
from pathlib import Path

import httpx
import pandas as pd

from persist.parquet_writer import DEFAULT_PARQUET_ROOT
from scrapers.odds import (
    BABA_CODE_OBIHIRO,
    DEFAULT_RATE_LIMIT_SEC,
    USER_AGENT,
    fetch_and_parse,
)


def _partition_path(race_date: Date, parquet_root: Path = DEFAULT_PARQUET_ROOT) -> Path:
    y = race_date.year
    m = race_date.month
    return parquet_root / "odds" / f"year={y}" / f"month={m:02d}" / "data.parquet"


def _upsert(path: Path, new_df: pd.DataFrame) -> tuple[int, int]:
    """既存と PK 重複は新値で上書き、新規は追記。戻り値: (replaced, added)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if new_df.empty:
        return 0, 0
    pk = ["race_id", "horse_no", "snapshot_at_utc"]
    if path.exists():
        existing = pd.read_parquet(path)
        new_keys = new_df[pk].drop_duplicates()
        dup_mask = existing.set_index(pk).index.isin(new_keys.set_index(pk).index)
        replaced = int(dup_mask.sum())
        kept = existing.loc[~dup_mask]
        combined = pd.concat([kept, new_df], ignore_index=True)
        added = len(new_df) - replaced
    else:
        combined = new_df
        replaced, added = 0, len(new_df)
    combined.to_parquet(path, index=False, compression="zstd")
    return replaced, added


def run(
    race_date: Date,
    *,
    baba_code: int = BABA_CODE_OBIHIRO,
    max_races: int = 12,
    rate_limit_sec: float = DEFAULT_RATE_LIMIT_SEC,
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
) -> None:
    print(f"[odds] {race_date} 帯広 baba_code={baba_code}")
    all_rows = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
        for r in range(1, max_races + 1):
            rows = fetch_and_parse(race_date, r, baba_code=baba_code, client=client)
            if not rows:
                print(f"  R{r:>2}: オッズ取得不可")
                # それ以上のレースもないと判断
                if r >= 3 and not all_rows:
                    break
                continue
            print(f"  R{r:>2}: {len(rows)} 頭分取得")
            all_rows.extend(asdict(x) for x in rows)
            time.sleep(rate_limit_sec)

    if not all_rows:
        print("[odds] 取得行ゼロ")
        return

    df = pd.DataFrame(all_rows)
    out = _partition_path(race_date, parquet_root)
    replaced, added = _upsert(out, df)
    print(f"[odds] 完了: 出力={out}  追加={added} 上書き={replaced} 合計行={len(df)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--max-races", type=int, default=12)
    args = p.parse_args()
    d = Date.fromisoformat(args.date) if args.date else Date.today()
    run(d, max_races=args.max_races)


if __name__ == "__main__":
    main()
