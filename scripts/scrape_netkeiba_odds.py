"""
netkeiba から過去レースの最終オッズを一括取得 (再開可能・冪等)。

特徴:
- races テーブルから全 race_id を取得し、未取得分のみスクレイプ
- 50レースごとに Parquet に追記
- 取得不能だった race_id は checkpoint に記録して再試行 skip
- HTMLキャッシュは Colab セッション内のみ保持 (--no-html-cache で削除)

使用例:
  uv run python -m scripts.scrape_netkeiba_odds                    # 全期間
  uv run python -m scripts.scrape_netkeiba_odds --start 2024-01-01 # 2024年以降
  uv run python -m scripts.scrape_netkeiba_odds --rate-limit 2.0
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import duckdb
import httpx
import pandas as pd

from persist.parquet_writer import DEFAULT_PARQUET_ROOT
from scrapers.netkeiba_odds import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_RATE_LIMIT_SEC,
    USER_AGENT,
    fetch_odds_html,
    local_race_id_to_netkeiba,
    parse_netkeiba_odds,
)

OUTPUT_PATH = DEFAULT_PARQUET_ROOT / "odds_netkeiba.parquet"
CHECKPOINT_PATH = Path("data/checkpoints/netkeiba_odds_progress.json")


def list_target_race_ids(start: Optional[str] = None, end: Optional[str] = None) -> list[str]:
    con = duckdb.connect()
    glob = str(DEFAULT_PARQUET_ROOT / "races/**/*.parquet")
    con.execute(
        f"CREATE VIEW races AS SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
    )
    where_clauses = []
    if start:
        where_clauses.append(f"race_date >= '{start}'")
    if end:
        where_clauses.append(f"race_date <= '{end}'")
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    df = con.execute(
        f"SELECT race_id FROM races {where_sql} ORDER BY race_date, race_no"
    ).fetchdf()
    return df["race_id"].tolist()


def load_processed() -> tuple[set[str], set[str]]:
    """戻り値: (parquet保存済 race_id, no_data race_id)"""
    saved = set()
    if OUTPUT_PATH.exists():
        df = pd.read_parquet(OUTPUT_PATH, columns=["race_id_local"])
        saved = set(df["race_id_local"].unique())
    no_data = set()
    if CHECKPOINT_PATH.exists():
        ck = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        no_data = set(ck.get("no_data_race_ids", []))
    return saved, no_data


def save_checkpoint(no_data: set[str]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(
        json.dumps({"no_data_race_ids": sorted(no_data)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_batch(rows: list[dict]) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        # PK = (race_id_local, horse_no): 新値で上書き
        existing = existing.merge(
            new_df[["race_id_local", "horse_no"]].drop_duplicates().assign(_drop=True),
            on=["race_id_local", "horse_no"], how="left"
        )
        existing = existing[existing["_drop"].isna()].drop(columns="_drop")
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False, compression="zstd")


def run(
    start: Optional[str] = None,
    end: Optional[str] = None,
    rate_limit_sec: float = DEFAULT_RATE_LIMIT_SEC,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    batch_size: int = 50,
    max_races: Optional[int] = None,
    no_html_cache: bool = False,
) -> None:
    all_ids = list_target_race_ids(start=start, end=end)
    saved, no_data = load_processed()
    pending = [r for r in all_ids if r not in saved and r not in no_data]
    if max_races is not None:
        pending = pending[:max_races]

    print(
        f"[netkeiba] 対象期間: {start or 'all'} 〜 {end or 'all'}  "
        f"全 race_id={len(all_ids)} 既処理={len(saved)} no_data={len(no_data)} 残り={len(pending)}"
    )
    if not pending:
        print("[netkeiba] 全件処理済")
        return

    started = time.time()
    buffer: list[dict] = []
    new_no_data: set[str] = set()

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
        for i, race_id_local in enumerate(pending, 1):
            netkeiba_id = local_race_id_to_netkeiba(race_id_local)
            html = fetch_odds_html(netkeiba_id, cache_root=cache_root, client=client)
            cache_hit = (cache_root / f"{netkeiba_id}.html").exists()

            if html is None:
                new_no_data.add(race_id_local)
            else:
                rows = parse_netkeiba_odds(html, race_id_local, netkeiba_id)
                if not rows:
                    new_no_data.add(race_id_local)
                else:
                    buffer.extend(asdict(r) for r in rows)

            if not cache_hit:
                time.sleep(rate_limit_sec)
            if no_html_cache and cache_hit:
                p = cache_root / f"{netkeiba_id}.html"
                if p.exists():
                    p.unlink()

            if len(buffer) >= batch_size * 9:  # 50レース × 平均9頭 ≈ 450行で書き出し
                append_batch(buffer)
                buffer = []
                # checkpoint
                no_data.update(new_no_data)
                new_no_data.clear()
                save_checkpoint(no_data)
                # ETA
                elapsed = time.time() - started
                rate = i / elapsed
                eta = (len(pending) - i) / rate if rate > 0 else float("inf")
                print(
                    f"  [{i}/{len(pending)}] elapsed={elapsed:.0f}s  rate={rate:.2f}/s  "
                    f"ETA={eta/60:.1f}min  no_data累計={len(no_data) + len(new_no_data)}"
                )

    # 最終バッチ
    append_batch(buffer)
    no_data.update(new_no_data)
    save_checkpoint(no_data)

    elapsed = time.time() - started
    print(
        f"[netkeiba] 完了: 経過={elapsed:.0f}s ({elapsed/60:.1f}分)  "
        f"成功={len(pending) - len(new_no_data) - sum(1 for x in pending if x in no_data)} "
        f"no_data={len(no_data)}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", help="YYYY-MM-DD")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument("--rate-limit", type=float, default=DEFAULT_RATE_LIMIT_SEC)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--max-races", type=int, default=None)
    p.add_argument("--no-html-cache", action="store_true")
    args = p.parse_args()
    run(
        start=args.start,
        end=args.end,
        rate_limit_sec=args.rate_limit,
        batch_size=args.batch_size,
        max_races=args.max_races,
        no_html_cache=args.no_html_cache,
    )


if __name__ == "__main__":
    main()
