"""
全馬の血統情報を一括スクレイプする (再開可能・冪等)。

- entries テーブルから登場した全馬IDを取得
- 既に data/parquet/horse_pedigree.parquet に保存済みの馬は skip
- 取得した血統データを 50馬ごとに Parquet に追記
- ローカルHTMLキャッシュは data/raw_html/horse_info/ に保管

使用例:
  uv run python -m scripts.scrape_pedigrees
  uv run python -m scripts.scrape_pedigrees --rate-limit 3.0
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import duckdb
import httpx
import pandas as pd

from persist.parquet_writer import DEFAULT_PARQUET_ROOT
from scrapers.horse_info import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_RATE_LIMIT_SEC,
    USER_AGENT,
    fetch_horse_html,
    parse_horse_info,
)

OUTPUT_PATH = DEFAULT_PARQUET_ROOT / "horse_pedigree.parquet"


def list_all_horse_ids() -> list[str]:
    con = duckdb.connect()
    glob = str(DEFAULT_PARQUET_ROOT / "entries/**/*.parquet")
    con.execute(
        f"CREATE VIEW entries AS SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
    )
    df = con.execute(
        """
        SELECT DISTINCT horse_id
        FROM entries
        WHERE horse_id IS NOT NULL
        """
    ).fetchdf()
    return sorted(df["horse_id"].dropna().tolist())


def list_already_scraped() -> set[str]:
    if not OUTPUT_PATH.exists():
        return set()
    df = pd.read_parquet(OUTPUT_PATH, columns=["horse_id"])
    return set(df["horse_id"].tolist())


def append_batch(rows: list[dict]) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        # 重複排除 (horse_id PK)
        existing = existing[~existing["horse_id"].isin(new_df["horse_id"])]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False, compression="zstd")


def run(
    rate_limit_sec: float = DEFAULT_RATE_LIMIT_SEC,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    batch_size: int = 50,
    max_horses: int | None = None,
) -> None:
    all_ids = list_all_horse_ids()
    scraped = list_already_scraped()
    pending = [hid for hid in all_ids if hid not in scraped]
    if max_horses is not None:
        pending = pending[:max_horses]

    print(f"[pedigree] 全馬数={len(all_ids)}  既処理={len(scraped)}  対象={len(pending)}")
    if not pending:
        print("[pedigree] 全件処理済")
        return

    started = time.time()
    buffer: list[dict] = []
    fail_count = 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
        for i, hid in enumerate(pending, 1):
            cache_path = cache_root / f"{hid}.html"
            cache_hit = cache_path.exists()
            html = fetch_horse_html(hid, cache_root=cache_root, client=client)
            if html is None:
                fail_count += 1
                buffer.append({"horse_id": hid, "fetch_failed": True})
            else:
                try:
                    info = parse_horse_info(html, hid)
                    row = asdict(info)
                    row["fetch_failed"] = False
                    buffer.append(row)
                except Exception as e:
                    fail_count += 1
                    buffer.append({"horse_id": hid, "fetch_failed": True, "error": str(e)})

            if not cache_hit:
                time.sleep(rate_limit_sec)

            if len(buffer) >= batch_size:
                append_batch(buffer)
                buffer = []
                elapsed = time.time() - started
                rate = i / elapsed
                eta = (len(pending) - i) / rate if rate > 0 else float("inf")
                print(
                    f"  [{i}/{len(pending)}] elapsed={elapsed:.0f}s  "
                    f"rate={rate:.2f}/s  ETA={eta:.0f}s  failures={fail_count}"
                )

    append_batch(buffer)
    elapsed = time.time() - started
    print(
        f"[pedigree] 完了: 経過={elapsed:.0f}s  失敗={fail_count}/{len(pending)}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rate-limit", type=float, default=DEFAULT_RATE_LIMIT_SEC)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--max-horses", type=int, default=None, help="テスト用に処理数を制限")
    args = p.parse_args()
    run(
        rate_limit_sec=args.rate_limit,
        batch_size=args.batch_size,
        max_horses=args.max_horses,
    )


if __name__ == "__main__":
    main()
