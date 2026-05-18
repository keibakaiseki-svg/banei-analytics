"""
netkeiba から馬連・馬単・ワイド・三連複・三連単の確定オッズを一括取得 (再開可能・冪等)。

特徴:
- races テーブルから (race_id, entry_count) を取得
- 三連複/三連単は entry_count に応じて軸馬を iterate
- batch_size レースごとに各 Parquet に追記
- 券種別 checkpoint で no_data race_id を記録、再試行 skip

使用例:
  uv run python -m scripts.scrape_netkeiba_combo_odds
  uv run python -m scripts.scrape_netkeiba_combo_odds --start 2024-01-01 --end 2026-04-30
  uv run python -m scripts.scrape_netkeiba_combo_odds --bet-types umaren,wide,sanrenpuku
  uv run python -m scripts.scrape_netkeiba_combo_odds --max-races 10  # 動作確認
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import duckdb
import httpx
import pandas as pd

from persist.parquet_writer import DEFAULT_PARQUET_ROOT
from scrapers.netkeiba_combo_odds import (
    BET_TYPES,
    DEFAULT_COMBO_CACHE_ROOT,
    _axis_range,
    _cache_path,
    fetch_and_parse_combo_for_race,
)
from scrapers.netkeiba_odds import (
    DEFAULT_RATE_LIMIT_SEC,
    USER_AGENT,
    local_race_id_to_netkeiba,
)


def output_path(bet_type: str) -> Path:
    return DEFAULT_PARQUET_ROOT / f"odds_netkeiba_{bet_type}.parquet"


def checkpoint_path(bet_type: str) -> Path:
    return Path("data/checkpoints") / f"netkeiba_combo_{bet_type}_progress.json"


def list_target_races(start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """戻り値: race_id, entry_count, race_date 順"""
    con = duckdb.connect()
    glob = str(DEFAULT_PARQUET_ROOT / "races/**/*.parquet")
    con.execute(
        f"CREATE VIEW races AS SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
    )
    where = []
    if start:
        where.append(f"race_date >= '{start}'")
    if end:
        where.append(f"race_date <= '{end}'")
    where_sql = "WHERE " + " AND ".join(where) if where else ""
    return con.execute(
        f"SELECT race_id, entry_count FROM races {where_sql} ORDER BY race_date, race_no"
    ).fetchdf()


def load_processed(bet_type: str) -> tuple[set[str], set[str]]:
    out = output_path(bet_type)
    saved: set[str] = set()
    if out.exists():
        df = pd.read_parquet(out, columns=["race_id_local"])
        saved = set(df["race_id_local"].unique())
    no_data: set[str] = set()
    ck = checkpoint_path(bet_type)
    if ck.exists():
        no_data = set(json.loads(ck.read_text(encoding="utf-8")).get("no_data_race_ids", []))
    return saved, no_data


def save_checkpoint(bet_type: str, no_data: set[str]) -> None:
    ck = checkpoint_path(bet_type)
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(
        json.dumps({"no_data_race_ids": sorted(no_data)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_batch(bet_type: str, rows: list[dict]) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    out = output_path(bet_type)
    if out.exists():
        existing = pd.read_parquet(out)
        existing = existing.merge(
            new_df[["race_id_local", "combination"]].drop_duplicates().assign(_drop=True),
            on=["race_id_local", "combination"], how="left",
        )
        existing = existing[existing["_drop"].isna()].drop(columns="_drop")
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False, compression="zstd")


def estimate_requests(races_df: pd.DataFrame, bet_types: list[str]) -> int:
    n = 0
    for _, row in races_df.iterrows():
        entry = int(row["entry_count"]) if pd.notna(row["entry_count"]) else 10
        for bt in bet_types:
            if not BET_TYPES[bt]["needs_axis"]:
                n += 1
            else:
                n += len(_axis_range(bt, entry))
    return n


def remove_html_cache_for_race(
    cache_root: Path, bet_type: str, netkeiba_id: str, jiku: Optional[int]
) -> None:
    p = _cache_path(cache_root, bet_type, netkeiba_id, jiku)
    if p.exists():
        p.unlink()


def run(
    start: Optional[str] = None,
    end: Optional[str] = None,
    bet_types: Optional[list[str]] = None,
    rate_limit_sec: float = DEFAULT_RATE_LIMIT_SEC,
    cache_root: Path = DEFAULT_COMBO_CACHE_ROOT,
    batch_size: int = 50,
    max_races: Optional[int] = None,
    no_html_cache: bool = False,
) -> None:
    bet_types = bet_types or list(BET_TYPES.keys())
    for bt in bet_types:
        if bt not in BET_TYPES:
            raise ValueError(f"unknown bet_type: {bt}")

    races_df = list_target_races(start, end)
    print(f"[combo odds] 対象期間: {start or 'all'} 〜 {end or 'all'}  全 race={len(races_df):,}")

    pending_by_type: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for bt in bet_types:
        saved, no_data = load_processed(bt)
        pending = set(races_df["race_id"]) - saved - no_data
        pending_by_type[bt] = (pending, saved, no_data)
        print(f"  {BET_TYPES[bt]['label']:6s}: 既処理={len(saved):,}  no_data={len(no_data):,}  残り={len(pending):,}")

    # ユニーク race リスト (取得順): どの券種でも未処理なら対象
    union_mask = pd.Series(False, index=races_df.index)
    for bt in bet_types:
        union_mask |= races_df["race_id"].isin(pending_by_type[bt][0])
    pending_races = races_df[union_mask].reset_index(drop=True)
    if max_races is not None:
        pending_races = pending_races.head(max_races)

    if pending_races.empty:
        print("[combo odds] 全件処理済")
        return

    expected_req = estimate_requests(
        pending_races[pending_races["race_id"].isin(
            set().union(*[pending_by_type[bt][0] for bt in bet_types])
        )],
        bet_types,
    )
    eta_min = expected_req * rate_limit_sec / 60
    print(f"[combo odds] 取得開始: ユニーク race={len(pending_races):,}  期待 req≈{expected_req:,}  ETA≈{eta_min:.0f}分")

    started = time.time()
    buffers: dict[str, list[dict]] = {bt: [] for bt in bet_types}
    new_no_data: dict[str, set[str]] = {bt: set() for bt in bet_types}
    processed = 0
    total_req = 0

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
        for i, row in pending_races.iterrows():
            race_id_local = row["race_id"]
            entry_count = int(row["entry_count"]) if pd.notna(row["entry_count"]) else 10
            netkeiba_id = local_race_id_to_netkeiba(race_id_local)

            for bt in bet_types:
                if race_id_local not in pending_by_type[bt][0]:
                    continue
                try:
                    parsed_rows, req_count = fetch_and_parse_combo_for_race(
                        race_id_local, bt, entry_count,
                        client=client,
                        cache_root=cache_root,
                        rate_limit_sec=rate_limit_sec,
                    )
                except Exception as e:
                    print(f"  ERROR on {race_id_local}/{bt}: {type(e).__name__}: {e}")
                    continue
                total_req += req_count
                if not parsed_rows:
                    new_no_data[bt].add(race_id_local)
                else:
                    buffers[bt].extend(asdict(r) for r in parsed_rows)
                if no_html_cache:
                    # 取り終わったHTMLは削除 (Colab ディスク節約)
                    if not BET_TYPES[bt]["needs_axis"]:
                        remove_html_cache_for_race(cache_root, bt, netkeiba_id, None)
                    else:
                        for jiku in _axis_range(bt, entry_count):
                            remove_html_cache_for_race(cache_root, bt, netkeiba_id, jiku)

            processed += 1
            if processed % batch_size == 0:
                for bt in bet_types:
                    if buffers[bt]:
                        append_batch(bt, buffers[bt])
                        buffers[bt] = []
                    if new_no_data[bt]:
                        pending_by_type[bt][2].update(new_no_data[bt])
                        new_no_data[bt].clear()
                    save_checkpoint(bt, pending_by_type[bt][2])
                elapsed = time.time() - started
                race_rate = processed / elapsed
                req_rate = total_req / elapsed if elapsed > 0 else 0
                eta = (len(pending_races) - processed) / race_rate if race_rate > 0 else float("inf")
                print(
                    f"  [{processed:,}/{len(pending_races):,}] elapsed={elapsed:.0f}s  "
                    f"race_rate={race_rate:.2f}/s  req={total_req:,} ({req_rate:.1f}req/s)  ETA={eta/60:.1f}min"
                )

    # 最終バッチ
    for bt in bet_types:
        if buffers[bt]:
            append_batch(bt, buffers[bt])
        if new_no_data[bt]:
            pending_by_type[bt][2].update(new_no_data[bt])
        save_checkpoint(bt, pending_by_type[bt][2])

    elapsed = time.time() - started
    print(f"[combo odds] 完了: 経過={elapsed:.0f}s ({elapsed/60:.1f}分)  累計req={total_req:,}")
    for bt in bet_types:
        out = output_path(bt)
        if out.exists():
            df = pd.read_parquet(out, columns=["race_id_local"])
            print(f"  {BET_TYPES[bt]['label']:6s}: {df.race_id_local.nunique():,}レース / {len(df):,}行")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", help="YYYY-MM-DD", default="2024-01-01")
    p.add_argument("--end", help="YYYY-MM-DD", default="2026-04-30")
    p.add_argument(
        "--bet-types",
        help=f"カンマ区切り。指定なしで全券種: {','.join(BET_TYPES.keys())}",
        default=None,
    )
    p.add_argument("--rate-limit", type=float, default=DEFAULT_RATE_LIMIT_SEC)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--max-races", type=int, default=None, help="動作確認用にレース数制限")
    p.add_argument("--no-html-cache", action="store_true", help="使用後 HTML を削除しディスク節約")
    args = p.parse_args()
    bet_types = [b.strip() for b in args.bet_types.split(",")] if args.bet_types else None
    run(
        start=args.start,
        end=args.end,
        bet_types=bet_types,
        rate_limit_sec=args.rate_limit,
        batch_size=args.batch_size,
        max_races=args.max_races,
        no_html_cache=args.no_html_cache,
    )


if __name__ == "__main__":
    main()
