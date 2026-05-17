"""
過去レースの一括バックフィルスクリプト（再開可能）。

機能:
- 日付範囲を指定して全レースを取得・パース・Parquet 永続化
- チェックポイント保存により中断後の再開が可能
- 開催のない日は no_race_dates に記録してスキップ
- Colab 上で動くよう --no-html-cache でHTMLを使い捨てモードに切り替え可能

使用例:
  # 直近30日（ローカル開発・HTMLキャッシュ保持）
  uv run python backfill.py --start 2026-04-16 --end 2026-05-15

  # Colab 用（HTMLは処理後に削除してディスク節約）
  uv run python backfill.py --start 2014-04-01 --end 2024-03-31 --no-html-cache

  # チェックポイントを無視してやり直し
  uv run python backfill.py --start 2026-05-01 --end 2026-05-15 --no-resume
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from persist.parquet_writer import DEFAULT_PARQUET_ROOT, write_race
from scrapers.fetcher import (
    BABA_CODE_OBIHIRO,
    DEFAULT_CACHE_ROOT,
    _build_url,
    fetch_day,
)
from scrapers.race_page import parse_race_page

DEFAULT_CHECKPOINT = Path("data/checkpoints/backfill_progress.json")


@dataclass
class Checkpoint:
    completed_dates: set[str] = field(default_factory=set)
    no_race_dates: set[str] = field(default_factory=set)
    total_races: int = 0
    total_entries: int = 0
    total_payouts: int = 0
    last_updated: Optional[str] = None
    path: Path = DEFAULT_CHECKPOINT

    @classmethod
    def load(cls, path: Path = DEFAULT_CHECKPOINT) -> "Checkpoint":
        if not path.exists():
            return cls(path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            completed_dates=set(data.get("completed_dates", [])),
            no_race_dates=set(data.get("no_race_dates", [])),
            total_races=data.get("total_races", 0),
            total_entries=data.get("total_entries", 0),
            total_payouts=data.get("total_payouts", 0),
            last_updated=data.get("last_updated"),
            path=path,
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_updated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        data = {
            "completed_dates": sorted(self.completed_dates),
            "no_race_dates": sorted(self.no_race_dates),
            "total_races": self.total_races,
            "total_entries": self.total_entries,
            "total_payouts": self.total_payouts,
            "last_updated": self.last_updated,
        }
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def daterange(start: Date, end: Date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def process_date(
    race_date: Date,
    *,
    cache_root: Path,
    parquet_root: Path,
    no_html_cache: bool,
) -> tuple[int, int, int]:
    """1日分を処理。戻り値: (races, entries, payouts) の追加件数。"""
    fetched = fetch_day(race_date, cache_root=cache_root)
    if not fetched:
        return 0, 0, 0

    n_races = n_entries = n_payouts = 0
    for race_no, html in fetched:
        info, entries_df, payouts_df = parse_race_page(html)
        source_url = _build_url(race_date, race_no, BABA_CODE_OBIHIRO)
        summary = write_race(
            info, entries_df, payouts_df, source_url=source_url, parquet_root=parquet_root
        )
        # 処理した行数 = replaced + added（再処理時でも0にならないように）
        n_races += sum(summary["races"])
        n_entries += sum(summary["entries"])
        n_payouts += sum(summary["payouts"])

    if no_html_cache:
        day_dir = cache_root / race_date.strftime("%Y-%m-%d")
        if day_dir.exists():
            shutil.rmtree(day_dir)

    return n_races, n_entries, n_payouts


def run(
    start: Date,
    end: Date,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    resume: bool = True,
    no_html_cache: bool = False,
    save_every: int = 5,
) -> Checkpoint:
    ck = Checkpoint.load(checkpoint_path) if resume else Checkpoint(path=checkpoint_path)
    print(
        f"[backfill] range={start}〜{end}  "
        f"resume={resume}  completed={len(ck.completed_dates)}  no_race={len(ck.no_race_dates)}"
    )

    processed_since_save = 0
    started_at = time.time()
    total_days = (end - start).days + 1
    seen_days = 0

    for d in daterange(start, end):
        seen_days += 1
        date_str = d.strftime("%Y-%m-%d")
        if resume and (date_str in ck.completed_dates or date_str in ck.no_race_dates):
            continue

        try:
            n_r, n_e, n_p = process_date(
                d,
                cache_root=cache_root,
                parquet_root=parquet_root,
                no_html_cache=no_html_cache,
            )
        except Exception as e:
            print(f"  [{date_str}] ERROR: {e!r} -- スキップ")
            continue

        if n_r == 0:
            ck.no_race_dates.add(date_str)
            print(f"  [{date_str}] 開催なし")
        else:
            ck.completed_dates.add(date_str)
            ck.total_races += n_r
            ck.total_entries += n_e
            ck.total_payouts += n_p
            elapsed = time.time() - started_at
            print(
                f"  [{date_str}] races+={n_r} entries+={n_e} payouts+={n_p}  "
                f"(cum: r={ck.total_races} e={ck.total_entries} p={ck.total_payouts})  "
                f"elapsed={elapsed:.0f}s  progress={seen_days}/{total_days}"
            )

        processed_since_save += 1
        if processed_since_save >= save_every:
            ck.save()
            processed_since_save = 0

    ck.save()
    elapsed = time.time() - started_at
    print()
    print(
        f"[backfill] 完了: 経過={elapsed:.0f}s  "
        f"完了日={len(ck.completed_dates)}  開催なし日={len(ck.no_race_dates)}  "
        f"累計レース={ck.total_races} エントリ={ck.total_entries} 払戻={ck.total_payouts}"
    )
    return ck


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="チェックポイントファイル",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="チェックポイントを無視して全期間を再処理",
    )
    p.add_argument(
        "--no-html-cache",
        action="store_true",
        help="処理後に当日HTMLを削除（Colab/ディスク節約向け）",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="N日処理ごとにチェックポイント保存",
    )
    args = p.parse_args()

    start = Date.fromisoformat(args.start)
    end = Date.fromisoformat(args.end)
    if end < start:
        print("ERROR: --end は --start 以降にしてください", file=sys.stderr)
        sys.exit(1)

    run(
        start,
        end,
        checkpoint_path=Path(args.checkpoint),
        resume=not args.no_resume,
        no_html_cache=args.no_html_cache,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
