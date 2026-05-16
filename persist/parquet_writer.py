"""
パース結果を月次パーティション付き Parquet に永続化する。

ディレクトリレイアウト（Hive パーティション）:
  data/parquet/<table>/year=YYYY/month=MM/data.parquet

主要テーブル:
  - races    : 1 row per race
  - entries  : 1 row per (race_id, horse_no) — 出馬+結果を統合
  - payouts  : 1 row per (race_id, bet_type, combination)

冪等性: 同一 PK の行が既存ファイルに含まれている場合は **新しい値で上書き**。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from scrapers.race_page import RaceInfo

DEFAULT_PARQUET_ROOT = Path("data/parquet")


def make_race_id(race_date: str, race_no: int, baba_code: int) -> str:
    """race_id = YYYYMMDD_BB_NN  例: '20260504_3_01'"""
    return f"{race_date.replace('-', '')}_{baba_code}_{race_no:02d}"


def race_info_to_row(info: RaceInfo, *, entry_count: int, source_url: str) -> dict:
    race_id = make_race_id(info.race_date, info.race_no, info.baba_code)
    return {
        "race_id": race_id,
        "race_date": info.race_date,
        "race_no": info.race_no,
        "baba_code": info.baba_code,
        "course_name": info.course_name,
        "distance_m": info.distance_m,
        "weather": info.weather,
        "track_water_pct": info.track_water_pct,
        "race_name": info.race_name,
        "race_class": info.race_class,
        "entry_count": entry_count,
        "prize_1st": info.prizes.get(1),
        "prize_2nd": info.prizes.get(2),
        "prize_3rd": info.prizes.get(3),
        "prize_4th": info.prizes.get(4),
        "prize_5th": info.prizes.get(5),
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "source_url": source_url,
    }


def attach_race_id(
    df: pd.DataFrame, info: RaceInfo, *, drop_old_keys: bool = True
) -> pd.DataFrame:
    """entries/payouts DataFrameに race_id 列を付与し、レース内ローカル列を整理。"""
    if df.empty:
        return df
    race_id = make_race_id(info.race_date, info.race_no, info.baba_code)
    df = df.copy()
    df.insert(0, "race_id", race_id)
    if drop_old_keys:
        # entries テーブルの race_date/race_no は races テーブルにあるので残置不要
        for c in ("race_date", "race_no"):
            if c in df.columns:
                df = df.drop(columns=c)
    return df


def _partition_path(
    table: str, race_date: str, parquet_root: Path = DEFAULT_PARQUET_ROOT
) -> Path:
    y, m, _ = race_date.split("-")
    return parquet_root / table / f"year={y}" / f"month={m}" / "data.parquet"


def _upsert_parquet(
    path: Path, new_df: pd.DataFrame, primary_keys: list[str]
) -> tuple[int, int]:
    """
    既存Parquetがあれば読み込み、PK重複は new_df の値で上書きしてマージ。
    戻り値: (replaced_rows, added_rows)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if new_df.empty:
        return 0, 0

    if path.exists():
        existing = pd.read_parquet(path)
        # 重複しているPKを既存から除外
        merge_keys = new_df[primary_keys].drop_duplicates()
        dup_mask = existing.set_index(primary_keys).index.isin(
            merge_keys.set_index(primary_keys).index
        )
        replaced = int(dup_mask.sum())
        kept = existing.loc[~dup_mask]
        combined = pd.concat([kept, new_df], ignore_index=True)
        added = len(new_df) - replaced
    else:
        combined = new_df
        replaced, added = 0, len(new_df)

    combined.to_parquet(path, index=False, compression="zstd")
    return replaced, added


def write_race(
    info: RaceInfo,
    entries_df: pd.DataFrame,
    payouts_df: pd.DataFrame,
    *,
    source_url: str,
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
) -> dict:
    """1レース分の3テーブルを永続化。"""
    race_row = pd.DataFrame([race_info_to_row(info, entry_count=len(entries_df), source_url=source_url)])
    entries = attach_race_id(entries_df, info)
    payouts = attach_race_id(payouts_df, info)

    summary: dict = {}
    summary["races"] = _upsert_parquet(
        _partition_path("races", info.race_date, parquet_root), race_row, ["race_id"]
    )
    summary["entries"] = _upsert_parquet(
        _partition_path("entries", info.race_date, parquet_root),
        entries,
        ["race_id", "horse_no"],
    )
    summary["payouts"] = _upsert_parquet(
        _partition_path("payouts", info.race_date, parquet_root),
        payouts,
        ["race_id", "bet_type", "combination"],
    )
    return summary
