"""
keiba.go.jp の OddsTanFuku ページから当日レースの最終オッズを取得する。

注意: このページは **当日 (or 直近) のみ参照可能** で、過去レースのオッズは
keiba.go.jp 側では取得できない。going-forward の運用で蓄積する設計。

URL: https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/OddsTanFuku?k_raceDate=YYYY/MM/DD&k_raceNo=N&k_babaCode=3

取得項目:
- race_id, snapshot_time
- horse_no, gate, horse_name
- win_odds (単勝), place_odds_low/high (複勝3着払い)
- body_weight_kg, body_weight_diff, load_weight_kg
- jockey_name, allowance_marker, trainer_name
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import httpx

BASE_URL = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/OddsTanFuku"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
BABA_CODE_OBIHIRO = 3
DEFAULT_RATE_LIMIT_SEC = 2.5
DEFAULT_CACHE_ROOT = Path("data/raw_html/odds")
ALLOWANCE_MARKERS = {"☆", "△", "▲", "◇", "★"}


@dataclass
class OddsRow:
    race_id: str
    snapshot_at_utc: str
    horse_no: int
    gate: Optional[int]
    horse_name: Optional[str]
    win_odds: Optional[float]
    place_odds_low: Optional[float]
    place_odds_high: Optional[float]
    sex_age: Optional[str]
    body_weight_kg: Optional[int]
    body_weight_diff_kg: Optional[int]
    load_weight_kg: Optional[int]
    jockey_name: Optional[str]
    allowance_marker: Optional[str]
    trainer_name: Optional[str]
    change_info: Optional[str]


def _build_url(race_date: Date, race_no: int, baba_code: int = BABA_CODE_OBIHIRO) -> str:
    params = {
        "k_raceDate": race_date.strftime("%Y/%m/%d"),
        "k_raceNo": race_no,
        "k_babaCode": baba_code,
    }
    return f"{BASE_URL}?{urlencode(params)}"


def fetch_odds_html(
    race_date: Date,
    race_no: int,
    *,
    baba_code: int = BABA_CODE_OBIHIRO,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    force_refresh: bool = True,
    client: Optional[httpx.Client] = None,
) -> Optional[str]:
    """オッズページを取得。複数snapshotを取れるよう、デフォルトは force_refresh=True。"""
    own_client = client is None
    if own_client:
        client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0)
    try:
        resp = client.get(_build_url(race_date, race_no, baba_code))
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        html = resp.text
        if "オッズの情報がありません" in html or "odd_popular_table_02" not in html:
            return None
        # snapshot キャッシュ (時刻付きファイル名)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cache_dir = cache_root / race_date.strftime("%Y-%m-%d")
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"R{race_no:02d}_{ts}.html").write_text(html, encoding="utf-8")
        return html
    finally:
        if own_client and client is not None:
            client.close()


def _to_float(s: str) -> Optional[float]:
    s = s.strip().rstrip("-")
    if not s or s == "－":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _split_jockey(raw: str) -> tuple[str, Optional[str]]:
    raw = raw.strip()
    if raw and raw[0] in ALLOWANCE_MARKERS:
        return raw[1:].strip(), raw[0]
    return raw, None


def parse_odds(html: str, race_id: str, snapshot_at_utc: str) -> list[OddsRow]:
    """HTMLから 1レース分の全馬のオッズ行を抽出。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    t = soup.find("table", class_="odd_popular_table_02")
    if t is None:
        return []
    rows = []
    for tr in t.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"], recursive=False)]
        # data 行は13セル (枠/馬番/馬名/単勝/複勝低/複勝高/性齢/体重(増減)/積載/騎手/所属/調教師/変更情報)
        if len(cells) < 12:
            continue
        # ヘッダ行を除外 (馬番が int 化できないなら skip)
        horse_no = _to_int(cells[1])
        if horse_no is None:
            continue
        # 馬体重 + 増減 (例: "1014 （0）")
        body_weight_kg = None
        body_weight_diff_kg = None
        bw_match = re.match(r"(\d+)\s*[（(]([+\-]?\d+)[）)]", cells[7])
        if bw_match:
            body_weight_kg = int(bw_match.group(1))
            body_weight_diff_kg = int(bw_match.group(2))
        # 性齢 cell
        # 13セルパターン: [枠, 馬番, 馬名, 単勝, 複勝低, 複勝高, 性齢, 体重, 積載, 騎手, 所属, 調教師, 変更]
        jockey_name, allowance_marker = _split_jockey(cells[9])
        rows.append(OddsRow(
            race_id=race_id,
            snapshot_at_utc=snapshot_at_utc,
            horse_no=horse_no,
            gate=_to_int(cells[0]),
            horse_name=cells[2] or None,
            win_odds=_to_float(cells[3]),
            place_odds_low=_to_float(cells[4]),
            place_odds_high=_to_float(cells[5]),
            sex_age=cells[6] or None,
            body_weight_kg=body_weight_kg,
            body_weight_diff_kg=body_weight_diff_kg,
            load_weight_kg=_to_int(cells[8]),
            jockey_name=jockey_name,
            allowance_marker=allowance_marker,
            trainer_name=cells[11] or None,
            change_info=(cells[12] if len(cells) > 12 else None) or None,
        ))
    return rows


def fetch_and_parse(
    race_date: Date,
    race_no: int,
    *,
    baba_code: int = BABA_CODE_OBIHIRO,
    client: Optional[httpx.Client] = None,
) -> list[OddsRow]:
    """1レース分の最終オッズを取得・パース。"""
    html = fetch_odds_html(race_date, race_no, baba_code=baba_code, client=client)
    if html is None:
        return []
    race_id = f"{race_date.strftime('%Y%m%d')}_{baba_code}_{race_no:02d}"
    snapshot_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return parse_odds(html, race_id, snapshot_at)
