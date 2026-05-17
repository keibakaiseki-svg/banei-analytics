"""
netkeiba NAR から **過去レースの最終オッズ** を取得する。

特徴:
- keiba.go.jp の OddsTanFuku は当日のみだが、netkeiba は過去レースもオッズ閲覧可能
- 単勝・複勝オッズ (全馬・確定値)
- 文字コードは EUC-JP

URL: https://nar.netkeiba.com/odds/?race_id=YYYYJJMMDDRR&type=b1
  YYYY: 年
  JJ:   netkeiba 競馬場コード (帯広 = 65)
  MMDD: 月日
  RR:   レース番号 (2桁)

例: 2020/05/02 第1R 帯広 → race_id = 202065050201
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import httpx

BASE_URL = "https://nar.netkeiba.com/odds/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
JYO_CODE_OBIHIRO = 65  # netkeiba 帯広コード
DEFAULT_RATE_LIMIT_SEC = 1.5
DEFAULT_CACHE_ROOT = Path("data/raw_html/netkeiba_odds")


@dataclass
class NetkeibaOddsRow:
    race_id_local: str       # 我々の race_id (YYYYMMDD_3_NN)
    race_id_netkeiba: str    # netkeiba の race_id
    horse_no: int
    gate: Optional[int]
    horse_name: Optional[str]
    win_odds: Optional[float]
    place_odds_low: Optional[float]
    place_odds_high: Optional[float]


def local_race_id_to_netkeiba(race_id_local: str, jyo_code: int = JYO_CODE_OBIHIRO) -> str:
    """20260502_3_01 → 202665050201"""
    parts = race_id_local.split("_")
    if len(parts) != 3:
        raise ValueError(f"invalid local race_id: {race_id_local}")
    date_str = parts[0]  # YYYYMMDD
    race_no = int(parts[2])
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    return f"{year}{jyo_code:02d}{month}{day}{race_no:02d}"


def netkeiba_race_id_from_date(race_date: Date, race_no: int, jyo_code: int = JYO_CODE_OBIHIRO) -> str:
    return f"{race_date.year}{jyo_code:02d}{race_date.month:02d}{race_date.day:02d}{race_no:02d}"


def _build_url(netkeiba_race_id: str, bet_type: str = "b1") -> str:
    return f"{BASE_URL}?{urlencode({'race_id': netkeiba_race_id, 'type': bet_type})}"


def fetch_odds_html(
    netkeiba_race_id: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    force_refresh: bool = False,
    client: Optional[httpx.Client] = None,
) -> Optional[str]:
    cache_dir = cache_root
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{netkeiba_race_id}.html"
    if cache.exists() and not force_refresh:
        return cache.read_text(encoding="utf-8")

    own_client = client is None
    if own_client:
        client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0)
    try:
        resp = client.get(_build_url(netkeiba_race_id))
        # 400/404 は「該当レースなし」として扱う (netkeiba は古い無効レースを 400 で返す)
        if resp.status_code in (400, 404):
            return None
        resp.raise_for_status()
        # netkeiba は EUC-JP
        html = resp.content.decode("euc-jp", errors="replace")
        if "RaceOdds_HorseList_Table" not in html:
            return None
        cache.write_text(html, encoding="utf-8")
        return html
    finally:
        if own_client and client is not None:
            client.close()


def _to_float(s: str) -> Optional[float]:
    s = s.strip()
    if not s or s in {"-", "--", "---", "－"}:
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


def parse_netkeiba_odds(
    html: str, race_id_local: str, race_id_netkeiba: str
) -> list[NetkeibaOddsRow]:
    """HTMLから単勝・複勝オッズを抽出。
    netkeiba のオッズページは Table 0=単勝, Table 1=複勝 の構造。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table", class_="RaceOdds_HorseList_Table")
    if len(tables) < 2:
        return []

    # 単勝オッズ (table 0): 枠/馬番/印/馬名/オッズ
    win_map: dict[int, dict] = {}
    for tr in tables[0].find_all("tr")[1:]:
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 5:
            continue
        horse_no = _to_int(cells[1])
        if horse_no is None:
            continue
        win_map[horse_no] = {
            "gate": _to_int(cells[0]),
            "horse_name": cells[3] or None,
            "win_odds": _to_float(cells[4]),
        }

    # 複勝オッズ (table 1): "X - Y" 形式
    place_map: dict[int, tuple[Optional[float], Optional[float]]] = {}
    for tr in tables[1].find_all("tr")[1:]:
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 5:
            continue
        horse_no = _to_int(cells[1])
        if horse_no is None:
            continue
        odds_text = cells[4]
        low = high = None
        m = re.match(r"^([\d.]+)\s*-\s*([\d.]+)$", odds_text)
        if m:
            low = float(m.group(1))
            high = float(m.group(2))
        place_map[horse_no] = (low, high)

    # 統合
    rows: list[NetkeibaOddsRow] = []
    for horse_no, w in win_map.items():
        low, high = place_map.get(horse_no, (None, None))
        rows.append(NetkeibaOddsRow(
            race_id_local=race_id_local,
            race_id_netkeiba=race_id_netkeiba,
            horse_no=horse_no,
            gate=w["gate"],
            horse_name=w["horse_name"],
            win_odds=w["win_odds"],
            place_odds_low=low,
            place_odds_high=high,
        ))
    return rows


def fetch_and_parse_by_local_id(
    race_id_local: str,
    *,
    jyo_code: int = JYO_CODE_OBIHIRO,
    client: Optional[httpx.Client] = None,
) -> list[NetkeibaOddsRow]:
    netkeiba_id = local_race_id_to_netkeiba(race_id_local, jyo_code)
    html = fetch_odds_html(netkeiba_id, client=client)
    if html is None:
        return []
    return parse_netkeiba_odds(html, race_id_local, netkeiba_id)
