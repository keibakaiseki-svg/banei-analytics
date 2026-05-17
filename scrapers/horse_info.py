"""
keiba.go.jp の馬詳細(血統)ページから血統情報を抽出する。

URL pattern:
  https://www.keiba.go.jp/KeibaWeb/DataRoom/RaceHorseInfo?k_lineageLoginCode=XXX&k_activeCode=1

抽出データ:
  - 馬名, 性別, 毛色, 生年月日, 産地, 生産牧場, 馬主, 調教師
  - 父・父父・父母・母・母父・母母 (品種付き: 日輓/半血/中半血 等)
  - 地方収得賞金, 中央収得賞金, 中央付加賞金
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import httpx

BASE_URL = "https://www.keiba.go.jp/KeibaWeb/DataRoom/RaceHorseInfo"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
DEFAULT_RATE_LIMIT_SEC = 2.5
DEFAULT_CACHE_ROOT = Path("data/raw_html/horse_info")


@dataclass
class HorseInfo:
    horse_id: str
    horse_name: Optional[str] = None
    sex: Optional[str] = None
    age: Optional[int] = None
    status: Optional[str] = None  # 現役/引退
    breed: Optional[str] = None  # 日輓/半血/中半血 等
    color: Optional[str] = None
    birth_date: Optional[str] = None  # YYYY-MM-DD
    trainer: Optional[str] = None
    owner: Optional[str] = None
    breeder: Optional[str] = None
    birth_place: Optional[str] = None
    sire: Optional[str] = None
    sire_breed: Optional[str] = None
    sire_sire: Optional[str] = None
    sire_sire_breed: Optional[str] = None
    sire_dam: Optional[str] = None
    sire_dam_breed: Optional[str] = None
    dam: Optional[str] = None
    dam_breed: Optional[str] = None
    dam_sire: Optional[str] = None
    dam_sire_breed: Optional[str] = None
    dam_dam: Optional[str] = None
    dam_dam_breed: Optional[str] = None
    earnings_local: Optional[int] = None
    earnings_central: Optional[int] = None
    earnings_central_bonus: Optional[int] = None


def _build_url(horse_id: str) -> str:
    return f"{BASE_URL}?{urlencode({'k_lineageLoginCode': horse_id, 'k_activeCode': 1})}"


def fetch_horse_html(
    horse_id: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    force_refresh: bool = False,
    client: Optional[httpx.Client] = None,
) -> Optional[str]:
    cache = cache_root / f"{horse_id}.html"
    if cache.exists() and not force_refresh:
        return cache.read_text(encoding="utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)

    own_client = client is None
    if own_client:
        client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0)
    try:
        resp = client.get(_build_url(horse_id))
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        html = resp.text
        if "血統表" not in html:
            return None
        cache.write_text(html, encoding="utf-8")
        return html
    finally:
        if own_client and client is not None:
            client.close()


_PEDIGREE_KEYS = ["父父", "父母", "母父", "母母", "父", "母"]


def _parse_pedigree_segment(text: str) -> dict[str, tuple[Optional[str], Optional[str]]]:
    """血統表セクションから各続柄を抽出。"""
    out: dict[str, tuple[Optional[str], Optional[str]]] = {}
    # 父父・父母・母父・母母 は2文字キー優先で探す。残り「父」「母」は単独で。
    # 各キーごとに「キー （品種）名前」を捕捉
    m = re.search(r"血統表(.*?)(着別回数|$)", text, re.DOTALL)
    seg = m.group(1) if m else text
    for key in _PEDIGREE_KEYS:
        # 「キー （品種）名前」 → 品種, 名前
        # キーの直後がスペース、その後 （..）, その後ホースネーム
        pattern = rf"(?<!\w){re.escape(key)}\s+（([^）]+)）\s*([぀-ヿ一-鿿A-Za-zＡ-Ｚ０-９0-9・ー\s]+?)(?=\s*(?:父父|父母|母父|母母|父|母|着別回数|生涯|$))"
        m = re.search(pattern, seg)
        if m:
            out[key] = (m.group(1).strip(), m.group(2).strip())
    return out


_PRICE_PAT = re.compile(r"([\d,]+)")


def _to_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    s = s.replace(",", "").strip()
    try:
        return int(s)
    except ValueError:
        return None


def parse_horse_info(html: str, horse_id: str) -> HorseInfo:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    info = HorseInfo(horse_id=horse_id)

    # 馬名・性別・年齢・状態・品種
    # 例: "シオン 牝 3 現役 （日輓）"
    m = re.search(r"([^\s（]+)\s+([牡牝騙])\s+(\d+)\s+(現役|引退)\s+（([^）]+)）", text)
    if m:
        info.horse_name = m.group(1)
        info.sex = m.group(2)
        info.age = int(m.group(3))
        info.status = m.group(4)
        info.breed = m.group(5)

    # 生年月日: "生年月日 2023.05.24生"
    m = re.search(r"生年月日\s*(\d{4})\.(\d{1,2})\.(\d{1,2})", text)
    if m:
        info.birth_date = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # 調教師: "調教師 久　田　　　　守（ばんえい）"
    m = re.search(r"調教師\s*(.+?)（", text)
    if m:
        info.trainer = re.sub(r"\s+", "", m.group(1))

    # 毛色
    m = re.search(r"毛色\s*([^\s]+?)\s+馬主", text)
    if m:
        info.color = m.group(1).strip()

    # 馬主: "馬主 佐々木　　正　人 中央収得賞金"
    m = re.search(r"馬主\s*(.+?)\s+中央収得賞金", text)
    if m:
        info.owner = re.sub(r"\s+", "", m.group(1))

    # 産地
    m = re.search(r"産地\s*([^\s]+)\s+生産牧場", text)
    if m:
        info.birth_place = m.group(1).strip()

    # 生産牧場
    m = re.search(r"生産牧場\s*(.+?)\s+中央付加賞金", text)
    if m:
        info.breeder = re.sub(r"\s+", "", m.group(1))

    # 賞金
    m = re.search(r"地方収得賞金\s*([\d,]+)", text)
    if m:
        info.earnings_local = _to_int(m.group(1))
    m = re.search(r"中央収得賞金\s*([\d,]+)", text)
    if m:
        info.earnings_central = _to_int(m.group(1))
    m = re.search(r"中央付加賞金\s*([\d,]+)", text)
    if m:
        info.earnings_central_bonus = _to_int(m.group(1))

    # 血統
    ped = _parse_pedigree_segment(text)
    info.sire_breed, info.sire = ped.get("父", (None, None))
    info.sire_sire_breed, info.sire_sire = ped.get("父父", (None, None))
    info.sire_dam_breed, info.sire_dam = ped.get("父母", (None, None))
    info.dam_breed, info.dam = ped.get("母", (None, None))
    info.dam_sire_breed, info.dam_sire = ped.get("母父", (None, None))
    info.dam_dam_breed, info.dam_dam = ped.get("母母", (None, None))

    return info


def fetch_and_parse(horse_id: str, **kwargs) -> Optional[HorseInfo]:
    html = fetch_horse_html(horse_id, **kwargs)
    if html is None:
        return None
    return parse_horse_info(html, horse_id)
