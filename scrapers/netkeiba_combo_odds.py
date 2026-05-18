"""
netkeiba NAR から **過去レースの確定オッズ** (馬連・馬単・ワイド・三連複・三連単) を取得する。

netkeiba NAR の組合せオッズは AJAX エンドポイント `odds_get_form.html` から
取得できる。HTML は `<table class="Odds_Table">` の連続(caption-style multi-table):
  - 1行目=1セル(軸となる馬番のキャプション)
  - 2行目以降=[相手馬番, オッズ]

仕様メモ:
  - 馬連/ワイド/馬単: 1リクエスト/レース で全組合せ取得
  - 三連複: `&jiku=N` で軸馬指定。N=1..(entry-2) を iterate して dedup
  - 三連単: `&jiku=N` で 1着固定馬指定。N=1..entry を iterate
  - ワイドのみオッズ範囲 "low - high"、それ以外は単一値
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import httpx

from scrapers.netkeiba_odds import (
    DEFAULT_RATE_LIMIT_SEC,
    JYO_CODE_OBIHIRO,
    USER_AGENT,
    local_race_id_to_netkeiba,
)

AJAX_URL = "https://nar.netkeiba.com/odds/odds_get_form.html"
DEFAULT_COMBO_CACHE_ROOT = Path("data/raw_html/netkeiba_combo_odds")

# bet_type → メタ情報
BET_TYPES: dict[str, dict] = {
    "umaren":     {"param": "b4", "ordered": False, "size": 2, "label": "馬連",   "needs_axis": False},
    "wide":       {"param": "b5", "ordered": False, "size": 2, "label": "ワイド", "needs_axis": False},
    "umatan":     {"param": "b6", "ordered": True,  "size": 2, "label": "馬単",   "needs_axis": False},
    "sanrenpuku": {"param": "b7", "ordered": False, "size": 3, "label": "三連複", "needs_axis": True},
    "sanrentan":  {"param": "b8", "ordered": True,  "size": 3, "label": "三連単", "needs_axis": True},
}


@dataclass
class ComboOddsRow:
    race_id_local: str
    race_id_netkeiba: str
    bet_type: str
    combination: str          # 正規化済 "3-7" or "3-7-10" (馬単/三連単は順序保持)
    odds_min: float
    odds_max: Optional[float]  # ワイドのみ高値、それ以外は None


# ============================================================================
# Fetch
# ============================================================================

def _build_combo_url(netkeiba_race_id: str, bet_type: str, jiku: Optional[int] = None) -> str:
    params = {"type": BET_TYPES[bet_type]["param"], "race_id": netkeiba_race_id}
    if jiku is not None:
        params["jiku"] = str(jiku)
    return f"{AJAX_URL}?{urlencode(params)}"


def _cache_path(cache_root: Path, bet_type: str, netkeiba_race_id: str, jiku: Optional[int]) -> Path:
    name = f"{netkeiba_race_id}.html" if jiku is None else f"{netkeiba_race_id}_jiku{jiku}.html"
    return cache_root / bet_type / name


def fetch_combo_html(
    netkeiba_race_id: str,
    bet_type: str,
    *,
    jiku: Optional[int] = None,
    cache_root: Path = DEFAULT_COMBO_CACHE_ROOT,
    force_refresh: bool = False,
    client: Optional[httpx.Client] = None,
) -> Optional[str]:
    """1ページ(=1軸 or 軸不要券種) ぶんを取得・キャッシュ。"""
    if bet_type not in BET_TYPES:
        raise ValueError(f"unknown bet_type: {bet_type}")
    cache = _cache_path(cache_root, bet_type, netkeiba_race_id, jiku)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not force_refresh:
        return cache.read_text(encoding="utf-8")

    own_client = client is None
    if own_client:
        client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0)
    try:
        resp = client.get(_build_combo_url(netkeiba_race_id, bet_type, jiku))
        if resp.status_code in (400, 404):
            return None
        resp.raise_for_status()
        html = resp.content.decode("euc-jp", errors="replace")
        # オッズテーブルの有無で実データかを判定
        if "Odds_Table" not in html:
            return None
        cache.write_text(html, encoding="utf-8")
        return html
    finally:
        if own_client and client is not None:
            client.close()


# ============================================================================
# Parse: caption-style multi-table
# ============================================================================

_NUM_PAT = re.compile(r"[\d,]+\.\d+")


def parse_combo_odds_html(
    html: str,
    bet_type: str,
    race_id_local: str,
    race_id_netkeiba: str,
    *,
    axis_horse: Optional[int] = None,
) -> list[ComboOddsRow]:
    """1ページ分のHTML(=1軸 or 軸不要券種) をパース。

    netkeiba NAR の `<table class="Odds_Table">` はテーブルごとに
      行0=1セル(その表が表す軸馬の番号 caption_horse)
      行1+=[相手馬番, オッズ(or low-high)]
    の caption-style 形式。

    size=2 (馬連/ワイド/馬単): combination = (caption_horse, row_horse)
    size=3 + axis_horse 指定 (三連複/三連単): combination = (axis_horse, caption_horse, row_horse)
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")

    info = BET_TYPES[bet_type]
    size = info["size"]
    ordered = info["ordered"]
    if size == 3 and axis_horse is None:
        raise ValueError(f"{bet_type} requires axis_horse")

    rows_out: list[ComboOddsRow] = []
    seen: set[str] = set()

    for table in soup.find_all("table", class_="Odds_Table"):
        trs = table.find_all("tr")
        if len(trs) < 2:
            continue
        # 1行目から caption_horse 抽出
        first_cells = [c.get_text(" ", strip=True) for c in trs[0].find_all(["td", "th"])]
        if not first_cells:
            continue
        try:
            caption_horse = int(first_cells[0])
        except (ValueError, TypeError):
            continue
        if not (1 <= caption_horse <= 30):
            continue

        for tr in trs[1:]:
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if len(cells) < 2:
                continue
            try:
                row_horse = int(cells[0])
            except ValueError:
                continue
            if not (1 <= row_horse <= 30):
                continue
            odds_text = " ".join(cells[1:])
            nums = _NUM_PAT.findall(odds_text)
            if not nums:
                continue
            odds_min = float(nums[0].replace(",", ""))
            odds_max: Optional[float] = None
            if bet_type == "wide" and len(nums) >= 2:
                odds_max = float(nums[1].replace(",", ""))

            if size == 2:
                combo_nums: tuple[int, ...] = (caption_horse, row_horse)
            else:
                combo_nums = (axis_horse, caption_horse, row_horse)
            # 重複馬番を含む組合せは捨てる
            if len(set(combo_nums)) != len(combo_nums):
                continue
            if not ordered:
                combo_nums = tuple(sorted(combo_nums))
            combo_str = "-".join(str(n) for n in combo_nums)
            if combo_str in seen:
                continue
            seen.add(combo_str)
            rows_out.append(ComboOddsRow(
                race_id_local=race_id_local,
                race_id_netkeiba=race_id_netkeiba,
                bet_type=bet_type,
                combination=combo_str,
                odds_min=odds_min,
                odds_max=odds_max,
            ))

    return rows_out


# ============================================================================
# Race-level orchestration (axis iteration & dedup)
# ============================================================================

def _axis_range(bet_type: str, entry_count: int) -> list[int]:
    """軸馬として叩く範囲。
    三連複: 最小元が 1..N-2 の組合せで網羅。jiku=1..N-2
    三連単: 1着固定 = 1..N
    """
    if bet_type == "sanrenpuku":
        return list(range(1, max(1, entry_count - 1)))  # 1..N-2 inclusive
    if bet_type == "sanrentan":
        return list(range(1, entry_count + 1))  # 1..N
    return []


def fetch_and_parse_combo_for_race(
    race_id_local: str,
    bet_type: str,
    entry_count: int,
    *,
    jyo_code: int = JYO_CODE_OBIHIRO,
    client: Optional[httpx.Client] = None,
    cache_root: Path = DEFAULT_COMBO_CACHE_ROOT,
    rate_limit_sec: float = DEFAULT_RATE_LIMIT_SEC,
) -> tuple[list[ComboOddsRow], int]:
    """1レース1券種ぶんの全組合せを取得 (軸馬iterate含む)。

    戻り値: (parsed_rows, http_requests_made)
    """
    netkeiba_id = local_race_id_to_netkeiba(race_id_local, jyo_code)
    info = BET_TYPES[bet_type]
    own_client = client is None
    if own_client:
        client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0)
    req_count = 0
    try:
        if not info["needs_axis"]:
            cache_hit = _cache_path(cache_root, bet_type, netkeiba_id, None).exists()
            html = fetch_combo_html(netkeiba_id, bet_type, client=client, cache_root=cache_root)
            if not cache_hit:
                req_count += 1
                time.sleep(rate_limit_sec)
            if html is None:
                return [], req_count
            return parse_combo_odds_html(html, bet_type, race_id_local, netkeiba_id), req_count

        # 軸iterate
        rows_by_combo: dict[str, ComboOddsRow] = {}
        for jiku in _axis_range(bet_type, entry_count):
            cache_hit = _cache_path(cache_root, bet_type, netkeiba_id, jiku).exists()
            html = fetch_combo_html(netkeiba_id, bet_type, jiku=jiku, client=client, cache_root=cache_root)
            if not cache_hit:
                req_count += 1
                time.sleep(rate_limit_sec)
            if html is None:
                continue
            for r in parse_combo_odds_html(
                html, bet_type, race_id_local, netkeiba_id, axis_horse=jiku
            ):
                # 三連複は軸間overlap → 最初に出会ったオッズで固定 (どの軸も同じはず)
                if r.combination not in rows_by_combo:
                    rows_by_combo[r.combination] = r
        return list(rows_by_combo.values()), req_count
    finally:
        if own_client and client is not None:
            client.close()


def fetch_all_bet_types_for_race(
    race_id_local: str,
    entry_count: int,
    *,
    bet_types: Optional[list[str]] = None,
    jyo_code: int = JYO_CODE_OBIHIRO,
    client: Optional[httpx.Client] = None,
    cache_root: Path = DEFAULT_COMBO_CACHE_ROOT,
    rate_limit_sec: float = DEFAULT_RATE_LIMIT_SEC,
) -> dict[str, list[ComboOddsRow]]:
    """1レース分、全(指定)券種をまとめて取得。テスト/単発予測用。"""
    bet_types = bet_types or list(BET_TYPES.keys())
    result: dict[str, list[ComboOddsRow]] = {}

    own_client = client is None
    if own_client:
        client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0)
    try:
        for bt in bet_types:
            rows, _ = fetch_and_parse_combo_for_race(
                race_id_local, bt, entry_count,
                jyo_code=jyo_code, client=client,
                cache_root=cache_root, rate_limit_sec=rate_limit_sec,
            )
            result[bt] = rows
        return result
    finally:
        if own_client and client is not None:
            client.close()
