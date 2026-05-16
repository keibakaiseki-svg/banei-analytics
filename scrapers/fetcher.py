"""
レート制限付き HTML フェッチャ（生HTMLをディスクキャッシュ）。

ばんえい競馬 (帯広, baba_code=3) のRaceMarkTableを取得する。
"""

from __future__ import annotations

import time
from datetime import date as Date
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import httpx

BASE_URL = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/RaceMarkTable"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
BABA_CODE_OBIHIRO = 3
DEFAULT_CACHE_ROOT = Path("data/raw_html")
DEFAULT_RATE_LIMIT_SEC = 2.5


def _cache_path(race_date: Date, race_no: int, cache_root: Path) -> Path:
    return (
        cache_root
        / race_date.strftime("%Y-%m-%d")
        / f"race_mark_table_R{race_no:02d}.html"
    )


def _build_url(race_date: Date, race_no: int, baba_code: int) -> str:
    params = {
        "k_raceDate": race_date.strftime("%Y/%m/%d"),
        "k_raceNo": race_no,
        "k_babaCode": baba_code,
    }
    return f"{BASE_URL}?{urlencode(params)}"


def fetch_race(
    race_date: Date,
    race_no: int,
    *,
    baba_code: int = BABA_CODE_OBIHIRO,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    force_refresh: bool = False,
    client: Optional[httpx.Client] = None,
) -> Optional[str]:
    """
    指定レースのHTMLを取得。キャッシュがあれば再利用。
    レースが存在しない（404 or 中身が空）場合は None を返す。
    """
    cache = _cache_path(race_date, race_no, cache_root)
    if cache.exists() and not force_refresh:
        return cache.read_text(encoding="utf-8")

    cache.parent.mkdir(parents=True, exist_ok=True)
    url = _build_url(race_date, race_no, baba_code)

    own_client = client is None
    if own_client:
        client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0)
    try:
        resp = client.get(url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        html = resp.text
        # レースが存在しない日も200を返すサイトのため、簡易チェック
        if "成績表" not in html and "出馬表" not in html:
            return None
        cache.write_text(html, encoding="utf-8")
        return html
    finally:
        if own_client and client is not None:
            client.close()


def fetch_day(
    race_date: Date,
    *,
    baba_code: int = BABA_CODE_OBIHIRO,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    max_races: int = 12,
    rate_limit_sec: float = DEFAULT_RATE_LIMIT_SEC,
    force_refresh: bool = False,
) -> list[tuple[int, str]]:
    """
    1日分の全レースHTMLを取得。
    レート制限を守り、キャッシュヒット時はsleepしない。
    開催がない日は空リストを返す。
    """
    results: list[tuple[int, str]] = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
        for r in range(1, max_races + 1):
            cache = _cache_path(race_date, r, cache_root)
            cache_hit = cache.exists() and not force_refresh
            html = fetch_race(
                race_date,
                r,
                baba_code=baba_code,
                cache_root=cache_root,
                force_refresh=force_refresh,
                client=client,
            )
            if html is None:
                # それ以上のレースもないと判断
                break
            results.append((r, html))
            if not cache_hit:
                time.sleep(rate_limit_sec)
    return results
