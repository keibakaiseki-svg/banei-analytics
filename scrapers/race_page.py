"""
keiba.go.jp の RaceMarkTable ページをパースしてばんえい競馬のレース情報を抽出する。

データ取得対象（Phase 0 仕様）:
- レース情報: 開催日、レース番号、距離、天候、馬場水分量、レース名、クラス、賞金
- 結果テーブル: 着順、枠番、馬番、馬名、性齢、積載重量、騎手、減量マーカー、調教師、
                馬体重、馬体重差、タイム(秒)、着差(タイムから逆算)、人気、完走ステータス
- 払戻: 単勝・複勝・馬連複・馬連単・ワイド・三連複・三連単

CAUTION:
- 着差列はサイト側で空欄。time_diff_from_winner として逆算する。
- 上り3F列もばんえいでは常時空欄。
- 「差」列は馬体重増減kg。前走比のkg差で符号付き。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlencode

import httpx
import pandas as pd
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/RaceMarkTable"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# ばんえい競馬の馬場コード = 3（帯広）
BABA_CODE_OBIHIRO = 3

# 騎手減量マーカー（実データで観察された記号。詳細kgはPhase以降で確定）
ALLOWANCE_MARKERS = {"☆", "△", "▲", "◇", "★"}


@dataclass
class RaceInfo:
    race_date: str          # YYYY-MM-DD
    race_no: int
    baba_code: int
    course_name: str        # "帯広"
    distance_m: int         # 200
    weather: Optional[str]  # "小雨" など
    track_water_pct: Optional[float]  # 馬場水分量
    race_name: Optional[str]
    race_class: Optional[str]  # "C2-15" など
    prizes: dict[int, int] = field(default_factory=dict)  # {着順: 円}


def fetch_race_page(
    race_date: str, race_no: int, baba_code: int = BABA_CODE_OBIHIRO
) -> str:
    """RaceMarkTable HTML を取得。race_date は 'YYYY/MM/DD' 形式。"""
    params = {
        "k_raceDate": race_date,
        "k_raceNo": race_no,
        "k_babaCode": baba_code,
    }
    url = f"{BASE_URL}?{urlencode(params)}"
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def _parse_time_to_seconds(text: str) -> Optional[float]:
    """'2:11.5' -> 131.5 秒。空文字や 'M:SS.s' 以外は None。"""
    text = text.strip()
    if not text:
        return None
    m = re.match(r"^(\d+):(\d+(?:\.\d+)?)$", text)
    if not m:
        return None
    minutes = int(m.group(1))
    seconds = float(m.group(2))
    return minutes * 60 + seconds


def _split_jockey_allowance(raw_jockey: str) -> tuple[str, Optional[str]]:
    """
    '☆今井千(ばんえい)' -> ('今井千(ばんえい)', '☆')
    減量マーカーが先頭にあれば分離。なければ (name, None)。
    """
    raw_jockey = raw_jockey.strip()
    if raw_jockey and raw_jockey[0] in ALLOWANCE_MARKERS:
        return raw_jockey[1:].strip(), raw_jockey[0]
    return raw_jockey, None


def _parse_race_header(soup: BeautifulSoup) -> RaceInfo:
    """レース情報ヘッダ部をパース。"""
    text = soup.get_text(" ", strip=True)

    # 開催日・コース・レース番号
    # 例: "2026年5月4日 （月）帯 広第１競走"
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日.*?帯\s*広第(\S+?)競走", text)
    race_date = "?"
    race_no = -1
    course_name = "帯広"
    if m:
        race_date = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        race_no = _zenkaku_num_to_int(m.group(4))

    # 距離
    distance_m = -1
    m = re.search(r"(\d+)ｍ", text)
    if m:
        distance_m = int(m.group(1))

    # 天候・馬場水分量
    weather = None
    track_water_pct: Optional[float] = None
    m = re.search(r"天候：(\S+?)\s*馬場：([\d.]+)", text)
    if m:
        weather = m.group(1)
        track_water_pct = float(m.group(2))
    else:
        # 単独で馬場のみ
        m = re.search(r"馬場：([\d.]+)", text)
        if m:
            track_water_pct = float(m.group(1))

    # レース名・クラス
    # パターン:
    #   R1  "飛花琳＆崇獅 初誕生記念Ｃ２－１５"
    #   R11 "菖蒲特別Ａ２－１混合"   ← クラスの後ろに '混合' が付く
    #   R12 "Ｂ４－３"                ← クラスのみ・レース名なし
    race_name = None
    race_class = None
    m = re.search(r"馬場：[\d.]+\s+(\S[^（]+?)\s+（", text)
    if m:
        raw = m.group(1).strip()
        cm = re.search(r"([Ａ-Ｚ][０-９][－\-]?[０-９]*)", raw)
        if cm:
            race_class = _zenkaku_to_hankaku(cm.group(1))
            before = raw[: cm.start()].strip()
            race_name = before or None
        else:
            race_name = raw

    # 賞金 1〜5着
    prizes: dict[int, int] = {}
    for pm in re.finditer(r"(\d)着\s+([\d,]+)円", text):
        prizes[int(pm.group(1))] = int(pm.group(2).replace(",", ""))

    return RaceInfo(
        race_date=race_date,
        race_no=race_no,
        baba_code=BABA_CODE_OBIHIRO,
        course_name=course_name,
        distance_m=distance_m,
        weather=weather,
        track_water_pct=track_water_pct,
        race_name=race_name,
        race_class=race_class,
        prizes=prizes,
    )


_ZENKAKU_DIGITS = "０１２３４５６７８９"


def _zenkaku_num_to_int(s: str) -> int:
    """'１' -> 1, '１２' -> 12。半角数字も許容。"""
    s = s.translate(str.maketrans(_ZENKAKU_DIGITS, "0123456789"))
    return int(s)


def _zenkaku_to_hankaku(s: str) -> str:
    table = str.maketrans(
        "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ－",
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-",
    )
    return s.translate(table)


def _find_result_table(soup: BeautifulSoup) -> Optional[Tag]:
    """成績表テーブルを特定。
    nested tables の get_text() は子孫を全部拾ってしまうため、
    「直下tdのうち最初の1セルが '着順'」かつ「直下tdに '馬名' 単独セル」を満たす
    tr を持つテーブルだけを最内テーブルとして採用する。"""
    for t in soup.find_all("table"):
        direct_trs = t.find_all("tr", recursive=False)
        if len(direct_trs) < 2:
            continue
        for tr in direct_trs:
            cells = tr.find_all("td", recursive=False)
            cell_texts = [c.get_text(strip=True) for c in cells]
            if not cell_texts:
                continue
            # 1セル目が '着順' 単独 & 馬名セルが単独で存在
            if cell_texts[0] == "着順" and "馬名" in cell_texts:
                return t
    return None


def _find_payout_table(soup: BeautifulSoup) -> Optional[Tag]:
    """払戻金テーブルを特定。
    1セル目に bet_type ヘッダ ('Ｒ' または '単勝') を持つ最内テーブルを採用。"""
    for t in soup.find_all("table"):
        direct_trs = t.find_all("tr", recursive=False)
        if len(direct_trs) < 3:
            continue
        for tr in direct_trs:
            cells = tr.find_all("td", recursive=False)
            cell_texts = [c.get_text(strip=True) for c in cells]
            if not cell_texts:
                continue
            if cell_texts[0] in {"Ｒ", "R"} and "単勝" in cell_texts:
                return t
    return None


def _parse_result_table(table: Tag, race_info: RaceInfo) -> pd.DataFrame:
    """
    成績表をパース。空欄(着差・上り3F)はそのまま保持し、time_diff_from_winner は別途計算。
    完走ステータス（finished / dead_heat / disqualified 等）も付与。
    """
    rows = table.find_all("tr", recursive=False)
    # ヘッダ行を特定（直下tdに "馬名" を含む行）
    header_idx = next(
        (
            i
            for i, r in enumerate(rows)
            if "馬名"
            in [c.get_text(strip=True) for c in r.find_all("td", recursive=False)]
        ),
        None,
    )
    if header_idx is None:
        raise ValueError("成績表のヘッダ行が見つかりませんでした")

    records = []
    for r in rows[header_idx + 1 :]:
        cells = r.find_all("td", recursive=False)
        if len(cells) < 15:
            continue
        text = [c.get_text(strip=True).replace("　", " ") for c in cells]

        raw_pos = text[0]
        finish_pos: Optional[int] = None
        finish_status = "finished"
        if raw_pos.isdigit():
            finish_pos = int(raw_pos)
        else:
            # イレギュラー: "失", "中", "故", "除", "取", "降"
            finish_status = {
                "失": "disqualified",
                "中": "cancelled",
                "故": "fell",
                "除": "scratched_late",
                "取": "scratched_early",
                "降": "demoted",
            }.get(raw_pos, "unknown")

        jockey_name, allowance_marker = _split_jockey_allowance(text[7])

        # 馬IDと騎手ID（hrefから抽出）
        horse_link = cells[3].find("a")
        horse_id = None
        if horse_link and "k_lineageLoginCode=" in horse_link.get("href", ""):
            m = re.search(r"k_lineageLoginCode=(\w+)", horse_link["href"])
            horse_id = m.group(1) if m else None
        jockey_link = cells[7].find("a")
        jockey_id = None
        if jockey_link and "k_riderLicenseNo=" in jockey_link.get("href", ""):
            m = re.search(r"k_riderLicenseNo=(\w+)", jockey_link["href"])
            jockey_id = m.group(1) if m else None
        trainer_link = cells[8].find("a")
        trainer_id = None
        if trainer_link and "k_trainerLicenseNo=" in trainer_link.get("href", ""):
            m = re.search(r"k_trainerLicenseNo=(\w+)", trainer_link["href"])
            trainer_id = m.group(1) if m else None

        records.append(
            {
                "race_date": race_info.race_date,
                "race_no": race_info.race_no,
                "raw_position_text": raw_pos,
                "finish_pos": finish_pos,
                "finish_status": finish_status,
                "post_position": _to_int(text[1]),
                "horse_no": _to_int(text[2]),
                "horse_id": horse_id,
                "horse_name": text[3],
                "affiliation": text[4],
                "sex_age": text[5],
                "load_weight_kg": _to_int(text[6]),
                "jockey_id": jockey_id,
                "jockey_name": jockey_name,
                "allowance_marker": allowance_marker,
                "trainer_id": trainer_id,
                "trainer_name": text[8],
                "body_weight_kg": _to_int(text[9]),
                "body_weight_diff_kg": _to_int(text[10]),
                "finish_time_sec": _parse_time_to_seconds(text[11]),
                "raw_margin_text": text[12],
                "raw_up_3f_text": text[13],
                "popularity": _to_int(text[14]),
            }
        )

    df = pd.DataFrame(records)

    # 着差（タイムから逆算）
    if not df.empty and df["finish_time_sec"].notna().any():
        winner_time = df["finish_time_sec"].min()
        df["time_diff_from_winner_sec"] = df["finish_time_sec"] - winner_time
        # 直前着差
        df_sorted = df.sort_values("finish_pos", na_position="last").reset_index()
        df_sorted["time_diff_from_prev_sec"] = df_sorted["finish_time_sec"].diff()
        df = df_sorted.set_index("index").sort_index()

    # 同着判定（同一 finish_pos が複数存在する場合）
    if not df.empty and df["finish_pos"].notna().any():
        dup_mask = df.duplicated(subset=["finish_pos"], keep=False) & df[
            "finish_pos"
        ].notna()
        df.loc[dup_mask, "finish_status"] = "dead_heat"

    return df


def _to_int(s: str) -> Optional[int]:
    s = s.strip()
    if not s or s in {"-", "−"}:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_payout_table(table: Tag) -> pd.DataFrame:
    """
    払戻金テーブルをパース。複勝/ワイドは複数組が <br> 区切りで詰まっているのを展開。
    """
    rows = table.find_all("tr", recursive=False)
    bet_types = ["単勝", "複勝", "馬連複", "馬連単", "ワイド", "三連複", "三連単"]

    # データ行を探す（21〜22セルで、'円'を含む）
    data_row = None
    for r in rows:
        cells = r.find_all("td", recursive=False)
        if len(cells) >= 21 and any("円" in c.get_text() for c in cells):
            data_row = cells
            break
    if data_row is None:
        return pd.DataFrame()

    # 先頭セルは R番号
    # その後、bet_type ごとに (組番, 払戻金, 人気) が並ぶ
    # 複勝とワイドは複数組（最大3組）が <br> 区切り
    records = []
    cursor = 1  # 0番セルは R番号
    for bet in bet_types:
        if cursor + 2 >= len(data_row):
            break
        combo_cell = data_row[cursor]
        payout_cell = data_row[cursor + 1]
        pop_cell = data_row[cursor + 2]

        # <br>区切りで複数値が入る場合に対応
        combos = _split_br(combo_cell)
        payouts = _split_br(payout_cell)
        pops = _split_br(pop_cell)

        # 数が合わない場合は最小数だけ取る
        n = min(len(combos), len(payouts), len(pops))
        for i in range(n):
            records.append(
                {
                    "bet_type": bet,
                    "combination": combos[i],
                    "payout_yen": _clean_yen(payouts[i]),
                    "popularity": _to_int(pops[i]),
                }
            )
        cursor += 3

    return pd.DataFrame(records)


def _split_br(cell: Tag) -> list[str]:
    """セル内の <br> で区切られたテキストをリスト化。連結テキストの場合も分割を試みる。"""
    # まず <br> 分割
    parts = []
    for el in cell.children:
        if isinstance(el, str):
            t = el.strip()
            if t:
                parts.append(t)
        elif el.name == "br":
            continue
        else:
            t = el.get_text(strip=True)
            if t:
                parts.append(t)
    # 1要素しか取れない場合は連結文字列を分割（例: '130円310円160円'）
    if len(parts) == 1:
        s = parts[0]
        # '円'区切り
        if s.count("円") >= 2:
            return [p + "円" for p in s.split("円") if p]
        # 数字パターン分割（複勝の組番 '417' -> ['4','1','7'] は今回はそのまま）
    return parts


def _clean_yen(s: str) -> Optional[int]:
    s = s.replace(",", "").replace("円", "").strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_race_page(html: str) -> tuple[RaceInfo, pd.DataFrame, pd.DataFrame]:
    """HTMLからレース情報・結果・払戻をまとめてパース。"""
    soup = BeautifulSoup(html, "lxml")
    info = _parse_race_header(soup)

    result_tbl = _find_result_table(soup)
    result_df = (
        _parse_result_table(result_tbl, info) if result_tbl is not None else pd.DataFrame()
    )

    payout_tbl = _find_payout_table(soup)
    payout_df = _parse_payout_table(payout_tbl) if payout_tbl is not None else pd.DataFrame()

    return info, result_df, payout_df
