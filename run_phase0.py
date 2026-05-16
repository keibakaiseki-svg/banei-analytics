"""
Phase 0 動作検証スクリプト。

ローカルに保存済みの HTML（data/raw_html/2026-05-04/race_mark_table_R01.html）を読み込み、
パーサが期待通り動作するか確認する。
"""

from pathlib import Path

import pandas as pd

from scrapers.race_page import parse_race_page

RAW_HTML = Path("data/raw_html/2026-05-04/race_mark_table_R01.html")


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 30)

    html = RAW_HTML.read_text(encoding="utf-8")
    info, result_df, payout_df = parse_race_page(html)

    print("=" * 60)
    print("RACE INFO")
    print("=" * 60)
    for k, v in info.__dict__.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 60)
    print("RESULT (rows={})".format(len(result_df)))
    print("=" * 60)
    print(result_df)

    print()
    print("=" * 60)
    print("PAYOUT (rows={})".format(len(payout_df)))
    print("=" * 60)
    print(payout_df)


if __name__ == "__main__":
    main()
