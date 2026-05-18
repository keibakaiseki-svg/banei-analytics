"""
Walk-Forward v3: 単勝・複勝のEV比較で馬券種を自動選択する。

設計:
- 3つの LightGBM 二項分類器を訓練
  - target_win   : P(1着)
  - target_top3  : P(3着以内 = 複勝)
- 実netkeibaオッズで EV計算 (単勝 + 複勝範囲)
- 戦略:
  - bet_win_top1         : 単勝 (各レース予測最大馬・EV ≥ 1.0)
  - bet_place_top1       : 複勝 (top3予測最大馬・EV ≥ 1.0)
  - bet_auto_max_ev      : 各馬の単勝EV と複勝EV を比較、最大の方を採用
  - bet_auto_above_1.0   : 単複どちらでも EV>1.0 のもの全てベット

評価期間: 2024-01-01 〜 2026-04-30 (netkeiba 実オッズ100%)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from analytics.features import (
    FEATURE_COLS_BOOL,
    FEATURE_COLS_CATEGORICAL,
    FEATURE_COLS_NUMERIC,
    build_feature_matrix,
)
from analytics.walk_forward import month_range, prepare_features
from persist.parquet_writer import DEFAULT_PARQUET_ROOT


def get_real_odds(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> pd.DataFrame:
    df = pd.read_parquet(parquet_root / "odds_netkeiba.parquet")
    return df[["race_id_local", "horse_no", "win_odds", "place_odds_low", "place_odds_high"]].rename(
        columns={"race_id_local": "race_id"}
    )


def get_place_payouts(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> pd.DataFrame:
    """複勝の実払戻金 (1〜3着の馬それぞれ)。"""
    con = duckdb.connect()
    glob = str(parquet_root / "payouts/**/*.parquet")
    return con.execute(
        f"""
        SELECT race_id, CAST(combination AS INTEGER) AS horse_no, payout_yen AS place_payout
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE bet_type = '複勝' AND combination NOT LIKE '%-%'
        """
    ).fetchdf()


def run_v3(
    test_start: str = "2024-01-01",
    test_end: str = "2026-04-30",
    train_start: str = "2014-04-01",
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
) -> dict:
    import lightgbm as lgb

    print("=== 特徴量行列構築 ===")
    df = build_feature_matrix(parquet_root)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df[df["finish_pos"].notna()].copy()

    print("=== 実 win/place オッズ + 払戻 (netkeiba + payouts) ===")
    real_odds = get_real_odds(parquet_root)
    df = df.merge(real_odds, on=["race_id", "horse_no"], how="left")
    df = df.rename(columns={
        "win_odds": "real_win_odds",
        "place_odds_low": "real_place_low",
        "place_odds_high": "real_place_high",
    })
    # place 用は払戻金 (実)
    place_payout = get_place_payouts(parquet_root)
    df = df.merge(place_payout, on=["race_id", "horse_no"], how="left")

    df = prepare_features(df)
    feature_cols = FEATURE_COLS_NUMERIC + FEATURE_COLS_BOOL + FEATURE_COLS_CATEGORICAL

    months = month_range(test_start, test_end)
    monthly_records = []
    print(f"=== Walk-Forward ({test_start} 〜 {test_end}, {len(months)}ヶ月) ===")

    for m_idx, m in enumerate(months):
        train_cutoff = pd.Timestamp(m)
        test_cutoff = (train_cutoff + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()

        train = df[(df["race_date"] >= pd.Timestamp(train_start)) & (df["race_date"] < train_cutoff)]
        test = df[(df["race_date"] >= train_cutoff) & (df["race_date"] < test_cutoff)]
        if test.empty or len(train) < 1000:
            continue

        X_train = train[feature_cols]
        X_test = test[feature_cols].copy()

        # === 2つの分類器を訓練 ===
        model_win = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, verbose=-1, n_jobs=-1,
        )
        model_win.fit(X_train, train["target_win"], categorical_feature=FEATURE_COLS_CATEGORICAL)

        # target_place: 出走頭数仕様を反映した「複勝として払戻される」ラベル
        # 訓練は target_place が NULL でない行のみ
        train_place = train.dropna(subset=["target_place"])
        model_place = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, verbose=-1, n_jobs=-1,
        )
        model_place.fit(
            train_place[feature_cols],
            train_place["target_place"].astype(int),
            categorical_feature=FEATURE_COLS_CATEGORICAL,
        )

        test = test.copy()
        test["pred_p_win"] = model_win.predict_proba(X_test)[:, 1]
        test["pred_p_place"] = model_place.predict_proba(X_test)[:, 1]

        # EV 計算
        test["ev_win"] = test["pred_p_win"] * test["real_win_odds"]
        test["ev_place"] = test["pred_p_place"] * test["real_place_low"]

        # 4頭以下のレースは複勝なし → place 系の戦略から除外
        test["place_eligible"] = test["entry_count"] >= 5

        # === 戦略1: 単勝 Top1 (EV>1.0) ===
        # 単勝オッズは全馬分あるので dropna は実質ほぼ無処理 (バグ無し)
        idx_win = test.groupby("race_id")["pred_p_win"].idxmax()
        bets_win = test.loc[idx_win].copy()
        bets_win = bets_win[bets_win["ev_win"] > 1.0]
        bets_win = bets_win.dropna(subset=["real_win_odds"])  # netkeiba実オッズが必要
        if not bets_win.empty:
            paid = (bets_win[bets_win["finish_pos"] == 1]["real_win_odds"] * 100).sum()
            staked = len(bets_win) * 100
            wins = (bets_win["finish_pos"] == 1).sum()
            monthly_records.append({
                "month": m, "strategy": "win_top1_ev>1.0",
                "bets": len(bets_win), "wins": int(wins),
                "win_rate": wins / len(bets_win),
                "staked": staked, "paid": int(paid),
                "roi": round(paid / staked, 3),
            })

        # === 戦略2: 複勝 Top1 (EV>1.0) ===
        # 4頭以下は複勝なし → place_eligible=True のみ
        test_place = test[test["place_eligible"]].copy()
        if not test_place.empty:
            idx_place = test_place.groupby("race_id")["pred_p_place"].idxmax()
            bets_place = test_place.loc[idx_place].copy()
            bets_place = bets_place[bets_place["ev_place"] > 1.0]
            bets_place = bets_place.dropna(subset=["real_place_low"])
            if not bets_place.empty:
                # 払戻は payouts テーブルにあるものだけ加算 (実際に複勝が成立した馬のみ)
                paid = bets_place["place_payout"].fillna(0).astype(float).sum()
                staked = len(bets_place) * 100
                hits = bets_place["place_payout"].notna().sum()
                monthly_records.append({
                    "month": m, "strategy": "place_top1_ev>1.0",
                    "bets": len(bets_place), "wins": int(hits),
                    "win_rate": hits / len(bets_place),
                    "staked": staked, "paid": int(paid),
                    "roi": round(paid / staked, 3),
                })

        # === 戦略3: auto - 各馬の最大EV券種を選択 ===
        # 4頭以下のレースの ev_place は使えないため、placeが使えない馬は強制的にwin扱い
        test["effective_ev_place"] = np.where(test["place_eligible"], test["ev_place"], -1)
        test["ev_max"] = test[["ev_win", "effective_ev_place"]].max(axis=1)
        test["bet_type"] = np.where(test["ev_win"] >= test["effective_ev_place"], "win", "place")

        # 各レースで ev_max が最大の馬を選択
        idx_auto = test.groupby("race_id")["ev_max"].idxmax()
        bets_auto = test.loc[idx_auto].copy()
        bets_auto = bets_auto[bets_auto["ev_max"] > 1.0]

        if not bets_auto.empty:
            staked = len(bets_auto) * 100
            paid = 0
            hits = 0
            for _, row in bets_auto.iterrows():
                if row["bet_type"] == "win":
                    if row["finish_pos"] == 1 and pd.notna(row["real_win_odds"]):
                        paid += row["real_win_odds"] * 100
                        hits += 1
                else:  # place
                    if row["finish_pos"] <= 3 and pd.notna(row["place_payout"]):
                        paid += row["place_payout"]
                        hits += 1
            monthly_records.append({
                "month": m, "strategy": "auto_max_ev>1.0",
                "bets": len(bets_auto), "wins": hits,
                "win_rate": hits / len(bets_auto),
                "staked": staked, "paid": int(paid),
                "roi": round(paid / staked, 3),
            })

        # === 戦略4: auto しきい値別 (1.05, 1.10, 1.20) ===
        for thr in [1.05, 1.10, 1.20]:
            bets_t = bets_auto[bets_auto["ev_max"] > thr]
            if bets_t.empty:
                continue
            staked = len(bets_t) * 100
            paid = 0
            hits = 0
            for _, row in bets_t.iterrows():
                if row["bet_type"] == "win":
                    if row["finish_pos"] == 1 and pd.notna(row["real_win_odds"]):
                        paid += row["real_win_odds"] * 100
                        hits += 1
                else:
                    if row["finish_pos"] <= 3 and pd.notna(row["place_payout"]):
                        paid += row["place_payout"]
                        hits += 1
            monthly_records.append({
                "month": m, "strategy": f"auto_max_ev>{thr:.2f}",
                "bets": len(bets_t), "wins": hits,
                "win_rate": hits / len(bets_t),
                "staked": staked, "paid": int(paid),
                "roi": round(paid / staked, 3),
            })

        # === 比較ベンチマーク: popularity (1番人気・単勝) ===
        pop_test = test[test["popularity"] == 1]
        bets_pop = pop_test.groupby("race_id").first().reset_index().dropna(subset=["real_win_odds"])
        if not bets_pop.empty:
            paid = (bets_pop[bets_pop["finish_pos"] == 1]["real_win_odds"] * 100).sum()
            staked = len(bets_pop) * 100
            wins = (bets_pop["finish_pos"] == 1).sum()
            monthly_records.append({
                "month": m, "strategy": "popularity",
                "bets": len(bets_pop), "wins": int(wins),
                "win_rate": wins / len(bets_pop),
                "staked": staked, "paid": int(paid),
                "roi": round(paid / staked, 3),
            })

        if (m_idx + 1) % 6 == 0:
            print(f"  進捗: {m_idx+1}/{len(months)} ヶ月")

    monthly = pd.DataFrame(monthly_records)
    overall = (
        monthly.groupby("strategy")
        .agg(bets=("bets", "sum"), wins=("wins", "sum"),
             staked=("staked", "sum"), paid=("paid", "sum"))
        .reset_index()
    )
    overall["win_rate"] = (overall["wins"] / overall["bets"]).round(3)
    overall["roi"] = (overall["paid"] / overall["staked"]).round(3)

    return {"monthly": monthly, "overall": overall}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-start", default="2024-01-01")
    p.add_argument("--test-end", default="2026-04-30")
    p.add_argument("--output", default="data/backtest/v3_multi_bettype.parquet")
    args = p.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    res = run_v3(test_start=args.test_start, test_end=args.test_end)

    print()
    print("=== Walk-Forward v3 (馬券種自動選択) 結果 ===")
    print(res["overall"].sort_values("roi", ascending=False).to_string(index=False))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res["monthly"].to_parquet(out, index=False, compression="zstd")
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
