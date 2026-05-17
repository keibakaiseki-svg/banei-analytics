"""
Walk-Forward backtest: 月次再学習で 単勝勝率予測モデル を評価する。

設計:
- 訓練窓: Expanding (初期 N年 → 毎月拡張)
- モデル: LightGBM 二項分類 (target_win)
- 評価: 各テスト月の予測確率最大馬に 100円 を 単勝で賭けた場合のROI
- 比較ベースライン:
  - random   : 各レースランダム馬選択
  - lowest_no: 各レース最小馬番
  - popularity: 各レース1番人気
  - model    : LightGBM 予測最大馬
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from analytics.features import (
    FEATURE_COLS_BOOL,
    FEATURE_COLS_CATEGORICAL,
    FEATURE_COLS_NUMERIC,
    build_feature_matrix,
)
from persist.parquet_writer import DEFAULT_PARQUET_ROOT


def month_range(start: str, end: str) -> list[str]:
    """月初の文字列リストを返す ['YYYY-MM-01']."""
    s = pd.Timestamp(start).to_period("M").to_timestamp()
    e = pd.Timestamp(end).to_period("M").to_timestamp()
    out = []
    cur = s
    while cur <= e:
        out.append(cur.strftime("%Y-%m-%d"))
        cur = (cur + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """カテゴリ列を Categorical に変換、NaN を数値特徴量については 0 に補完。"""
    out = df.copy()
    for c in FEATURE_COLS_CATEGORICAL:
        out[c] = out[c].astype("category")
    for c in FEATURE_COLS_BOOL:
        out[c] = out[c].fillna(False).astype("int8")
    return out


def get_payouts(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> pd.DataFrame:
    """単勝・複勝の払戻を取得 (combination = 馬番)。"""
    con = duckdb.connect()
    glob = str(parquet_root / "payouts/**/*.parquet")
    df = con.execute(
        f"""
        SELECT race_id,
               bet_type,
               CAST(combination AS INTEGER) AS horse_no,
               payout_yen
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE bet_type IN ('単勝', '複勝')
          AND combination NOT LIKE '%-%'
        """
    ).fetchdf()
    return df


@dataclass
class BacktestResult:
    monthly: pd.DataFrame
    overall: dict
    bet_log: pd.DataFrame


def run_backtest(
    feature_df: pd.DataFrame,
    payouts: pd.DataFrame,
    train_start: str = "2014-04-01",
    test_start: str = "2017-04-01",
    test_end: str = "2026-04-30",
    include_popularity: bool = True,
    use_lightgbm: bool = True,
) -> BacktestResult:
    """月次再学習 Walk-Forward を実行。"""
    import lightgbm as lgb

    feature_cols = FEATURE_COLS_NUMERIC + FEATURE_COLS_BOOL + FEATURE_COLS_CATEGORICAL
    if not include_popularity:
        feature_cols = [c for c in feature_cols if c != "popularity"]

    feature_df = prepare_features(feature_df)
    feature_df["race_date"] = pd.to_datetime(feature_df["race_date"])
    # 完走レコードのみ
    feature_df = feature_df[feature_df["finish_pos"].notna()].copy()

    # 払戻を馬番でマージ (単勝のみ・1着馬の払戻)
    win_pay = payouts[payouts["bet_type"] == "単勝"][["race_id", "horse_no", "payout_yen"]]
    win_pay = win_pay.rename(columns={"payout_yen": "win_payout_if_won"})

    months = month_range(test_start, test_end)
    bet_log = []
    monthly_records = []

    for m in months:
        train_cutoff = pd.Timestamp(m)
        test_cutoff = (train_cutoff + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()

        train = feature_df[
            (feature_df["race_date"] >= pd.Timestamp(train_start))
            & (feature_df["race_date"] < train_cutoff)
        ]
        test = feature_df[
            (feature_df["race_date"] >= train_cutoff)
            & (feature_df["race_date"] < test_cutoff)
        ]
        if test.empty:
            continue
        if len(train) < 1000:
            continue

        # LightGBM 訓練
        X_train = train[feature_cols]
        y_train = train["target_win"]
        X_test = test[feature_cols]
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=63,
            min_child_samples=50,
            verbose=-1,
            n_jobs=-1,
        )
        cat_features = FEATURE_COLS_CATEGORICAL
        model.fit(
            X_train, y_train,
            categorical_feature=cat_features,
        )
        # 予測確率
        test = test.copy()
        test["pred_p_win"] = model.predict_proba(X_test)[:, 1]

        # 各レースの予測Top馬 = 単勝ベット先
        idx_model = test.groupby("race_id")["pred_p_win"].idxmax()
        bets_model = test.loc[idx_model, ["race_id", "horse_no", "finish_pos"]].copy()
        bets_model["strategy"] = "model"

        # 1番人気
        pop_test = test[test["popularity"] == 1]
        bets_pop = pop_test.groupby("race_id").first().reset_index()[
            ["race_id", "horse_no", "finish_pos"]
        ].copy()
        bets_pop["strategy"] = "popularity"

        # 最小馬番
        idx_min = test.groupby("race_id")["horse_no"].idxmin()
        bets_min = test.loc[idx_min, ["race_id", "horse_no", "finish_pos"]].copy()
        bets_min["strategy"] = "lowest_no"

        # ランダム: 各レースで1頭を再現可能シードで選ぶ
        rng = np.random.default_rng(42)
        rand_indices = (
            test.assign(_rand=rng.random(len(test)))
            .groupby("race_id")["_rand"]
            .idxmax()
        )
        bets_rand = test.loc[rand_indices, ["race_id", "horse_no", "finish_pos"]].copy()
        bets_rand["strategy"] = "random"

        bets = pd.concat([bets_model, bets_pop, bets_min, bets_rand], ignore_index=True)
        bets = bets.merge(win_pay, on=["race_id", "horse_no"], how="left")
        bets["won"] = (bets["finish_pos"] == 1).astype(int)
        bets["payout"] = np.where(bets["won"] == 1, bets["win_payout_if_won"].fillna(0), 0)
        bets["month"] = m
        bet_log.append(bets)

        for strat in ["model", "popularity", "lowest_no", "random"]:
            s = bets[bets["strategy"] == strat]
            if s.empty:
                continue
            bets_n = len(s)
            wins = int(s["won"].sum())
            staked = bets_n * 100
            paid = float(s["payout"].sum())
            roi = paid / staked if staked > 0 else 0
            monthly_records.append({
                "month": m,
                "strategy": strat,
                "bets": bets_n,
                "wins": wins,
                "win_rate": wins / bets_n if bets_n > 0 else 0,
                "staked": staked,
                "paid": int(paid),
                "roi": round(roi, 3),
            })

    monthly = pd.DataFrame(monthly_records)
    bet_log_df = pd.concat(bet_log, ignore_index=True) if bet_log else pd.DataFrame()

    overall = (
        monthly.groupby("strategy").agg(
            bets=("bets", "sum"),
            wins=("wins", "sum"),
            staked=("staked", "sum"),
            paid=("paid", "sum"),
        )
        .reset_index()
    )
    overall["win_rate"] = overall["wins"] / overall["bets"]
    overall["roi"] = overall["paid"] / overall["staked"]

    return BacktestResult(monthly=monthly, overall=overall, bet_log=bet_log_df)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-start", default="2014-04-01")
    p.add_argument("--test-start", default="2017-04-01")
    p.add_argument("--test-end", default="2026-04-30")
    p.add_argument("--no-popularity", action="store_true", help="popularity特徴量を除外")
    p.add_argument("--output", default="data/backtest/baseline.parquet")
    args = p.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("=== 特徴量構築 ===")
    df = build_feature_matrix()
    print(f"  shape: {df.shape}")

    print("=== 払戻データ取得 ===")
    payouts = get_payouts()
    print(f"  shape: {payouts.shape}")

    print(f"=== Walk-Forward backtest ({args.test_start} 〜 {args.test_end}) ===")
    res = run_backtest(
        df, payouts,
        train_start=args.train_start,
        test_start=args.test_start,
        test_end=args.test_end,
        include_popularity=not args.no_popularity,
    )

    print()
    print("=== 全体集計 (ベット戦略別 ROI) ===")
    print(res.overall.to_string(index=False))

    print()
    print("=== 月次ROI 直近12ヶ月 ===")
    print(res.monthly.sort_values(["month", "strategy"]).tail(48).to_string(index=False))

    # 保存
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.monthly.to_parquet(out, index=False, compression="zstd")
    print(f"\n月次結果を保存: {out}")


if __name__ == "__main__":
    main()
