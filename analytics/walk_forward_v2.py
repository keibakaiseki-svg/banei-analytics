"""
Walk-Forward backtest v2: 実オッズ (netkeiba) + 水分量適性特徴量を使った改良版。

設計:
- 評価期間 2024-01-01 〜 2026-04-30 (netkeiba 実オッズ 100% カバー)
- 訓練窓: Expanding (2014-04 開始)
- 水分量適性特徴量: 2023-12-31 までのデータから一括計算 → 固定使用 (リーケージなし)
- EV計算: 実 win_odds を使用 (assumed_odds 不使用)
- 戦略: model_top1 / model_top1_ev_gate / popularity 等を比較
"""

from __future__ import annotations

import argparse
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
from analytics.walk_forward import month_range, prepare_features
from persist.parquet_writer import DEFAULT_PARQUET_ROOT


def build_water_adaptability_features(
    cutoff_date: str = "2024-01-01",
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
) -> pd.DataFrame:
    """各馬の水分量帯別スピード指数 (cutoff より前のデータで計算)。リーケージ防止。"""
    con = duckdb.connect()
    for t in ("races", "entries"):
        glob = str(parquet_root / t / "**/*.parquet")
        con.execute(
            f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
        )
    return con.execute(
        f"""
        WITH races_w AS (
            SELECT *, CASE
                WHEN track_water_pct < 2.0 THEN 'dry'
                WHEN track_water_pct < 3.0 THEN 'normal'
                ELSE 'wet'
            END AS water_band
            FROM races
            WHERE race_date < '{cutoff_date}'
        ),
        baselines AS (
            SELECT r.race_class, r.water_band, MEDIAN(e.finish_time_sec) AS expected_time
            FROM races_w r JOIN entries e USING (race_id)
            WHERE e.finish_pos IS NOT NULL AND e.finish_time_sec IS NOT NULL
              AND r.race_class IS NOT NULL
            GROUP BY r.race_class, r.water_band
        ),
        variants AS (
            SELECT r.race_date, MEDIAN(e.finish_time_sec - b.expected_time) AS variant_sec
            FROM races_w r JOIN entries e USING (race_id)
            JOIN baselines b ON b.race_class = r.race_class AND b.water_band = r.water_band
            WHERE e.finish_pos IS NOT NULL AND e.finish_time_sec IS NOT NULL
            GROUP BY r.race_date
        ),
        entry_sf AS (
            SELECT e.horse_id, r.water_band,
                   (b.expected_time - (e.finish_time_sec - v.variant_sec)) AS sf
            FROM races_w r JOIN entries e USING (race_id)
            LEFT JOIN baselines b ON b.race_class = r.race_class AND b.water_band = r.water_band
            LEFT JOIN variants v ON v.race_date = r.race_date
            WHERE e.finish_pos IS NOT NULL AND e.finish_time_sec IS NOT NULL
              AND e.horse_id IS NOT NULL
        )
        SELECT
            horse_id,
            COUNT(*) AS adapt_total_runs,
            ROUND(AVG(sf), 2) AS adapt_overall_sf,
            SUM(CASE WHEN water_band = 'dry' THEN 1 ELSE 0 END) AS adapt_n_dry,
            ROUND(AVG(CASE WHEN water_band = 'dry' THEN sf END), 2) AS adapt_sf_dry,
            SUM(CASE WHEN water_band = 'normal' THEN 1 ELSE 0 END) AS adapt_n_normal,
            ROUND(AVG(CASE WHEN water_band = 'normal' THEN sf END), 2) AS adapt_sf_normal,
            SUM(CASE WHEN water_band = 'wet' THEN 1 ELSE 0 END) AS adapt_n_wet,
            ROUND(AVG(CASE WHEN water_band = 'wet' THEN sf END), 2) AS adapt_sf_wet
        FROM entry_sf
        WHERE sf IS NOT NULL
        GROUP BY horse_id
        """
    ).fetchdf()


def get_real_odds(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> pd.DataFrame:
    """netkeiba 実 win_odds を取得 (PK: race_id, horse_no)。"""
    df = pd.read_parquet(parquet_root / "odds_netkeiba.parquet")
    return df[["race_id_local", "horse_no", "win_odds", "place_odds_low", "place_odds_high"]].rename(
        columns={"race_id_local": "race_id"}
    )


def run_v2(
    test_start: str = "2024-01-01",
    test_end: str = "2026-04-30",
    train_start: str = "2014-04-01",
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
) -> dict:
    import lightgbm as lgb

    print("=== 特徴量行列構築 (per-race expanding adaptability 込み) ===")
    df = build_feature_matrix(parquet_root)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df[df["finish_pos"].notna()].copy()

    print("=== 実 win_odds (netkeiba) ===")
    real_odds = get_real_odds(parquet_root)
    real_odds = real_odds.rename(columns={"win_odds": "real_win_odds"})
    df = df.merge(real_odds[["race_id", "horse_no", "real_win_odds"]],
                  on=["race_id", "horse_no"], how="left")
    print(f"  real_win_odds coverage in test period: "
          f"{df[df.race_date >= test_start]['real_win_odds'].notna().mean()*100:.1f}%")

    df = prepare_features(df)

    # features.py に既に adapt_* が含まれているので追加 cols 不要
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

        # 訓練
        X_train = train[feature_cols]
        y_train = train["target_win"]
        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, verbose=-1, n_jobs=-1,
        )
        model.fit(X_train, y_train, categorical_feature=FEATURE_COLS_CATEGORICAL)
        test = test.copy()
        test["pred_p_win"] = model.predict_proba(test[feature_cols])[:, 1]

        # 戦略1: model_top1 (各レース予測最大馬・実オッズで決済)
        idx_top = test.groupby("race_id")["pred_p_win"].idxmax()
        bets_top = test.loc[idx_top].copy()
        bets_top["strategy"] = "model_top1"

        # 戦略2: model_top1_ev_gate (実オッズで EV 計算・>1.0 のみ)
        test["ev_real"] = test["pred_p_win"] * test["real_win_odds"]
        ev_top = test.loc[idx_top].copy()
        ev_top = ev_top[ev_top["ev_real"] > 1.0].copy()
        ev_top["strategy"] = "model_ev_real_gt_1.0"

        # 戦略3: 1番人気 (基準ベンチマーク)
        pop_test = test[test["popularity"] == 1]
        bets_pop = pop_test.groupby("race_id").first().reset_index()
        bets_pop["strategy"] = "popularity"

        # 戦略4: model_top1_ev_gate しきい値別 (1.1, 1.2)
        ev_top_11 = test.loc[idx_top].copy()
        ev_top_11 = ev_top_11[ev_top_11["ev_real"] > 1.1].copy()
        ev_top_11["strategy"] = "model_ev_real_gt_1.1"

        ev_top_12 = test.loc[idx_top].copy()
        ev_top_12 = ev_top_12[ev_top_12["ev_real"] > 1.2].copy()
        ev_top_12["strategy"] = "model_ev_real_gt_1.2"

        for bets, strat in [
            (bets_top, "model_top1"),
            (ev_top, "model_ev_real_gt_1.0"),
            (ev_top_11, "model_ev_real_gt_1.1"),
            (ev_top_12, "model_ev_real_gt_1.2"),
            (bets_pop, "popularity"),
        ]:
            if bets.empty:
                continue
            bets = bets.dropna(subset=["real_win_odds"])  # 実オッズ無いものは除外
            if bets.empty:
                continue
            wins = (bets["finish_pos"] == 1).sum()
            paid = (bets[bets["finish_pos"] == 1]["real_win_odds"] * 100).sum()
            staked = len(bets) * 100
            monthly_records.append({
                "month": m,
                "strategy": strat,
                "bets": int(len(bets)),
                "wins": int(wins),
                "win_rate": float(wins / len(bets)),
                "staked": int(staked),
                "paid": int(paid),
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
    p.add_argument("--output", default="data/backtest/v2_real_odds.parquet")
    args = p.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    res = run_v2(test_start=args.test_start, test_end=args.test_end)

    print()
    print("=== Walk-Forward v2 結果 ===")
    print(res["overall"].sort_values("roi", ascending=False).to_string(index=False))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res["monthly"].to_parquet(out, index=False, compression="zstd")
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
