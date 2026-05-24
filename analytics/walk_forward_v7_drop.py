"""
Walk-Forward v7 (drop variant): popularity を除外したモデルで 79ヶ月 ロバスト検証。

v6 (with popularity) との比較で:
- 穴馬探知能力 (高オッズベット率)
- ROI の長期安定性
- sanrenpuku の bet 数変化 (PL 確率分散の影響)

popularity_mode:
  'current'      : popularity をそのまま使用 (v6 と同じ)
  'drop'         : popularity を 0 で定数化 (実質除外)
  'conditional'  : 若馬/初心馬/Openのみ raw、それ以外は中和

使用例:
    uv run python -m analytics.walk_forward_v7_drop --popularity-mode drop
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.features import (
    FEATURE_COLS_BOOL, FEATURE_COLS_CATEGORICAL, FEATURE_COLS_NUMERIC,
    build_feature_matrix,
)
from analytics.plackett_luce import wide_top_k_from_entry
from analytics.walk_forward import month_range, prepare_features
from analytics.walk_forward_v4 import (
    COMBO_BET_TYPES, COMBO_LABEL,
    compute_combo_ev_for_race, determine_combo_hit,
    get_combo_odds, get_place_payouts, get_real_odds,
)
from analytics.walk_forward_v6 import (
    WATER_FILTERS_BY_STRATEGY, get_race_water, water_bin,
)
from persist.parquet_writer import DEFAULT_PARQUET_ROOT


def adjust_popularity(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = df.copy()
    if mode == "current":
        return df
    if mode == "drop":
        df["popularity"] = 0
        return df
    if mode == "conditional":
        is_open = df["race_class"].isna() | (
            df["race_class"].astype(str).str.contains("オープン|open", case=False, na=False)
        )
        is_young = df["age"] <= 3
        is_short_career = df["lifetime_starts"].fillna(99) <= 10
        use_raw = is_young | is_short_career | is_open
        neutral = df["entry_count"] / 2
        df["popularity"] = np.where(use_raw, df["popularity"], neutral)
        return df
    raise ValueError(f"unknown mode: {mode}")


def run(
    popularity_mode: str = "drop",
    test_start: str = "2019-10-01",
    test_end: str = "2026-04-30",
    train_start: str = "2014-04-01",
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
    bets_log_path: Path | None = None,
    prob_thr_san: float = 0.20,
) -> dict:
    import lightgbm as lgb

    bet_types = COMBO_BET_TYPES
    all_bets_log: list[dict] = []

    print(f"=== Walk-Forward v7 (popularity_mode={popularity_mode}) ===")
    df = build_feature_matrix(parquet_root)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df[df["finish_pos"].notna()].copy()

    real_odds = get_real_odds(parquet_root)
    df = df.merge(real_odds, on=["race_id", "horse_no"], how="left").rename(columns={
        "win_odds": "real_win_odds",
        "place_odds_low": "real_place_low",
        "place_odds_high": "real_place_high",
    })
    df = df.merge(get_place_payouts(parquet_root), on=["race_id", "horse_no"], how="left")

    water = get_race_water(parquet_root)
    water["water_bin"] = water["track_water_pct"].apply(water_bin)
    df = df.merge(water[["race_id", "water_bin"]], on="race_id", how="left")

    combo_odds: dict[str, pd.DataFrame] = {}
    for bt in bet_types:
        co = get_combo_odds(parquet_root, bt)
        combo_odds[bt] = co

    df = prepare_features(df)
    df = adjust_popularity(df, popularity_mode)

    feature_cols = FEATURE_COLS_NUMERIC + FEATURE_COLS_BOOL + FEATURE_COLS_CATEGORICAL

    fpos_df = df[["race_id", "horse_no", "finish_pos"]].dropna(subset=["finish_pos"])
    fpos_df = fpos_df.sort_values(["race_id", "finish_pos"])
    top_by_race: dict[str, list[int]] = (
        fpos_df.groupby("race_id").apply(lambda g: g.head(3)["horse_no"].astype(int).tolist()).to_dict()
    )

    months = month_range(test_start, test_end)
    records: list[dict] = []

    for m_idx, m in enumerate(months):
        train_cutoff = pd.Timestamp(m)
        test_cutoff = (train_cutoff + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()
        train = df[(df["race_date"] >= pd.Timestamp(train_start)) & (df["race_date"] < train_cutoff)]
        test = df[(df["race_date"] >= train_cutoff) & (df["race_date"] < test_cutoff)]
        if test.empty or len(train) < 1000:
            continue

        X_train = train[feature_cols]
        X_test = test[feature_cols].copy()

        model_win = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, verbose=-1, n_jobs=-1,
        )
        model_win.fit(X_train, train["target_win"], categorical_feature=FEATURE_COLS_CATEGORICAL)

        train_place = train.dropna(subset=["target_place"])
        model_place = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, verbose=-1, n_jobs=-1,
        )
        model_place.fit(
            train_place[feature_cols], train_place["target_place"].astype(int),
            categorical_feature=FEATURE_COLS_CATEGORICAL,
        )

        test = test.copy()
        test["pred_p_win"] = model_win.predict_proba(X_test)[:, 1]
        test["pred_p_place"] = model_place.predict_proba(X_test)[:, 1]
        test["ev_win"] = test["pred_p_win"] * test["real_win_odds"]
        test["ev_place"] = test["pred_p_place"] * test["real_place_low"]
        test["place_eligible"] = test["entry_count"] >= 5

        # === place_top1 (water filt) ===
        place_eligible_test = test[test["place_eligible"]].copy()
        if not place_eligible_test.empty:
            idx_pl = place_eligible_test.groupby("race_id")["pred_p_place"].idxmax()
            bp = place_eligible_test.loc[idx_pl].copy()
            bp = bp[bp["ev_place"] > 1.0].dropna(subset=["real_place_low"])
            bp_filt = bp[bp["water_bin"].isin(WATER_FILTERS_BY_STRATEGY["place_top1"])]

            for variant_label, b in [("place_top1_raw", bp), ("place_top1_water_filt", bp_filt)]:
                if b.empty:
                    continue
                paid = b["place_payout"].fillna(0).astype(float).sum()
                hits = b["place_payout"].notna().sum()
                # 高オッズ件数
                high_odds = (b["real_place_low"] >= 2.0).sum()
                records.append({
                    "month": m, "strategy": variant_label,
                    "bets": len(b), "wins": int(hits),
                    "high_odds_bets": int(high_odds),
                    "win_rate": hits / len(b),
                    "staked": len(b) * 100, "paid": int(paid),
                    "roi": round(paid / (len(b) * 100), 3),
                })
                if bets_log_path is not None:
                    for _, r in b.iterrows():
                        hit = bool(pd.notna(r["place_payout"]))
                        all_bets_log.append({
                            "strategy": variant_label, "month": str(m),
                            "race_id": r["race_id"], "bet_type": "place",
                            "combination": str(int(r["horse_no"])),
                            "prob": float(r["pred_p_place"]),
                            "ev": float(r["ev_place"]),
                            "odds": float(r["real_place_low"]),
                            "hit": hit,
                            "payout_per_100yen": float(r["place_payout"]) if hit else 0.0,
                            "water_bin": r["water_bin"],
                        })

        # === sanrenpuku (water filt) ===
        test_race_ids = set(test["race_id"].unique())
        month_combo_odds = {bt: co[co["race_id"].isin(test_race_ids)] for bt, co in combo_odds.items()}
        combo_ev_rows = []
        for race_id, race_df in test.groupby("race_id"):
            ro = {bt: month_combo_odds[bt][month_combo_odds[bt]["race_id"] == race_id] for bt in bet_types}
            ev_df = compute_combo_ev_for_race(race_df, ro)
            if not ev_df.empty:
                combo_ev_rows.append(ev_df)
        if not combo_ev_rows:
            continue
        combo_ev = pd.concat(combo_ev_rows, ignore_index=True)
        rid_water = test.groupby("race_id")["water_bin"].first().to_dict()
        ec_lookup = test.groupby("race_id")["entry_count"].first().to_dict()
        combo_ev["water_bin"] = combo_ev["race_id"].map(rid_water)

        sub = combo_ev[(combo_ev["bet_type"] == "sanrenpuku") & (combo_ev["prob"] >= prob_thr_san)]
        if sub.empty:
            continue
        idx_t = sub.groupby("race_id")["ev"].idxmax()
        bs = sub.loc[idx_t].copy()
        bs = bs[bs["ev"] > 1.0]

        for variant_label, b in [("sanrenpuku_p20_raw", bs),
                                  ("sanrenpuku_p20_water_filt",
                                   bs[bs["water_bin"].isin(WATER_FILTERS_BY_STRATEGY["sanrenpuku_top1"])])]:
            if b.empty:
                continue
            paid = 0.0
            hits = 0
            high_odds = 0
            for _, r in b.iterrows():
                rid = r["race_id"]
                if rid not in top_by_race:
                    continue
                finishers = top_by_race[rid]
                wide_k = wide_top_k_from_entry(int(ec_lookup.get(rid, 10)))
                hit = determine_combo_hit(r["combination"], finishers, "sanrenpuku", wide_top_k=wide_k)
                if hit:
                    paid += r["odds_min"] * 100
                    hits += 1
                if r["odds_min"] >= 10.0:
                    high_odds += 1
                if bets_log_path is not None:
                    all_bets_log.append({
                        "strategy": variant_label, "month": str(m),
                        "race_id": rid, "bet_type": "sanrenpuku",
                        "combination": r["combination"],
                        "prob": float(r["prob"]),
                        "ev": float(r["ev"]),
                        "odds": float(r["odds_min"]),
                        "hit": bool(hit),
                        "payout_per_100yen": float(r["odds_min"] * 100) if hit else 0.0,
                        "water_bin": r["water_bin"],
                    })
            records.append({
                "month": m, "strategy": variant_label,
                "bets": len(b), "wins": int(hits),
                "high_odds_bets": int(high_odds),
                "win_rate": hits / len(b),
                "staked": len(b) * 100, "paid": int(paid),
                "roi": round(paid / (len(b) * 100), 3),
            })

        if (m_idx + 1) % 12 == 0:
            print(f"  進捗: {m_idx+1}/{len(months)} ヶ月")

    if bets_log_path is not None and all_bets_log:
        bets_log_path = Path(bets_log_path)
        bets_log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_bets_log).to_parquet(bets_log_path, index=False, compression="zstd")
        print(f"\nベットログ保存: {bets_log_path} ({len(all_bets_log):,} bets)")

    monthly = pd.DataFrame(records)
    overall = (
        monthly.groupby("strategy")
        .agg(bets=("bets", "sum"), wins=("wins", "sum"),
             high_odds_bets=("high_odds_bets", "sum"),
             staked=("staked", "sum"), paid=("paid", "sum"))
        .reset_index()
    )
    overall["win_rate"] = (overall["wins"] / overall["bets"]).round(3)
    overall["roi"] = (overall["paid"] / overall["staked"]).round(3)
    overall["high_odds_rate"] = (overall["high_odds_bets"] / overall["bets"]).round(3)
    return {"monthly": monthly, "overall": overall, "popularity_mode": popularity_mode}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--popularity-mode", default="drop", choices=["current", "drop", "conditional"])
    p.add_argument("--test-start", default="2019-10-01")
    p.add_argument("--test-end", default="2026-04-30")
    p.add_argument("--output", default=None)
    p.add_argument("--bets-log-path", default=None)
    p.add_argument("--prob-thr-san", type=float, default=0.20,
                   help="sanrenpuku prob threshold (drop variant では 0.10 推奨)")
    args = p.parse_args()

    out = args.output or f"data/backtest/v7_pop_{args.popularity_mode}.parquet"
    log = args.bets_log_path or f"data/backtest/v7_pop_{args.popularity_mode}_bets_log.parquet"

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    res = run(
        popularity_mode=args.popularity_mode,
        test_start=args.test_start,
        test_end=args.test_end,
        bets_log_path=Path(log),
        prob_thr_san=args.prob_thr_san,
    )

    print()
    print(f"=== Walk-Forward v7 (popularity_mode={args.popularity_mode}) 結果 ===")
    print(res["overall"].sort_values("roi", ascending=False).to_string(index=False))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res["monthly"].to_parquet(out_path, index=False, compression="zstd")
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
