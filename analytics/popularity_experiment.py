"""
popularity 重視度の調整実験。

仮説: モデルの popularity SHAP 0.827 は他特徴量の 16倍 で偏重。
      これが「市場に追従して穴馬を取れない」原因の可能性。

3 variants:
  A. current: popularity をそのまま使用 (raw)
  B. drop: popularity を完全除外
  C. conditional: 若馬 (age<=3) or キャリア浅い馬 (lifetime_starts<=10)
                  または Open class のみで popularity を使用、それ以外は中和値

各 variant で同じ train/test 分割を使い、place_top1 と sanrenpuku_p20 の
ROI を比較する。

train: 2014-2014-04〜2024-12  /  test: 2025-01-01〜2026-04-30
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
from analytics.plackett_luce import combo_probs_for_race, wide_top_k_from_entry
from analytics.walk_forward import prepare_features
from analytics.walk_forward_v4 import (
    determine_combo_hit, get_combo_odds, get_place_payouts, get_real_odds,
    compute_combo_ev_for_race,
)
from analytics.walk_forward_v6 import water_bin, WATER_FILTERS_BY_STRATEGY
from persist.parquet_writer import DEFAULT_PARQUET_ROOT


def adjust_popularity(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """popularity 列を mode に応じて変換。

    mode:
      'current': そのまま (no-op)
      'drop': popularity 列を削除 (FEATURE_COLS から外す)
      'conditional': 若馬 (age<=3) or キャリア浅 (lifetime_starts<=10) or
                     Open class (race_class == 'open' か NaN) のみ raw、
                     その他は中和値 (entry_count / 2)
    """
    df = df.copy()
    if mode == "current":
        return df
    if mode == "drop":
        df["popularity"] = 0  # 効果無効化
        return df
    if mode == "conditional":
        # Open class フラグ (race_class が None / 'open' / NaN は重賞 = Open とみなす)
        is_open = df["race_class"].isna() | (df["race_class"].astype(str).str.contains("オープン|open", case=False, na=False))
        is_young = df["age"] <= 3
        is_short_career = df["lifetime_starts"].fillna(99) <= 10
        use_raw = is_young | is_short_career | is_open
        neutral = df["entry_count"] / 2  # 中央値
        df["popularity"] = np.where(use_raw, df["popularity"], neutral)
        return df
    raise ValueError(f"unknown mode: {mode}")


def run_experiment(
    mode: str,
    train_end: str = "2024-12-31",
    test_start: str = "2025-01-01",
    test_end: str = "2026-04-30",
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
) -> dict:
    import lightgbm as lgb

    print(f"\n=== Variant: {mode} ===")
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

    # water bin
    df["water_bin"] = df["track_water_pct"].apply(water_bin)
    # popularity 計算 (win_odds 順位)
    df["popularity"] = df.groupby("race_id")["real_win_odds"].rank(method="min", ascending=True)
    df = prepare_features(df)
    # popularity 調整
    df = adjust_popularity(df, mode)

    feature_cols = FEATURE_COLS_NUMERIC + FEATURE_COLS_BOOL + FEATURE_COLS_CATEGORICAL
    train = df[(df["race_date"] >= "2014-04-01") & (df["race_date"] <= train_end)]
    test = df[(df["race_date"] >= test_start) & (df["race_date"] <= test_end)]
    print(f"  train: {len(train):,}, test: {len(test):,}")

    # 訓練
    model_win = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, verbose=-1, n_jobs=-1,
    )
    model_win.fit(train[feature_cols], train["target_win"],
                  categorical_feature=FEATURE_COLS_CATEGORICAL)

    train_p = train.dropna(subset=["target_place"])
    model_place = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, verbose=-1, n_jobs=-1,
    )
    model_place.fit(train_p[feature_cols], train_p["target_place"].astype(int),
                    categorical_feature=FEATURE_COLS_CATEGORICAL)

    test = test.copy()
    test["pred_p_win"] = model_win.predict_proba(test[feature_cols])[:, 1]
    test["pred_p_place"] = model_place.predict_proba(test[feature_cols])[:, 1]
    test["ev_win"] = test["pred_p_win"] * test["real_win_odds"]
    test["ev_place"] = test["pred_p_place"] * test["real_place_low"]
    test["place_eligible"] = test["entry_count"] >= 5

    # === place_top1_water_filt ===
    test_place = test[test["place_eligible"]].copy()
    idx_pl = test_place.groupby("race_id")["pred_p_place"].idxmax()
    bp = test_place.loc[idx_pl].copy()
    bp = bp[bp["ev_place"] > 1.0].dropna(subset=["real_place_low"])
    bp = bp[bp["water_bin"].isin(WATER_FILTERS_BY_STRATEGY["place_top1"])]

    place_result = {
        "bets": len(bp),
        "hits": int(bp["place_payout"].notna().sum()),
        "staked": len(bp) * 100,
        "paid": int(bp["place_payout"].fillna(0).astype(float).sum()),
    }
    place_result["roi"] = place_result["paid"] / place_result["staked"] if place_result["staked"] else 0
    place_result["hit_rate"] = place_result["hits"] / place_result["bets"] if place_result["bets"] else 0

    # === sanrenpuku_p20_water_filt ===
    combo_odds = get_combo_odds(parquet_root, "sanrenpuku")
    combo_odds = combo_odds[combo_odds["race_id"].isin(test["race_id"])]
    combo_ev_rows = []
    for race_id, race_df in test.groupby("race_id"):
        ro = {"sanrenpuku": combo_odds[combo_odds["race_id"] == race_id]}
        # 他券種は不要なので簡略化
        ev_df = compute_combo_ev_for_race(race_df, ro)
        if not ev_df.empty:
            combo_ev_rows.append(ev_df)
    combo_ev = pd.concat(combo_ev_rows, ignore_index=True) if combo_ev_rows else pd.DataFrame()

    san_result = {"bets": 0, "hits": 0, "staked": 0, "paid": 0, "roi": 0, "hit_rate": 0}
    if not combo_ev.empty:
        sub = combo_ev[(combo_ev["bet_type"] == "sanrenpuku") & (combo_ev["prob"] >= 0.20)]
        idx_t = sub.groupby("race_id")["ev"].idxmax()
        bs = sub.loc[idx_t].copy()
        bs = bs[bs["ev"] > 1.0]
        # water bin 付加
        rid_water = test.groupby("race_id")["water_bin"].first().to_dict()
        ec_lookup = test.groupby("race_id")["entry_count"].first().to_dict()
        top_by_race = (test[test["finish_pos"].notna()]
                       .sort_values(["race_id", "finish_pos"])
                       .groupby("race_id")
                       .apply(lambda g: g.head(3)["horse_no"].astype(int).tolist())
                       .to_dict())
        bs["water_bin"] = bs["race_id"].map(rid_water)
        bs = bs[bs["water_bin"].isin(WATER_FILTERS_BY_STRATEGY["sanrenpuku_top1"])]

        paid = 0
        hits = 0
        for _, r in bs.iterrows():
            rid = r["race_id"]
            if rid not in top_by_race:
                continue
            finishers = top_by_race[rid]
            entry = int(ec_lookup.get(rid, 10))
            wide_k = wide_top_k_from_entry(entry)
            if determine_combo_hit(r["combination"], finishers, "sanrenpuku", wide_top_k=wide_k):
                paid += r["odds_min"] * 100
                hits += 1
        san_result = {
            "bets": len(bs), "hits": hits,
            "staked": len(bs) * 100, "paid": int(paid),
        }
        san_result["roi"] = san_result["paid"] / san_result["staked"] if san_result["staked"] else 0
        san_result["hit_rate"] = san_result["hits"] / san_result["bets"] if san_result["bets"] else 0

    # 高オッズ的中率 (穴馬抽出度のproxy)
    high_odds_hits_p = 0
    if place_result["bets"]:
        high_odds_hits_p = (bp[bp["real_place_low"] >= 2.0]).shape[0]
    high_odds_hits_s = 0
    if san_result["bets"] and "bs" in dir():
        high_odds_hits_s = bs[bs["odds_min"] >= 10.0].shape[0]

    print(f"  place_top1_water_filt: bets={place_result['bets']}, hit={place_result['hit_rate']:.3f}, ROI={place_result['roi']:.3f}")
    print(f"  sanrenpuku_p20_water_filt: bets={san_result['bets']}, hit={san_result['hit_rate']:.3f}, ROI={san_result['roi']:.3f}")
    print(f"  穴狙い度 (place odds≥2.0): {high_odds_hits_p}/{place_result['bets']}")

    return {
        "mode": mode,
        "place": place_result,
        "sanrenpuku": san_result,
        "high_odds_place_bets": high_odds_hits_p,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--test-end", default="2026-04-30")
    args = p.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    results = []
    for mode in ["current", "drop", "conditional"]:
        res = run_experiment(
            mode=mode,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
        )
        results.append(res)

    # まとめ
    print("\n\n=== サマリ ===")
    for r in results:
        p = r["place"]
        s = r["sanrenpuku"]
        print(f"\n{r['mode']:>12}: "
              f"place bets={p['bets']}/ROI={p['roi']:.3f} | "
              f"sanrenpuku bets={s['bets']}/ROI={s['roi']:.3f}")
        print(f"               穴狙い (odds≥2.0): {r['high_odds_place_bets']}件")


if __name__ == "__main__":
    main()
