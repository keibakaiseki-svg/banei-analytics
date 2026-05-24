"""
騎手特徴量の追加実験。

現状の特徴量は `has_allowance` (減量) と `jockey_changed` (前走と違う) のみ。
騎手自身の能力差や主戦騎手 / 主戦コンビの効果は捉えていない。

追加する特徴量 (リーケージなし: 当該レース時点までの累積):
  1. jk_career_win_rate     騎手の累積勝率
  2. jk_career_top3_rate    騎手の累積複勝率
  3. jk_recent100_win_rate  直近100戦勝率
  4. jk_horse_pair_rides    この馬×この騎手の累積騎乗回数 (主戦騎手)
  5. jk_horse_pair_win_rate その組合せの勝率
  6. jk_trainer_pair_winrate 騎手×厩舎の勝率 (主戦コンビ)
  7. is_main_jockey_for_horse 直近5走中3回以上騎乗 = 主戦騎手 フラグ

実験設計: place_top1 と sanrenpuku_p20 で
  - baseline (現状 features)
  - +jockey features (追加版)
を比較

train: 2014-04-01 〜 2024-12-31, test: 2025-01-01 〜 2026-04-30 (16ヶ月)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from analytics.features import (
    FEATURE_COLS_BOOL, FEATURE_COLS_CATEGORICAL, FEATURE_COLS_NUMERIC,
    build_feature_matrix,
)
from analytics.plackett_luce import wide_top_k_from_entry
from analytics.walk_forward import prepare_features
from analytics.walk_forward_v4 import (
    compute_combo_ev_for_race, determine_combo_hit,
    get_combo_odds, get_place_payouts, get_real_odds,
)
from analytics.walk_forward_v6 import WATER_FILTERS_BY_STRATEGY, water_bin
from persist.parquet_writer import DEFAULT_PARQUET_ROOT


def compute_jockey_features(parquet_root: Path) -> pd.DataFrame:
    """リーケージなしの騎手特徴量を SQL window で計算。"""
    con = duckdb.connect()
    glob = str(parquet_root / "entries/**/*.parquet")
    races_glob = str(parquet_root / "races/**/*.parquet")
    print("  騎手累積特徴量を計算中 (SQL window)...")

    # 各 entry に race_date を付加して時系列で累積
    sql = f"""
    WITH ent AS (
        SELECT e.race_id, e.horse_id, e.jockey_id, e.trainer_id, e.finish_pos,
               e.horse_no, r.race_date
        FROM read_parquet('{glob}', hive_partitioning=true) e
        JOIN read_parquet('{races_glob}', hive_partitioning=true) r ON e.race_id = r.race_id
        WHERE e.jockey_id IS NOT NULL
    ),
    -- 騎手の累積勝率 (この race を含まない)
    jk_career AS (
        SELECT race_id, horse_id, jockey_id, trainer_id,
               COUNT(*) OVER (PARTITION BY jockey_id ORDER BY race_date, race_id
                              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS jk_career_rides,
               SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END) OVER (PARTITION BY jockey_id ORDER BY race_date, race_id
                              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS jk_career_wins,
               SUM(CASE WHEN finish_pos <= 3 THEN 1 ELSE 0 END) OVER (PARTITION BY jockey_id ORDER BY race_date, race_id
                              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS jk_career_top3,
               -- 直近100戦
               COUNT(*) OVER (PARTITION BY jockey_id ORDER BY race_date, race_id
                              ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING) AS jk_recent_rides,
               SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END) OVER (PARTITION BY jockey_id ORDER BY race_date, race_id
                              ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING) AS jk_recent_wins
        FROM ent
    ),
    -- 馬×騎手コンビの累積 (主戦騎手検出)
    jk_horse_pair AS (
        SELECT race_id, horse_id, jockey_id,
               COUNT(*) OVER (PARTITION BY horse_id, jockey_id ORDER BY race_date, race_id
                              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS pair_rides,
               SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END) OVER (PARTITION BY horse_id, jockey_id ORDER BY race_date, race_id
                              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS pair_wins
        FROM ent
    ),
    -- 騎手×厩舎コンビの累積 (主戦コンビ)
    jt_pair AS (
        SELECT race_id, horse_id, jockey_id, trainer_id,
               COUNT(*) OVER (PARTITION BY jockey_id, trainer_id ORDER BY race_date, race_id
                              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS jt_rides,
               SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END) OVER (PARTITION BY jockey_id, trainer_id ORDER BY race_date, race_id
                              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS jt_wins
        FROM ent
    )
    SELECT
        c.race_id, c.horse_id, c.jockey_id,
        CASE WHEN c.jk_career_rides > 30 THEN CAST(c.jk_career_wins AS DOUBLE) / c.jk_career_rides ELSE 0.13 END AS jk_career_win_rate,
        CASE WHEN c.jk_career_rides > 30 THEN CAST(c.jk_career_top3 AS DOUBLE) / c.jk_career_rides ELSE 0.35 END AS jk_career_top3_rate,
        CASE WHEN c.jk_recent_rides > 30 THEN CAST(c.jk_recent_wins AS DOUBLE) / c.jk_recent_rides ELSE 0.13 END AS jk_recent_win_rate,
        h.pair_rides AS jk_horse_pair_rides,
        CASE WHEN h.pair_rides > 5 THEN CAST(h.pair_wins AS DOUBLE) / h.pair_rides ELSE 0.13 END AS jk_horse_pair_win_rate,
        CASE WHEN j.jt_rides > 30 THEN CAST(j.jt_wins AS DOUBLE) / j.jt_rides ELSE 0.13 END AS jk_trainer_pair_win_rate,
        CASE WHEN h.pair_rides >= 3 THEN 1 ELSE 0 END AS is_main_jockey_for_horse
    FROM jk_career c
    LEFT JOIN jk_horse_pair h USING (race_id, horse_id, jockey_id)
    LEFT JOIN jt_pair j USING (race_id, horse_id, jockey_id)
    """
    df = con.execute(sql).fetchdf()
    print(f"  騎手特徴量: {len(df):,} 行")
    return df


def run_experiment(
    variant: str,
    train_end: str = "2024-12-31",
    test_start: str = "2025-01-01",
    test_end: str = "2026-04-30",
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
) -> dict:
    import lightgbm as lgb

    print(f"\n=== Variant: {variant} ===")
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
    df["water_bin"] = df["track_water_pct"].apply(water_bin)
    df["popularity"] = df.groupby("race_id")["real_win_odds"].rank(method="min", ascending=True)
    df = prepare_features(df)

    # 騎手特徴量 (variant=jockey の場合のみ追加)
    extra_numeric: list[str] = []
    if variant == "jockey":
        jk = compute_jockey_features(parquet_root)
        df = df.merge(jk, on=["race_id", "horse_id"], how="left")
        # jockey_id_x が重複している可能性あり、cleanup
        if "jockey_id_y" in df.columns:
            df = df.drop(columns=["jockey_id_y"])
            df = df.rename(columns={"jockey_id_x": "jockey_id"})
        extra_numeric = [
            "jk_career_win_rate", "jk_career_top3_rate", "jk_recent_win_rate",
            "jk_horse_pair_rides", "jk_horse_pair_win_rate", "jk_trainer_pair_win_rate",
            "is_main_jockey_for_horse",
        ]
        for c in extra_numeric:
            df[c] = df[c].fillna(0.13 if "rate" in c else 0)

    feature_cols = FEATURE_COLS_NUMERIC + extra_numeric + FEATURE_COLS_BOOL + FEATURE_COLS_CATEGORICAL
    print(f"  Features: {len(feature_cols)} (含む jockey: {len(extra_numeric)})")

    train = df[(df["race_date"] >= "2014-04-01") & (df["race_date"] <= train_end)]
    test = df[(df["race_date"] >= test_start) & (df["race_date"] <= test_end)]
    print(f"  train: {len(train):,}, test: {len(test):,}")

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

    # SHAP 特徴量重要度 (上位10)
    imp = pd.DataFrame({
        "feature": feature_cols,
        "gain": model_place.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    print(f"\n  Top 10 feature gain (place model):")
    print(imp.head(10).to_string(index=False))
    if variant == "jockey":
        jk_imp = imp[imp["feature"].str.startswith(("jk_", "is_main"))]
        print(f"\n  騎手特徴量の順位:")
        print(jk_imp.to_string(index=False))

    test = test.copy()
    test["pred_p_win"] = model_win.predict_proba(test[feature_cols])[:, 1]
    test["pred_p_place"] = model_place.predict_proba(test[feature_cols])[:, 1]
    test["ev_win"] = test["pred_p_win"] * test["real_win_odds"]
    test["ev_place"] = test["pred_p_place"] * test["real_place_low"]
    test["place_eligible"] = test["entry_count"] >= 5

    # place_top1_water_filt
    pe = test[test["place_eligible"]].copy()
    idx_pl = pe.groupby("race_id")["pred_p_place"].idxmax()
    bp = pe.loc[idx_pl].copy()
    bp = bp[bp["ev_place"] > 1.0].dropna(subset=["real_place_low"])
    bp = bp[bp["water_bin"].isin(WATER_FILTERS_BY_STRATEGY["place_top1"])]
    paid = bp["place_payout"].fillna(0).astype(float).sum()
    hits = bp["place_payout"].notna().sum()
    place_result = {
        "bets": len(bp), "hits": int(hits),
        "roi": paid / (len(bp) * 100) if len(bp) else 0,
    }

    # sanrenpuku_p20_water_filt
    combo_odds_san = get_combo_odds(parquet_root, "sanrenpuku")
    combo_odds_san = combo_odds_san[combo_odds_san["race_id"].isin(test["race_id"])]
    combo_ev_rows = []
    for race_id, race_df in test.groupby("race_id"):
        ro = {"sanrenpuku": combo_odds_san[combo_odds_san["race_id"] == race_id]}
        ev_df = compute_combo_ev_for_race(race_df, ro)
        if not ev_df.empty:
            combo_ev_rows.append(ev_df)
    combo_ev = pd.concat(combo_ev_rows, ignore_index=True) if combo_ev_rows else pd.DataFrame()

    san_result = {"bets": 0, "hits": 0, "roi": 0}
    if not combo_ev.empty:
        sub = combo_ev[(combo_ev["bet_type"] == "sanrenpuku") & (combo_ev["prob"] >= 0.20)]
        idx_t = sub.groupby("race_id")["ev"].idxmax()
        bs = sub.loc[idx_t].copy()
        bs = bs[bs["ev"] > 1.0]
        rid_water = test.groupby("race_id")["water_bin"].first().to_dict()
        ec_lookup = test.groupby("race_id")["entry_count"].first().to_dict()
        top_by_race = (test[test["finish_pos"].notna()]
                       .sort_values(["race_id", "finish_pos"])
                       .groupby("race_id")
                       .apply(lambda g: g.head(3)["horse_no"].astype(int).tolist())
                       .to_dict())
        bs["water_bin"] = bs["race_id"].map(rid_water)
        bs = bs[bs["water_bin"].isin(WATER_FILTERS_BY_STRATEGY["sanrenpuku_top1"])]
        paid_s = 0
        hits_s = 0
        for _, r in bs.iterrows():
            rid = r["race_id"]
            if rid not in top_by_race: continue
            finishers = top_by_race[rid]
            wide_k = wide_top_k_from_entry(int(ec_lookup.get(rid, 10)))
            if determine_combo_hit(r["combination"], finishers, "sanrenpuku", wide_top_k=wide_k):
                paid_s += r["odds_min"] * 100
                hits_s += 1
        san_result = {
            "bets": len(bs), "hits": int(hits_s),
            "roi": paid_s / (len(bs) * 100) if len(bs) else 0,
        }

    print(f"\n  place_top1_water_filt: bets={place_result['bets']}, hit={place_result['hits']}, ROI={place_result['roi']:.3f}")
    print(f"  sanrenpuku_p20_water_filt: bets={san_result['bets']}, hit={san_result['hits']}, ROI={san_result['roi']:.3f}")

    return {"variant": variant, "place": place_result, "sanrenpuku": san_result}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--test-end", default="2026-04-30")
    args = p.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    results = []
    for v in ["baseline", "jockey"]:
        res = run_experiment(variant=v, train_end=args.train_end,
                             test_start=args.test_start, test_end=args.test_end)
        results.append(res)

    print("\n\n=== サマリ ===")
    for r in results:
        p_r = r["place"]
        s_r = r["sanrenpuku"]
        print(f"  {r['variant']:>10}: place ROI={p_r['roi']:.3f} (bets={p_r['bets']}) | "
              f"sanrenpuku ROI={s_r['roi']:.3f} (bets={s_r['bets']})")


if __name__ == "__main__":
    main()
