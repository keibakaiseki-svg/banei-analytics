"""
Walk-Forward v6: water filter 統合版。

v4 の主要戦略 (place_top1, sanrenpuku_p>=0.20) に water_bin フィルタを meta-filter
として追加して raw vs filtered の ROI を比較する。

水分量による edge の出方:
- place_top1     : light (1-2) と wet (3-4) で ROI ≥ 1.05
- sanrenpuku_p20 : light (1-2) と normal (2-3) で ROI ≥ 1.11
- それ以外の bin (dry/heavy/不適合) はスキップ

dry (<1.0) はどの戦略でも勝てない罠 → 全戦略 default で除外。

使用例:
    uv run python -m analytics.walk_forward_v6
    uv run python -m analytics.walk_forward_v6 --test-start 2019-10-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from analytics.features import (
    FEATURE_COLS_BOOL,
    FEATURE_COLS_CATEGORICAL,
    FEATURE_COLS_NUMERIC,
    build_feature_matrix,
)
from analytics.plackett_luce import combo_probs_for_race, wide_top_k_from_entry
from analytics.walk_forward import month_range, prepare_features
from analytics.walk_forward_v4 import (
    COMBO_BET_TYPES,
    COMBO_LABEL,
    compute_combo_ev_for_race,
    determine_combo_hit,
    get_combo_odds,
    get_place_payouts,
    get_real_odds,
)
from persist.parquet_writer import DEFAULT_PARQUET_ROOT


# ===== Water filter 定義 (v4_extended 年別分析から導出) =====

WATER_BINS = [
    ("dry",    (0.0, 1.0)),
    ("light",  (1.0, 2.0)),
    ("normal", (2.0, 3.0)),
    ("wet",    (3.0, 4.0)),
    ("heavy",  (4.0, 99.0)),
]


def water_bin(pct: float | None) -> str:
    if pct is None or pd.isna(pct):
        return "unknown"
    for label, (lo, hi) in WATER_BINS:
        if lo <= pct < hi:
            return label
    return "unknown"


# 戦略別「収益化可能な水分量 bins」
WATER_FILTERS_BY_STRATEGY = {
    "place_top1":     {"light", "wet"},
    "sanrenpuku_top1": {"light", "normal"},
    # 他は default = dry 以外を許可 (まだ分析サンプル少)
    "_default_skip_dry": {"light", "normal", "wet", "heavy"},
}


def get_race_water(parquet_root: Path) -> pd.DataFrame:
    con = duckdb.connect()
    return con.execute(
        f"SELECT race_id, track_water_pct FROM read_parquet('{parquet_root}/races/**/*.parquet', hive_partitioning=true)"
    ).fetchdf()


# ============================================================================
# Walk-Forward
# ============================================================================

def run_v6(
    test_start: str = "2019-10-01",
    test_end: str = "2026-04-30",
    train_start: str = "2014-04-01",
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
    bet_types: list[str] | None = None,
    bets_log_path: Path | None = None,
) -> dict:
    import lightgbm as lgb

    bet_types = bet_types or COMBO_BET_TYPES
    all_bets_log: list[dict] = []

    print("=== 特徴量行列構築 ===")
    df = build_feature_matrix(parquet_root)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df[df["finish_pos"].notna()].copy()

    print("=== オッズ + 払戻読込 ===")
    real_odds = get_real_odds(parquet_root)
    df = df.merge(real_odds, on=["race_id", "horse_no"], how="left").rename(columns={
        "win_odds": "real_win_odds",
        "place_odds_low": "real_place_low",
        "place_odds_high": "real_place_high",
    })
    df = df.merge(get_place_payouts(parquet_root), on=["race_id", "horse_no"], how="left")

    # water_bin 付加
    water = get_race_water(parquet_root)
    water["water_bin"] = water["track_water_pct"].apply(water_bin)
    df = df.merge(water[["race_id", "water_bin"]], on="race_id", how="left")
    print(f"  water_bin 分布: {df.groupby('water_bin')['race_id'].nunique().to_dict()}")

    combo_odds: dict[str, pd.DataFrame] = {}
    for bt in bet_types:
        co = get_combo_odds(parquet_root, bt)
        combo_odds[bt] = co
        print(f"  {COMBO_LABEL[bt]:6s}: {co['race_id'].nunique() if not co.empty else 0:,}レース")

    df = prepare_features(df)
    feature_cols = FEATURE_COLS_NUMERIC + FEATURE_COLS_BOOL + FEATURE_COLS_CATEGORICAL

    fpos_df = df[["race_id", "horse_no", "finish_pos"]].dropna(subset=["finish_pos"])
    fpos_df = fpos_df.sort_values(["race_id", "finish_pos"])
    top_by_race: dict[str, list[int]] = (
        fpos_df.groupby("race_id").apply(lambda g: g.head(3)["horse_no"].astype(int).tolist()).to_dict()
    )

    months = month_range(test_start, test_end)
    records: list[dict] = []
    print(f"\n=== Walk-Forward v6 ({test_start}〜{test_end}, {len(months)}ヶ月) ===")

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

        # === place_top1 ===
        place_bets = (test[test["place_eligible"]]
                      .groupby("race_id")["pred_p_place"].idxmax())
        bp = test.loc[place_bets].copy()
        bp = bp[bp["ev_place"] > 1.0].dropna(subset=["real_place_low"])
        if not bp.empty:
            for variant, allowed_bins in [
                ("place_top1_raw", None),
                ("place_top1_water_filt", WATER_FILTERS_BY_STRATEGY["place_top1"]),
            ]:
                bb = bp if allowed_bins is None else bp[bp["water_bin"].isin(allowed_bins)]
                if bb.empty:
                    continue
                paid = bb["place_payout"].fillna(0).astype(float).sum()
                hits = bb["place_payout"].notna().sum()
                records.append({
                    "month": m, "strategy": variant,
                    "bets": len(bb), "wins": int(hits),
                    "win_rate": hits / len(bb),
                    "staked": len(bb) * 100, "paid": int(paid),
                    "roi": round(paid / (len(bb) * 100), 3),
                })
                if bets_log_path is not None:
                    for _, r in bb.iterrows():
                        hit = bool(pd.notna(r["place_payout"]))
                        all_bets_log.append({
                            "strategy": variant, "month": str(m),
                            "race_id": r["race_id"], "bet_type": "place",
                            "combination": str(int(r["horse_no"])),
                            "prob": float(r["pred_p_place"]),
                            "ev": float(r["ev_place"]),
                            "odds": float(r["real_place_low"]),
                            "hit": hit,
                            "payout_per_100yen": float(r["place_payout"]) if hit else 0.0,
                            "water_bin": r["water_bin"],
                        })

        # === combo (sanrenpuku top1 p>=0.20) ===
        test_race_ids = set(test["race_id"].unique())
        month_combo_odds = {
            bt: co[co["race_id"].isin(test_race_ids)] for bt, co in combo_odds.items()
        }
        combo_ev_rows = []
        for race_id, race_df in test.groupby("race_id"):
            ro = {bt: month_combo_odds[bt][month_combo_odds[bt]["race_id"] == race_id] for bt in bet_types}
            ev_df = compute_combo_ev_for_race(race_df, ro)
            if not ev_df.empty:
                combo_ev_rows.append(ev_df)
        combo_ev = pd.concat(combo_ev_rows, ignore_index=True) if combo_ev_rows else pd.DataFrame()

        if combo_ev.empty:
            continue
        # water_bin を combo_ev に付加
        rid_water = test.groupby("race_id")["water_bin"].first().to_dict()
        combo_ev["water_bin"] = combo_ev["race_id"].map(rid_water)
        ec_lookup = test.groupby("race_id")["entry_count"].first().to_dict()

        # sanrenpuku top1 EV>1.0, p>=0.20
        sub = combo_ev[(combo_ev["bet_type"] == "sanrenpuku") & (combo_ev["prob"] >= 0.20)]
        if sub.empty:
            continue
        idx_top = sub.groupby("race_id")["ev"].idxmax()
        bs = sub.loc[idx_top].copy()
        bs = bs[bs["ev"] > 1.0]
        if bs.empty:
            continue

        for variant, allowed_bins in [
            ("sanrenpuku_p20_raw", None),
            ("sanrenpuku_p20_water_filt", WATER_FILTERS_BY_STRATEGY["sanrenpuku_top1"]),
        ]:
            bb = bs if allowed_bins is None else bs[bs["water_bin"].isin(allowed_bins)]
            if bb.empty:
                continue
            paid = 0.0
            hits = 0
            for _, row in bb.iterrows():
                rid = row["race_id"]
                if rid not in top_by_race:
                    continue
                finishers = top_by_race[rid]
                wide_k = wide_top_k_from_entry(int(ec_lookup.get(rid, 10)))
                hit = determine_combo_hit(row["combination"], finishers, "sanrenpuku", wide_top_k=wide_k)
                if hit:
                    paid += row["odds_min"] * 100
                    hits += 1
                if bets_log_path is not None:
                    all_bets_log.append({
                        "strategy": variant, "month": str(m),
                        "race_id": rid, "bet_type": "sanrenpuku",
                        "combination": row["combination"],
                        "prob": float(row["prob"]),
                        "ev": float(row["ev"]),
                        "odds": float(row["odds_min"]),
                        "hit": bool(hit),
                        "payout_per_100yen": float(row["odds_min"] * 100) if hit else 0.0,
                        "water_bin": row["water_bin"],
                    })
            records.append({
                "month": m, "strategy": variant,
                "bets": len(bb), "wins": int(hits),
                "win_rate": hits / len(bb),
                "staked": len(bb) * 100, "paid": int(paid),
                "roi": round(paid / (len(bb) * 100), 3),
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
             staked=("staked", "sum"), paid=("paid", "sum"))
        .reset_index()
    )
    overall["win_rate"] = (overall["wins"] / overall["bets"]).round(3)
    overall["roi"] = (overall["paid"] / overall["staked"]).round(3)
    return {"monthly": monthly, "overall": overall}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-start", default="2019-10-01")
    p.add_argument("--test-end", default="2026-04-30")
    p.add_argument("--output", default="data/backtest/v6_water_filt.parquet")
    p.add_argument("--bets-log-path", default="data/backtest/v6_water_filt_bets_log.parquet")
    args = p.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    res = run_v6(
        test_start=args.test_start, test_end=args.test_end,
        bets_log_path=Path(args.bets_log_path) if args.bets_log_path else None,
    )

    print()
    print("=== Walk-Forward v6 (water filter付き) 結果 ===")
    print(res["overall"].sort_values("roi", ascending=False).to_string(index=False))

    # raw vs filtered の改善幅サマリ
    print("\n=== Raw vs Water-Filtered 改善幅 ===")
    o = res["overall"].set_index("strategy")
    for base in ["place_top1", "sanrenpuku_p20"]:
        raw_key = f"{base}_raw"
        filt_key = f"{base}_water_filt"
        if raw_key in o.index and filt_key in o.index:
            raw_roi = o.loc[raw_key, "roi"]
            filt_roi = o.loc[filt_key, "roi"]
            raw_bets = o.loc[raw_key, "bets"]
            filt_bets = o.loc[filt_key, "bets"]
            print(f"  {base}: raw ROI {raw_roi:.3f} (bets {raw_bets}) → filt {filt_roi:.3f} (bets {filt_bets})  "
                  f"Δ {filt_roi-raw_roi:+.3f}pp / bets {(1-filt_bets/raw_bets)*100:+.1f}%")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res["monthly"].to_parquet(out, index=False, compression="zstd")
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
