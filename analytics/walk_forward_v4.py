"""
Walk-Forward v4: 全7券種のEVを比較・自動選択するバックテスト。

v3 (単勝・複勝) を拡張:
- 馬連/馬単/ワイド/三連複/三連単 オッズを netkeiba scraping データから取得
- Plackett-Luce 近似で組合せ確率を導出 (pred_p_win から)
- 各券種で EV最大の組合せを抽出、しきい値で発券判定
- 「auto_max_ev」: 全7券種を比較してレースごとに最良券種を採用
- 払戻はオッズ×ベット(=¥100)から算出 (組合せ系は payouts テーブル非依存)

戦略一覧:
- bet_<type>_top1_ev>1.0  : 各券種で予測最大EV馬の単独ベット
- bet_auto_max_ev>1.0     : 全券種の最良EVを採用
- bet_auto_max_ev>1.05/1.10/1.20 : しきい値別

評価期間: 2024-01-01 〜 2026-04-30 (netkeiba実オッズ100%, combo odds Phase 6取得後)
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
from analytics.plackett_luce import (
    BET_TYPE_META,
    combo_probs_for_race,
    wide_top_k_from_entry,
)
from analytics.walk_forward import month_range, prepare_features
from persist.parquet_writer import DEFAULT_PARQUET_ROOT

COMBO_BET_TYPES = ["umaren", "umatan", "wide", "sanrenpuku", "sanrentan"]
COMBO_LABEL = {
    "umaren": "馬連", "umatan": "馬単", "wide": "ワイド",
    "sanrenpuku": "三連複", "sanrentan": "三連単",
}


# ============================================================================
# データ読込
# ============================================================================

def get_real_odds(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> pd.DataFrame:
    df = pd.read_parquet(parquet_root / "odds_netkeiba.parquet")
    return df[["race_id_local", "horse_no", "win_odds", "place_odds_low", "place_odds_high"]].rename(
        columns={"race_id_local": "race_id"}
    )


def get_place_payouts(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> pd.DataFrame:
    con = duckdb.connect()
    glob = str(parquet_root / "payouts/**/*.parquet")
    return con.execute(
        f"""
        SELECT race_id, CAST(combination AS INTEGER) AS horse_no, payout_yen AS place_payout
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE bet_type = '複勝' AND combination NOT LIKE '%-%'
        """
    ).fetchdf()


def get_combo_odds(parquet_root: Path, bet_type: str) -> pd.DataFrame:
    """券種別 netkeiba combo odds を読込。
    戻り値: race_id, bet_type, combination, odds_min, odds_max
    """
    f = parquet_root / f"odds_netkeiba_{bet_type}.parquet"
    if not f.exists():
        return pd.DataFrame(columns=["race_id", "bet_type", "combination", "odds_min", "odds_max"])
    df = pd.read_parquet(f)
    df = df.rename(columns={"race_id_local": "race_id"})
    return df[["race_id", "bet_type", "combination", "odds_min", "odds_max"]]


# ============================================================================
# 組合せ的中判定
# ============================================================================

def determine_combo_hit(
    combination: str,
    top_finishers: list[int],
    bet_type: str,
    *,
    wide_top_k: int | None = None,
) -> bool:
    """Args:
        combination: '3-7' or '3-7-10' (馬単/三連単は順序保持)
        top_finishers: [1着馬番, 2着馬番, 3着馬番] (3着がなければ短くてもOK)
        bet_type: umaren/umatan/wide/sanrenpuku/sanrentan
        wide_top_k: ワイド対象範囲 (8+: 3, 5-7: 2)
    """
    try:
        nums = [int(n) for n in combination.split("-")]
    except ValueError:
        return False
    if bet_type == "umaren":
        if len(top_finishers) < 2:
            return False
        return set(nums) == set(top_finishers[:2])
    if bet_type == "umatan":
        if len(top_finishers) < 2:
            return False
        return tuple(nums) == tuple(top_finishers[:2])
    if bet_type == "wide":
        if wide_top_k is None or len(top_finishers) < wide_top_k:
            return False
        return all(n in top_finishers[:wide_top_k] for n in nums)
    if bet_type == "sanrenpuku":
        if len(top_finishers) < 3:
            return False
        return set(nums) == set(top_finishers[:3])
    if bet_type == "sanrentan":
        if len(top_finishers) < 3:
            return False
        return tuple(nums) == tuple(top_finishers[:3])
    return False


# ============================================================================
# レース単位の EV 計算
# ============================================================================

def compute_combo_ev_for_race(
    race_df: pd.DataFrame,
    odds_long: dict[str, pd.DataFrame],
    *,
    score_col: str = "pred_p_win",
) -> pd.DataFrame:
    """1レース分の全組合せ EV を計算。

    Args:
        race_df: 1レース分の予測 (horse_no, pred_p_win 等)
        odds_long: {bet_type: race-filtered odds dataframe (combination, odds_min, odds_max)}

    Returns:
        DataFrame[race_id, bet_type, combination, prob, odds_min, odds_max, ev]
    """
    race_id = race_df["race_id"].iloc[0]
    entry_count = len(race_df)
    horse_nos = race_df["horse_no"].tolist()
    scores = race_df[score_col].to_numpy()

    rows: list[dict] = []
    wide_k = wide_top_k_from_entry(entry_count)

    for bt in COMBO_BET_TYPES:
        odds_df = odds_long.get(bt, pd.DataFrame())
        if odds_df.empty:
            continue
        if bt == "wide" and wide_k is None:
            continue

        probs = combo_probs_for_race(
            horse_nos, scores, bt,
            wide_top_k=wide_k if bt == "wide" else None,
        )

        # combination → prob, marge with odds
        for _, o in odds_df.iterrows():
            combo = o["combination"]
            prob = probs.get(combo, 0.0)
            odds_min = o["odds_min"]
            odds_max = o["odds_max"]
            if pd.isna(odds_min) or odds_min <= 0:
                continue
            # ワイドは min (低値) を採用 (保守的EV)
            ev = prob * float(odds_min)
            rows.append({
                "race_id": race_id,
                "bet_type": bt,
                "combination": combo,
                "prob": prob,
                "odds_min": float(odds_min),
                "odds_max": float(odds_max) if pd.notna(odds_max) else None,
                "ev": ev,
            })
    return pd.DataFrame(rows)


# ============================================================================
# Walk-Forward 本体
# ============================================================================

def run_v4(
    test_start: str = "2024-01-01",
    test_end: str = "2026-04-30",
    train_start: str = "2014-04-01",
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
    bet_types: list[str] | None = None,
    bets_log_path: Path | None = None,
) -> dict:
    """Walk-Forward v4 メイン。

    bets_log_path が指定されると、各戦略のベット履歴を Parquet に保存。
    後段の bankroll simulator (analytics/bankroll_sim.py) で資金管理シミュレーション
    に使用できる。
    """
    import lightgbm as lgb

    bet_types = bet_types or COMBO_BET_TYPES
    all_bets_log: list[dict] = []

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
    place_payout = get_place_payouts(parquet_root)
    df = df.merge(place_payout, on=["race_id", "horse_no"], how="left")

    print(f"=== combo オッズ読込 ({len(bet_types)}券種) ===")
    combo_odds: dict[str, pd.DataFrame] = {}
    for bt in bet_types:
        co = get_combo_odds(parquet_root, bt)
        combo_odds[bt] = co
        print(f"  {COMBO_LABEL[bt]:6s}: {co['race_id'].nunique() if not co.empty else 0:,}レース / {len(co):,}行")

    df = prepare_features(df)
    feature_cols = FEATURE_COLS_NUMERIC + FEATURE_COLS_BOOL + FEATURE_COLS_CATEGORICAL

    # 各レースの top finishers 抽出 (的中判定用)
    fpos_df = df[["race_id", "horse_no", "finish_pos"]].dropna(subset=["finish_pos"])
    fpos_df = fpos_df.sort_values(["race_id", "finish_pos"])
    top_by_race: dict[str, list[int]] = (
        fpos_df.groupby("race_id")
        .apply(lambda g: g.head(3)["horse_no"].astype(int).tolist())
        .to_dict()
    )

    months = month_range(test_start, test_end)
    monthly_records: list[dict] = []
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
            train_place[feature_cols],
            train_place["target_place"].astype(int),
            categorical_feature=FEATURE_COLS_CATEGORICAL,
        )

        test = test.copy()
        test["pred_p_win"] = model_win.predict_proba(X_test)[:, 1]
        test["pred_p_place"] = model_place.predict_proba(X_test)[:, 1]

        # 単複EV (v3互換)
        test["ev_win"] = test["pred_p_win"] * test["real_win_odds"]
        test["ev_place"] = test["pred_p_place"] * test["real_place_low"]
        test["place_eligible"] = test["entry_count"] >= 5

        # === Combo EV をレースごとに計算 ===
        # まず月内テストレースに該当するoddsを抽出 (高速化)
        test_race_ids = set(test["race_id"].unique())
        month_combo_odds: dict[str, pd.DataFrame] = {
            bt: co[co["race_id"].isin(test_race_ids)] for bt, co in combo_odds.items()
        }

        combo_ev_rows: list[pd.DataFrame] = []
        for race_id, race_df in test.groupby("race_id"):
            race_odds = {bt: month_combo_odds[bt][month_combo_odds[bt]["race_id"] == race_id] for bt in bet_types}
            ev_df = compute_combo_ev_for_race(race_df, race_odds)
            if not ev_df.empty:
                combo_ev_rows.append(ev_df)
        combo_ev = pd.concat(combo_ev_rows, ignore_index=True) if combo_ev_rows else pd.DataFrame()

        # === 戦略: 単勝/複勝/各券種 top1 EV>1.0 ===
        # 1) 単勝 Top1
        idx_win = test.groupby("race_id")["pred_p_win"].idxmax()
        bets_win = test.loc[idx_win].copy()
        bets_win = bets_win[bets_win["ev_win"] > 1.0].dropna(subset=["real_win_odds"])
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
            if bets_log_path is not None:
                for _, r in bets_win.iterrows():
                    hit = bool(r["finish_pos"] == 1)
                    all_bets_log.append({
                        "strategy": "win_top1_ev>1.0", "month": str(m),
                        "race_id": r["race_id"], "bet_type": "win",
                        "combination": str(int(r["horse_no"])),
                        "prob": float(r["pred_p_win"]),
                        "ev": float(r["ev_win"]),
                        "odds": float(r["real_win_odds"]),
                        "hit": hit,
                        "payout_per_100yen": float(r["real_win_odds"] * 100) if hit else 0.0,
                    })

        # 2) 複勝 Top1 (v3 と同じ)
        test_place = test[test["place_eligible"]].copy()
        if not test_place.empty:
            idx_place = test_place.groupby("race_id")["pred_p_place"].idxmax()
            bets_place = test_place.loc[idx_place].copy()
            bets_place = bets_place[bets_place["ev_place"] > 1.0].dropna(subset=["real_place_low"])
            if not bets_place.empty:
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
                if bets_log_path is not None:
                    for _, r in bets_place.iterrows():
                        hit = bool(pd.notna(r["place_payout"]))
                        all_bets_log.append({
                            "strategy": "place_top1_ev>1.0", "month": str(m),
                            "race_id": r["race_id"], "bet_type": "place",
                            "combination": str(int(r["horse_no"])),
                            "prob": float(r["pred_p_place"]),
                            "ev": float(r["ev_place"]),
                            "odds": float(r["real_place_low"]),
                            "hit": hit,
                            "payout_per_100yen": float(r["place_payout"]) if hit else 0.0,
                        })

        # 3) 各 combo 券種で top1 EV>1.0  (確率しきい値で longshot 排除)
        if not combo_ev.empty:
            ec_lookup = test.groupby("race_id")["entry_count"].first().to_dict()
            for bt in bet_types:
                sub = combo_ev[combo_ev["bet_type"] == bt]
                if sub.empty:
                    continue
                for prob_thr in [0.0, 0.05, 0.10, 0.20]:
                    sub_f = sub[sub["prob"] >= prob_thr]
                    if sub_f.empty:
                        continue
                    idx_top = sub_f.groupby("race_id")["ev"].idxmax()
                    bets = sub_f.loc[idx_top].copy()
                    bets = bets[bets["ev"] > 1.0]
                    if bets.empty:
                        continue
                    staked = len(bets) * 100
                    paid = 0
                    hits = 0
                    for _, row in bets.iterrows():
                        rid = row["race_id"]
                        if rid not in top_by_race:
                            continue
                        finishers = top_by_race[rid]
                        entry = int(ec_lookup.get(rid, 10))
                        wide_k = wide_top_k_from_entry(entry) if bt == "wide" else None
                        if determine_combo_hit(row["combination"], finishers, bt, wide_top_k=wide_k):
                            paid += row["odds_min"] * 100
                            hits += 1
                    strategy_label = f"{bt}_top1_ev>1.0_p>={prob_thr:.2f}"
                    monthly_records.append({
                        "month": m, "strategy": strategy_label,
                        "bets": len(bets), "wins": int(hits),
                        "win_rate": hits / len(bets),
                        "staked": staked, "paid": int(paid),
                        "roi": round(paid / staked, 3),
                    })
                    if bets_log_path is not None:
                        for _, row in bets.iterrows():
                            rid = row["race_id"]
                            finishers = top_by_race.get(rid, [])
                            entry = int(ec_lookup.get(rid, 10))
                            wide_k = wide_top_k_from_entry(entry) if bt == "wide" else None
                            hit = determine_combo_hit(row["combination"], finishers, bt, wide_top_k=wide_k)
                            all_bets_log.append({
                                "strategy": strategy_label, "month": str(m),
                                "race_id": rid, "bet_type": bt,
                                "combination": row["combination"],
                                "prob": float(row["prob"]),
                                "ev": float(row["ev"]),
                                "odds": float(row["odds_min"]),
                                "hit": bool(hit),
                                "payout_per_100yen": float(row["odds_min"] * 100) if hit else 0.0,
                            })

        # 4) auto_max_ev: 全券種からレースごと最良 EV を採用 (prob しきい値付き)
        cand_rows: list[dict] = []
        for _, r in test.iterrows():
            if pd.notna(r["real_win_odds"]) and r["ev_win"] > 0:
                cand_rows.append({
                    "race_id": r["race_id"], "bet_type": "win",
                    "combination": str(int(r["horse_no"])),
                    "prob": float(r["pred_p_win"]),
                    "ev": r["ev_win"],
                    "odds_min": r["real_win_odds"], "odds_max": None,
                    "horse_no": int(r["horse_no"]), "entry_count": int(r["entry_count"]),
                })
        for _, r in test_place.iterrows():
            if pd.notna(r["real_place_low"]) and r["ev_place"] > 0:
                cand_rows.append({
                    "race_id": r["race_id"], "bet_type": "place",
                    "combination": str(int(r["horse_no"])),
                    "prob": float(r["pred_p_place"]),
                    "ev": r["ev_place"],
                    "odds_min": r["real_place_low"], "odds_max": r["real_place_high"],
                    "horse_no": int(r["horse_no"]), "entry_count": int(r["entry_count"]),
                })
        if not combo_ev.empty:
            ec_map = test.groupby("race_id")["entry_count"].first().to_dict()
            for _, r in combo_ev.iterrows():
                cand_rows.append({
                    "race_id": r["race_id"], "bet_type": r["bet_type"],
                    "combination": r["combination"],
                    "prob": float(r["prob"]),
                    "ev": r["ev"],
                    "odds_min": r["odds_min"], "odds_max": r["odds_max"],
                    "horse_no": None, "entry_count": ec_map.get(r["race_id"], 10),
                })
        candidates = pd.DataFrame(cand_rows) if cand_rows else pd.DataFrame()

        if not candidates.empty:
            for prob_thr in [0.0, 0.05, 0.10, 0.20]:
                for ev_thr in [1.0, 1.10, 1.20]:
                    cand = candidates[(candidates["ev"] > ev_thr) & (candidates["prob"] >= prob_thr)]
                    if cand.empty:
                        continue
                    idx_max = cand.groupby("race_id")["ev"].idxmax()
                    bets = cand.loc[idx_max].copy()
                    staked = len(bets) * 100
                    paid = 0
                    hits = 0
                    for _, row in bets.iterrows():
                        rid = row["race_id"]
                        bt = row["bet_type"]
                        if rid not in top_by_race:
                            continue
                        finishers = top_by_race[rid]
                        if bt == "win":
                            if int(row["horse_no"]) == finishers[0]:
                                paid += row["odds_min"] * 100
                                hits += 1
                        elif bt == "place":
                            entry = int(row["entry_count"])
                            place_k = 3 if entry >= 8 else 2
                            # 複勝の払戻は odds_min を保守的に使用 (動的払戻の下限近似)
                            if int(row["horse_no"]) in finishers[:place_k]:
                                paid += row["odds_min"] * 100
                                hits += 1
                        else:
                            wide_k = wide_top_k_from_entry(int(row["entry_count"])) if bt == "wide" else None
                            if determine_combo_hit(row["combination"], finishers, bt, wide_top_k=wide_k):
                                paid += row["odds_min"] * 100
                                hits += 1
                    monthly_records.append({
                        "month": m, "strategy": f"auto_max_ev>{ev_thr:.2f}_p>={prob_thr:.2f}",
                        "bets": len(bets), "wins": int(hits),
                        "win_rate": hits / len(bets),
                        "staked": staked, "paid": int(paid),
                        "roi": round(paid / staked, 3),
                    })

        if (m_idx + 1) % 6 == 0:
            print(f"  進捗: {m_idx+1}/{len(months)} ヶ月")

    # ベットログ保存 (--bets-log-path 指定時)
    if bets_log_path is not None and all_bets_log:
        bets_log_path = Path(bets_log_path)
        bets_log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_bets_log).to_parquet(bets_log_path, index=False, compression="zstd")
        print(f"\nベットログ保存: {bets_log_path} ({len(all_bets_log):,} bets)")

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
    p.add_argument("--output", default="data/backtest/v4_all_bet_types.parquet")
    p.add_argument("--bet-types", default=None, help="カンマ区切り")
    p.add_argument("--bets-log-path", default=None, help="戦略別ベット履歴の Parquet 保存先 (bankroll sim 用)")
    args = p.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    bet_types = [b.strip() for b in args.bet_types.split(",")] if args.bet_types else None
    res = run_v4(
        test_start=args.test_start, test_end=args.test_end,
        bet_types=bet_types,
        bets_log_path=Path(args.bets_log_path) if args.bets_log_path else None,
    )

    print()
    print("=== Walk-Forward v4 (全7券種 EV比較) 結果 ===")
    print(res["overall"].sort_values("roi", ascending=False).to_string(index=False))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res["monthly"].to_parquet(out, index=False, compression="zstd")
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
