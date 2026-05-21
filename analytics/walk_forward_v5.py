"""
Walk-Forward v5: 広め買い戦略 (box / nagashi / 不一致シグナル)。

v4 は「各券種で予測 Top1 の組合せ単点買い」だったが、v5 は **複数組合せの
スプレッド** を試す。Kent様 仮説 = 「人気馬が飛びそうなレースを広く拾う」
を実装・検証する。

実装戦略:
  box_top_N        : Top N 頭を全組合せで box買い (N頭 → C(N,k) 点 stake ¥100 ずつ)
  nagashi_top1     : 我々 Top1 を軸に Top2..TopM を相手にする馬連流し
  anti_fav_nagashi : 我々 Top1 軸 - 人気1除く全頭 (人気馬切り戦略)
  disagree_only    : 我々 Top1 ≠ 人気1 のレースだけ Top1 EV ベット (フィルタ)

各戦略は **総ステーク × ROI** で v4 ベース戦略と比較される。

使用例:
  uv run python -m analytics.walk_forward_v5
"""

from __future__ import annotations

import argparse
from itertools import combinations as combos, permutations
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.features import (
    FEATURE_COLS_BOOL,
    FEATURE_COLS_CATEGORICAL,
    FEATURE_COLS_NUMERIC,
    build_feature_matrix,
)
from analytics.plackett_luce import BET_TYPE_META, wide_top_k_from_entry
from analytics.walk_forward import month_range, prepare_features
from analytics.walk_forward_v4 import (
    COMBO_BET_TYPES,
    COMBO_LABEL,
    determine_combo_hit,
    get_combo_odds,
    get_place_payouts,
    get_real_odds,
)
from persist.parquet_writer import DEFAULT_PARQUET_ROOT


# ============================================================================
# Helpers
# ============================================================================

def get_top_n_horses(race_df: pd.DataFrame, n: int, score_col: str = "pred_p_win") -> list[int]:
    """各レースの予測スコア top n 馬番リスト (降順)。"""
    return race_df.nlargest(n, score_col)["horse_no"].astype(int).tolist()


def get_popularity_top1(race_df: pd.DataFrame) -> int | None:
    """1番人気の馬番。"""
    pop1 = race_df[race_df["popularity"] == 1]
    if pop1.empty:
        return None
    return int(pop1["horse_no"].iloc[0])


def generate_combinations(horse_nos: list[int], size: int, ordered: bool) -> list[str]:
    """horse_no リストから組合せ文字列を生成。"""
    iter_fn = permutations if ordered else combos
    seen = set()
    out = []
    for c in iter_fn(horse_nos, size):
        if not ordered:
            c = tuple(sorted(c))
        s = "-".join(str(h) for h in c)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def winning_combo_str(finishers: list[int], size: int, ordered: bool) -> str | None:
    if len(finishers) < size:
        return None
    nums = finishers[:size]
    if not ordered:
        nums = sorted(nums)
    return "-".join(str(h) for h in nums)


# ============================================================================
# 戦略評価
# ============================================================================

def evaluate_box(
    test_df: pd.DataFrame,
    race_finishers: dict[str, list[int]],
    odds_lookup: dict[tuple[str, str], float],
    bet_type: str,
    n_picks: int,
    *,
    disagree_only: bool = False,
    stake_per_combo: int = 100,
) -> dict:
    """Box買い戦略の評価。

    Args:
        odds_lookup: {(race_id, combination_str): odds_min}
        n_picks: Top N 頭を選んで box
        disagree_only: True なら 我々Top1 ≠ 人気1 のレースのみ
    """
    info = BET_TYPE_META[bet_type]
    size = info["size"]
    ordered = info["ordered"]

    races_bet = 0
    races_hit = 0
    total_combos = 0
    total_staked = 0
    total_paid = 0.0

    for race_id, race_df in test_df.groupby("race_id"):
        if len(race_df) < n_picks:
            continue
        # disagree フィルタ
        if disagree_only:
            top1_model = get_top_n_horses(race_df, 1)[0]
            top1_pop = get_popularity_top1(race_df)
            if top1_pop is None or top1_model == top1_pop:
                continue

        top_horses = get_top_n_horses(race_df, n_picks)
        combos_list = generate_combinations(top_horses, size, ordered)
        if not combos_list:
            continue

        # オッズあり組合せのみ採用
        valid_combos = [c for c in combos_list if (race_id, c) in odds_lookup]
        if not valid_combos:
            continue

        race_stake = len(valid_combos) * stake_per_combo

        # 的中判定
        finishers = race_finishers.get(race_id, [])
        win_combo = winning_combo_str(finishers, size, ordered)
        race_paid = 0.0
        if win_combo and win_combo in valid_combos:
            odds = odds_lookup[(race_id, win_combo)]
            race_paid = odds * stake_per_combo
            races_hit += 1

        races_bet += 1
        total_combos += len(valid_combos)
        total_staked += race_stake
        total_paid += race_paid

    return {
        "races_bet": races_bet,
        "races_hit": races_hit,
        "total_combos": total_combos,
        "staked": total_staked,
        "paid": int(total_paid),
        "hit_rate": races_hit / races_bet if races_bet else 0,
        "roi": total_paid / total_staked if total_staked else 0,
    }


def evaluate_nagashi(
    test_df: pd.DataFrame,
    race_finishers: dict[str, list[int]],
    odds_lookup: dict[tuple[str, str], float],
    bet_type: str,
    *,
    anti_fav: bool = False,
    stake_per_combo: int = 100,
) -> dict:
    """馬連流し評価。
    軸 = 我々 Top1。相手 = Top2..TopM (anti_fav=False) or 1番人気除く全頭 (True)。
    """
    info = BET_TYPE_META[bet_type]
    if info["size"] != 2:
        raise ValueError("nagashi は 2-size 券種のみ対応")
    ordered = info["ordered"]

    races_bet = 0
    races_hit = 0
    total_combos = 0
    total_staked = 0
    total_paid = 0.0

    for race_id, race_df in test_df.groupby("race_id"):
        top_horses = get_top_n_horses(race_df, len(race_df))
        if not top_horses:
            continue
        axis = top_horses[0]
        if anti_fav:
            pop1 = get_popularity_top1(race_df)
            if pop1 is None:
                continue
            # axis が 人気1 そのものなら anti-fav として意味なし → skip
            if axis == pop1:
                continue
            partners = [h for h in top_horses[1:] if h != pop1]
        else:
            # 軸 vs 他全頭 (Top2..TopM)
            partners = top_horses[1:]
        if not partners:
            continue

        # combinations
        combos_list = []
        for p in partners:
            if ordered:
                combos_list.append(f"{axis}-{p}")
            else:
                pair = sorted([axis, p])
                combos_list.append(f"{pair[0]}-{pair[1]}")
        combos_list = list(set(combos_list))

        valid_combos = [c for c in combos_list if (race_id, c) in odds_lookup]
        if not valid_combos:
            continue
        race_stake = len(valid_combos) * stake_per_combo

        finishers = race_finishers.get(race_id, [])
        win_combo = winning_combo_str(finishers, 2, ordered)
        race_paid = 0.0
        if win_combo and win_combo in valid_combos:
            odds = odds_lookup[(race_id, win_combo)]
            race_paid = odds * stake_per_combo
            races_hit += 1

        races_bet += 1
        total_combos += len(valid_combos)
        total_staked += race_stake
        total_paid += race_paid

    return {
        "races_bet": races_bet,
        "races_hit": races_hit,
        "total_combos": total_combos,
        "staked": total_staked,
        "paid": int(total_paid),
        "hit_rate": races_hit / races_bet if races_bet else 0,
        "roi": total_paid / total_staked if total_staked else 0,
    }


# ============================================================================
# Walk-Forward 本体
# ============================================================================

def run_v5(
    test_start: str = "2024-01-01",
    test_end: str = "2026-04-30",
    train_start: str = "2014-04-01",
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
    bet_types: list[str] | None = None,
) -> dict:
    import lightgbm as lgb
    bet_types = bet_types or COMBO_BET_TYPES

    print("=== 特徴量行列構築 ===")
    df = build_feature_matrix(parquet_root)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df[df["finish_pos"].notna()].copy()

    print("=== オッズ + 払戻読込 ===")
    real_odds = get_real_odds(parquet_root)
    df = df.merge(real_odds, on=["race_id", "horse_no"], how="left")
    df = df.rename(columns={
        "win_odds": "real_win_odds",
        "place_odds_low": "real_place_low",
        "place_odds_high": "real_place_high",
    })
    place_payout = get_place_payouts(parquet_root)
    df = df.merge(place_payout, on=["race_id", "horse_no"], how="left")

    combo_odds: dict[str, pd.DataFrame] = {}
    for bt in bet_types:
        co = get_combo_odds(parquet_root, bt)
        combo_odds[bt] = co
        print(f"  {COMBO_LABEL[bt]:6s}: {co['race_id'].nunique() if not co.empty else 0:,}レース")

    # popularity 計算 (各レース内 win_odds が小さい順 = 人気順)
    df["popularity"] = df.groupby("race_id")["real_win_odds"].rank(method="min", ascending=True)

    df = prepare_features(df)
    feature_cols = FEATURE_COLS_NUMERIC + FEATURE_COLS_BOOL + FEATURE_COLS_CATEGORICAL

    # top finishers
    fpos_df = df[["race_id", "horse_no", "finish_pos"]].dropna(subset=["finish_pos"])
    fpos_df = fpos_df.sort_values(["race_id", "finish_pos"])
    top_by_race: dict[str, list[int]] = (
        fpos_df.groupby("race_id")
        .apply(lambda g: g.head(3)["horse_no"].astype(int).tolist())
        .to_dict()
    )

    # 各券種の (race_id, combination) → odds_min lookup を事前構築
    odds_lookups: dict[str, dict[tuple[str, str], float]] = {}
    for bt, co in combo_odds.items():
        if co.empty:
            odds_lookups[bt] = {}
            continue
        odds_lookups[bt] = dict(
            zip(zip(co["race_id"], co["combination"]), co["odds_min"].astype(float))
        )

    months = month_range(test_start, test_end)
    monthly_records: list[dict] = []
    print(f"\n=== Walk-Forward v5 ({test_start} 〜 {test_end}, {len(months)}ヶ月) ===")

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

        test = test.copy()
        test["pred_p_win"] = model_win.predict_proba(X_test)[:, 1]

        # === Box 戦略 ===
        for bt, n_picks_list in [
            ("umaren",     [3, 4, 5]),
            ("umatan",     [3, 4]),
            ("sanrenpuku", [3, 4, 5]),
            # wide / sanrentan は今回省略 (前者複雑・後者未取得)
        ]:
            if bt not in bet_types or not odds_lookups[bt]:
                continue
            for n_picks in n_picks_list:
                for disagree in [False, True]:
                    res = evaluate_box(
                        test, top_by_race, odds_lookups[bt],
                        bet_type=bt, n_picks=n_picks,
                        disagree_only=disagree,
                    )
                    if res["races_bet"] == 0:
                        continue
                    label = f"{bt}_box_top{n_picks}"
                    if disagree:
                        label += "_disagree"
                    monthly_records.append({
                        "month": m, "strategy": label,
                        "races_bet": res["races_bet"],
                        "races_hit": res["races_hit"],
                        "combos": res["total_combos"],
                        "hit_rate": res["hit_rate"],
                        "staked": res["staked"],
                        "paid": res["paid"],
                        "roi": round(res["roi"], 3),
                    })

        # === 馬連流し ===
        if "umaren" in bet_types and odds_lookups["umaren"]:
            for anti_fav in [False, True]:
                res = evaluate_nagashi(
                    test, top_by_race, odds_lookups["umaren"],
                    bet_type="umaren", anti_fav=anti_fav,
                )
                if res["races_bet"] == 0:
                    continue
                label = "umaren_nagashi_top1"
                if anti_fav:
                    label += "_anti_fav"
                monthly_records.append({
                    "month": m, "strategy": label,
                    "races_bet": res["races_bet"],
                    "races_hit": res["races_hit"],
                    "combos": res["total_combos"],
                    "hit_rate": res["hit_rate"],
                    "staked": res["staked"],
                    "paid": res["paid"],
                    "roi": round(res["roi"], 3),
                })

        if (m_idx + 1) % 6 == 0:
            print(f"  進捗: {m_idx+1}/{len(months)} ヶ月")

    monthly = pd.DataFrame(monthly_records)
    overall = (
        monthly.groupby("strategy")
        .agg(
            races_bet=("races_bet", "sum"),
            races_hit=("races_hit", "sum"),
            combos=("combos", "sum"),
            staked=("staked", "sum"),
            paid=("paid", "sum"),
        )
        .reset_index()
    )
    overall["hit_rate"] = (overall["races_hit"] / overall["races_bet"]).round(3)
    overall["roi"] = (overall["paid"] / overall["staked"]).round(3)
    overall["avg_combos_per_race"] = (overall["combos"] / overall["races_bet"]).round(1)
    return {"monthly": monthly, "overall": overall}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-start", default="2024-01-01")
    p.add_argument("--test-end", default="2026-04-30")
    p.add_argument("--output", default="data/backtest/v5_spread_strategies.parquet")
    p.add_argument("--bet-types", default=None, help="カンマ区切り")
    args = p.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    bet_types = [b.strip() for b in args.bet_types.split(",")] if args.bet_types else None
    res = run_v5(test_start=args.test_start, test_end=args.test_end, bet_types=bet_types)

    print()
    print("=== Walk-Forward v5 (広め買い戦略) 結果 ===")
    print(res["overall"].sort_values("roi", ascending=False).to_string(index=False))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res["monthly"].to_parquet(out, index=False, compression="zstd")
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
