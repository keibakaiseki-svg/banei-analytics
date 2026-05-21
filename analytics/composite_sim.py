"""
複合戦略シミュレーター (place_top1 + sanrenpuku_p>=0.20 並走)。

現実的な運用シナリオを模した資金フローシミュレーション:
- 初期資金: ¥100,000
- 毎月注入: ¥100,000
- SPAT4 還元: 0.57% (= 100ポイント/¥100, 17.5Mポイント=¥100K)
- 複数戦略を独立にベット (券種違いで同レース重複可)
- 戦略別の bet sizing rule を指定

入力: walk_forward_v4 が `--bets-log-path` で出力したベット履歴

使用例:
  uv run python -m analytics.composite_sim
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def simulate_composite(
    bets_df: pd.DataFrame,
    *,
    strategies: dict[str, dict],
    initial: float = 100_000,
    monthly_inject: float = 100_000,
    spat4_rebate_rate: float = 0.0057,
    min_stake: float = 100,
) -> dict:
    """
    Args:
        bets_df: 全戦略のベットログ (sorted by month, race_id) — strategy 列でフィルタ
        strategies: {strategy_name: {"mode": "flat"|"fraction", "stake": ... | "frac": ...}}
        initial: 初期資金
        monthly_inject: 毎月開始時の追加投入額
        spat4_rebate_rate: 総ステークに対する還元率

    Returns:
        final, total_deposited, profit, effective_ratio, placed, hits, max_dd, monthly_history(DataFrame)
    """
    bets_df = bets_df[bets_df["strategy"].isin(strategies)].copy()
    bets_df = bets_df.sort_values(["month", "race_id", "strategy"]).reset_index(drop=True)

    bankroll = initial
    total_deposited = initial
    peak_bankroll = initial
    max_dd = 0.0

    placed = 0
    hits = 0
    total_staked = 0.0
    total_paid = 0.0
    total_rebate = 0.0

    monthly_records: list[dict] = []
    monthly_stake = 0.0
    monthly_pnl = 0.0
    monthly_placed = 0
    monthly_hits = 0
    last_month: str | None = None

    def close_month(month: str) -> None:
        nonlocal bankroll, total_rebate, monthly_stake, monthly_pnl, monthly_placed, monthly_hits, peak_bankroll, max_dd
        rebate = monthly_stake * spat4_rebate_rate
        bankroll += rebate
        total_rebate += rebate
        monthly_pnl += rebate
        peak_bankroll = max(peak_bankroll, bankroll)
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        max_dd = max(max_dd, dd)
        monthly_records.append({
            "month": month,
            "end_bankroll": int(bankroll),
            "monthly_stake": int(monthly_stake),
            "monthly_pnl": int(monthly_pnl),
            "rebate": int(rebate),
            "placed": monthly_placed,
            "hits": monthly_hits,
        })
        monthly_stake = 0.0
        monthly_pnl = 0.0
        monthly_placed = 0
        monthly_hits = 0

    for _, bet in bets_df.iterrows():
        m = bet["month"]

        # 月境界
        if last_month is None:
            last_month = m
            # 初月は inject 不要 (initial が初月分とみなす)
        elif m != last_month:
            close_month(last_month)
            bankroll += monthly_inject
            total_deposited += monthly_inject
            last_month = m

        rule = strategies[bet["strategy"]]
        if rule["mode"] == "flat":
            stake = float(rule["stake"])
        elif rule["mode"] == "fraction":
            stake = max(min_stake, rule["frac"] * bankroll)
        else:
            raise ValueError(f"unknown mode: {rule['mode']}")

        if rule.get("max_stake"):
            stake = min(stake, rule["max_stake"])
        stake = min(stake, bankroll)
        if stake < min_stake:
            # 破産: ベット不能・以降この戦略はスキップ (他戦略は継続可能)
            continue

        odds_payout_ratio = float(bet["payout_per_100yen"]) / 100.0
        if bool(bet["hit"]):
            payout = stake * odds_payout_ratio
            bankroll = bankroll - stake + payout
            monthly_pnl += payout - stake
            total_paid += payout
            hits += 1
            monthly_hits += 1
        else:
            bankroll -= stake
            monthly_pnl -= stake

        total_staked += stake
        placed += 1
        monthly_stake += stake
        monthly_placed += 1

        peak_bankroll = max(peak_bankroll, bankroll)
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        max_dd = max(max_dd, dd)

    # 最終月締め
    if last_month is not None:
        close_month(last_month)

    profit = bankroll - total_deposited
    return {
        "final": bankroll,
        "total_deposited": total_deposited,
        "profit": profit,
        "effective_ratio": bankroll / total_deposited if total_deposited else 0,
        "total_staked": total_staked,
        "total_paid": total_paid,
        "total_rebate": total_rebate,
        "placed": placed,
        "hits": hits,
        "hit_rate": hits / placed if placed else 0,
        "max_dd": max_dd,
        "monthly": pd.DataFrame(monthly_records),
    }


# ============================================================================
# Config プリセット
# ============================================================================

PLACE = "place_top1_ev>1.0"
SANRENPUKU = "sanrenpuku_top1_ev>1.0_p>=0.20"

CONFIGS = {
    "Baseline_flat100": {
        PLACE:      {"mode": "flat", "stake": 100},
        SANRENPUKU: {"mode": "flat", "stake": 100},
    },
    "Flat1000_10x": {
        PLACE:      {"mode": "flat", "stake": 1000},
        SANRENPUKU: {"mode": "flat", "stake": 1000},
    },
    "Conservative_1%/flat100": {
        PLACE:      {"mode": "fraction", "frac": 0.01},
        SANRENPUKU: {"mode": "flat", "stake": 100},
    },
    "Moderate_3%/0.3%": {
        PLACE:      {"mode": "fraction", "frac": 0.03},
        SANRENPUKU: {"mode": "fraction", "frac": 0.003},
    },
    "Aggressive_5%/0.5%": {
        PLACE:      {"mode": "fraction", "frac": 0.05},
        SANRENPUKU: {"mode": "fraction", "frac": 0.005},
    },
    "Kent様提案_10%/10%(危険)": {
        PLACE:      {"mode": "fraction", "frac": 0.10},
        SANRENPUKU: {"mode": "fraction", "frac": 0.10},
    },
    "PlaceOnly_5%": {
        PLACE:      {"mode": "fraction", "frac": 0.05},
        SANRENPUKU: {"mode": "flat", "stake": 0},  # 実質無効化されない → 別処理
    },
}


def format_yen(x: float) -> str:
    return f"¥{int(x):>12,}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bets-log", default="data/backtest/v4_bets_log.parquet")
    p.add_argument("--initial", type=float, default=100_000)
    p.add_argument("--monthly-inject", type=float, default=100_000)
    p.add_argument("--rebate-rate", type=float, default=0.0057)
    args = p.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    df = pd.read_parquet(args.bets_log)
    print(f"=== Composite Sim 入力: {len(df):,} bets ===")
    print(f"  対象戦略: {PLACE}, {SANRENPUKU}")
    print(f"  初期資金: ¥{int(args.initial):,}, 毎月注入: ¥{int(args.monthly_inject):,}")
    print(f"  SPAT4 還元率: {args.rebate_rate*100:.2f}%")
    print()

    # ベース: no-bet baseline (預金のみ)
    n_months = df["month"].nunique()
    no_bet_final = args.initial + args.monthly_inject * (n_months - 1)
    print(f"= 預金のみ baseline (28ヶ月): final ¥{int(no_bet_final):,} =\n")

    rows = []
    for name, strategies in CONFIGS.items():
        # PlaceOnly は sanrenpuku=stake0 だと strategy filter で省略する必要あり
        active_strats = {s: r for s, r in strategies.items()
                         if not (r["mode"] == "flat" and r.get("stake") == 0)}
        res = simulate_composite(
            df, strategies=active_strats,
            initial=args.initial,
            monthly_inject=args.monthly_inject,
            spat4_rebate_rate=args.rebate_rate,
        )
        rows.append({
            "config": name,
            "placed": res["placed"],
            "hits": res["hits"],
            "hit_rate": f"{res['hit_rate']*100:.1f}%",
            "total_staked": format_yen(res["total_staked"]),
            "total_paid": format_yen(res["total_paid"]),
            "rebate": format_yen(res["total_rebate"]),
            "deposited": format_yen(res["total_deposited"]),
            "final": format_yen(res["final"]),
            "profit": f"{'+' if res['profit']>=0 else ''}¥{int(res['profit']):,}",
            "ratio": f"{res['effective_ratio']:.3f}x",
            "max_DD": f"{res['max_dd']*100:.1f}%",
        })

    print(pd.DataFrame(rows).to_string(index=False))

    # 月次推移 (Moderate config)
    print("\n=== Moderate (3%/0.3%) 月次推移 ===")
    res_mod = simulate_composite(
        df, strategies=CONFIGS["Moderate_3%/0.3%"],
        initial=args.initial,
        monthly_inject=args.monthly_inject,
        spat4_rebate_rate=args.rebate_rate,
    )
    monthly = res_mod["monthly"].copy()
    monthly["bankroll_str"] = monthly["end_bankroll"].apply(lambda x: f"¥{x:,}")
    monthly["pnl_str"] = monthly["monthly_pnl"].apply(lambda x: f"{'+' if x>=0 else ''}¥{x:,}")
    print(monthly[["month", "bankroll_str", "monthly_stake", "pnl_str", "rebate", "placed", "hits"]].to_string(index=False))


if __name__ == "__main__":
    main()
