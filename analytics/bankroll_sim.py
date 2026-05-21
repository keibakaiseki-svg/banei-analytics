"""
資金管理シミュレーター。

walk_forward_v4 が `--bets-log-path` で出力したベット履歴を読み込み、
資金管理ルール別の終資金・最大ドローダウン等を比較する。

入力 schema:
    strategy, month, race_id, bet_type, combination, prob, ev, odds, hit, payout_per_100yen

設定:
- Baseline: flat ¥100 stake, no filter
- A: flat ¥100 stake, odds≥1.5
- B: 10% bankroll stake (動的), odds≥1.5  ← Kent様提案
- C: 10% bankroll stake, no filter (純粋複利効果のみ)
- D: 5% bankroll stake, no filter (safer)

使用例:
    uv run python -m analytics.bankroll_sim
    uv run python -m analytics.bankroll_sim --strategies place_top1_ev>1.0,sanrenpuku_top1_ev>1.0_p>=0.20
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def simulate(
    bets: pd.DataFrame,
    *,
    initial: float = 10000.0,
    stake_mode: str = "flat",          # "flat" | "fraction"
    flat_stake: float = 100.0,
    stake_frac: float = 0.10,
    min_stake: float = 100.0,
    min_odds: Optional[float] = None,
    min_ev: Optional[float] = None,
    max_stake_cap: Optional[float] = None,
) -> dict:
    """1戦略の bet 列に対して逐次シミュレーション。

    Returns:
        final, placed, hits, total_staked, total_paid, roi, growth, max_drawdown, history
    """
    bankroll = initial
    history = [bankroll]
    placed = 0
    hits = 0
    total_staked = 0.0
    total_paid = 0.0
    peak = initial
    max_dd = 0.0

    for _, b in bets.iterrows():
        odds = float(b["odds"])
        if min_odds is not None and odds < min_odds:
            continue
        if min_ev is not None and float(b["ev"]) < min_ev:
            continue

        if stake_mode == "flat":
            stake = flat_stake
        else:
            stake = max(min_stake, stake_frac * bankroll)
        if max_stake_cap is not None:
            stake = min(stake, max_stake_cap)
        stake = min(stake, bankroll)
        if stake < min_stake:
            # 破産: これ以上ベット不能
            break

        if bool(b["hit"]):
            # payout_per_100yen はステーク 100円 ベース → 線形スケール
            payout = stake * (float(b["payout_per_100yen"]) / 100.0)
            bankroll = bankroll - stake + payout
            total_paid += payout
            hits += 1
        else:
            bankroll -= stake
        total_staked += stake
        placed += 1

        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        history.append(bankroll)

    return {
        "final": bankroll,
        "placed": placed,
        "hits": hits,
        "total_staked": total_staked,
        "total_paid": total_paid,
        "roi": total_paid / total_staked if total_staked > 0 else 0.0,
        "growth": bankroll / initial,
        "max_drawdown": max_dd,
        "history": history,
    }


CONFIGS = [
    ("Baseline: flat¥100 / no filter",
     {"stake_mode": "flat", "flat_stake": 100}),
    ("A: flat¥100 / odds≥1.5",
     {"stake_mode": "flat", "flat_stake": 100, "min_odds": 1.5}),
    ("B: 10%資金 / odds≥1.5  ★Kent様提案",
     {"stake_mode": "fraction", "stake_frac": 0.10, "min_odds": 1.5}),
    ("C: 10%資金 / no filter",
     {"stake_mode": "fraction", "stake_frac": 0.10}),
    ("D:  5%資金 / no filter (safer)",
     {"stake_mode": "fraction", "stake_frac": 0.05}),
    ("E:  2%資金 / no filter (Kelly代用)",
     {"stake_mode": "fraction", "stake_frac": 0.02}),
]


def format_yen(x: float) -> str:
    return f"¥{int(x):>10,}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bets-log", default="data/backtest/v4_bets_log.parquet")
    p.add_argument("--strategies", default=None,
                   help="カンマ区切り。未指定で 'Top 5 ROI 戦略' を自動選択")
    p.add_argument("--initial", type=float, default=10000.0)
    p.add_argument("--top-n", type=int, default=5, help="自動選択時のTopN")
    args = p.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    bets_log = Path(args.bets_log)
    if not bets_log.exists():
        print(f"ERROR: ベットログがありません: {bets_log}")
        print("先に: uv run python -m analytics.walk_forward_v4 --bets-log-path data/backtest/v4_bets_log.parquet")
        return

    df = pd.read_parquet(bets_log)
    print(f"=== Bets log: {len(df):,} bets across {df['strategy'].nunique()} strategies ===")
    df = df.sort_values(["month", "race_id"]).reset_index(drop=True)

    # 戦略選択
    if args.strategies:
        target = [s.strip() for s in args.strategies.split(",")]
    else:
        # 全戦略の "no-management ROI" でランキング → top N
        agg = df.groupby("strategy").apply(
            lambda g: g["payout_per_100yen"].sum() / (100 * len(g))
        ).sort_values(ascending=False)
        target = agg.head(args.top_n).index.tolist()
        print(f"\nTop {args.top_n} 戦略 (flat¥100 ROI 順):")
        for s in target:
            print(f"  {s}: ROI={agg.loc[s]:.3f}")

    print()
    for strategy in target:
        sub = df[df["strategy"] == strategy]
        if sub.empty:
            print(f"!! 戦略 {strategy} が見つかりません")
            continue

        print(f"\n{'='*100}")
        print(f"戦略: {strategy}  (bets={len(sub):,}  hit_rate={sub['hit'].mean():.3f})")
        print(f"{'='*100}")

        rows = []
        for label, kwargs in CONFIGS:
            res = simulate(sub, initial=args.initial, **kwargs)
            rows.append({
                "config": label,
                "placed": res["placed"],
                "hits": res["hits"],
                "win%": f"{(res['hits']/res['placed']*100 if res['placed'] else 0):.1f}",
                "staked": format_yen(res["total_staked"]),
                "paid": format_yen(res["total_paid"]),
                "ROI": f"{res['roi']:.3f}",
                "final": format_yen(res["final"]),
                "growth": f"{res['growth']:.2f}x",
                "max_DD": f"{res['max_drawdown']*100:.1f}%",
            })
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
