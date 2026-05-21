"""
Market impact + 還元ルール訂正版シミュレーター。

pari-mutuel (ばんえい) の実効オッズ圧縮を考慮する:
    horse_pool = total_pool × (1 - margin) / displayed_odds
    effective_odds = displayed_odds × horse_pool / (horse_pool + our_bet)

SPAT4 還元ルール訂正:
    複勝・単勝は還元対象外 (実運用に合わせる)
    馬連/馬単/ワイド/三連複/三連単 のみ 0.57% 還元

Grid search で資金フラクション (place_frac, sanrenpuku_frac) を振り、
最終資金を最大化する組合せを探す。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ===== モデル定数 =====

POOL_ASSUMPTIONS_YEN = {
    # 帯広ばんえいの典型プール規模 (¥/レース)。要キャリブレーション
    "win":        3_000_000,
    "place":      3_000_000,
    "umaren":    10_000_000,
    "umatan":     3_000_000,
    "wide":       3_000_000,
    "sanrenpuku":20_000_000,
    "sanrentan": 20_000_000,
}

MARGINS = {
    "win":        0.20,
    "place":      0.20,
    "umaren":     0.225,
    "umatan":     0.225,
    "wide":       0.225,
    "sanrenpuku": 0.275,
    "sanrentan": 0.275,
}

# SPAT4 還元対象 (単勝・複勝は対象外)
REBATE_ELIGIBLE = {"umaren", "umatan", "wide", "sanrenpuku", "sanrentan"}
SPAT4_REBATE_RATE = 0.0057


def effective_odds(displayed: float, bet: float, pool: float, margin: float) -> float:
    """pari-mutuel 実効オッズ。プールに対する自己ベットの圧縮を反映。"""
    if displayed <= 0 or pool <= 0:
        return displayed
    horse_pool = pool * (1.0 - margin) / displayed
    if horse_pool <= 0:
        return displayed
    return displayed * horse_pool / (horse_pool + bet)


def apply_bet(
    bankroll: float, bet_type: str, displayed_odds: float, hit: bool, payout_per_100yen: float,
    *, stake: float, use_impact: bool, pool_override: float | None = None,
) -> tuple[float, float, float]:
    """1ベットを処理。戻り値: (new_bankroll, effective_payout, rebate)

    payout_per_100yen は実際の払戻 (¥100 ステーク時) を表す。
    複勝の場合は actual place_payout、組合せ系では odds_min * 100。
    impact 計算では payout_per_100yen / 100 を「displayed odds」として使い、
    pari-mutuel プール影響を係数として乗算する。
    """
    rebate = stake * SPAT4_REBATE_RATE if bet_type in REBATE_ELIGIBLE else 0.0
    if not hit:
        return bankroll - stake + rebate, 0.0, rebate
    base_odds = payout_per_100yen / 100.0  # 実 payout 倍率
    if not use_impact:
        payout = stake * base_odds
    else:
        pool = pool_override if pool_override is not None else POOL_ASSUMPTIONS_YEN.get(bet_type, 5_000_000)
        margin = MARGINS.get(bet_type, 0.25)
        eff_odds = effective_odds(base_odds, stake, pool, margin)
        payout = stake * eff_odds
    return bankroll - stake + payout + rebate, payout, rebate


def simulate(
    bets_df: pd.DataFrame,
    *,
    strategies: dict[str, dict],
    initial: float = 100_000,
    monthly_inject: float = 100_000,
    use_impact: bool = True,
    min_stake: float = 100.0,
) -> dict:
    """戦略別 sizing rule を持つ複合シミュレーション。"""
    df = bets_df[bets_df["strategy"].isin(strategies)].copy()
    df = df.sort_values(["month", "race_id", "strategy"]).reset_index(drop=True)

    bankroll = initial
    total_deposited = initial
    peak = initial
    max_dd = 0.0

    placed = hits = 0
    total_staked = total_paid = total_rebate = 0.0

    last_month = None
    monthly_history: list[dict] = []
    m_stake = m_pnl = 0.0
    m_placed = m_hits = 0

    def close_month(month):
        nonlocal m_stake, m_pnl, m_placed, m_hits, peak, max_dd
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        monthly_history.append({
            "month": month,
            "bankroll_end": int(bankroll),
            "stake": int(m_stake),
            "pnl": int(m_pnl),
            "placed": m_placed,
            "hits": m_hits,
        })
        m_stake = m_pnl = 0.0
        m_placed = m_hits = 0

    for _, bet in df.iterrows():
        if last_month is None:
            last_month = bet["month"]
        elif bet["month"] != last_month:
            close_month(last_month)
            bankroll += monthly_inject
            total_deposited += monthly_inject
            last_month = bet["month"]

        rule = strategies[bet["strategy"]]
        if rule["mode"] == "flat":
            stake = float(rule["stake"])
        else:
            stake = max(min_stake, rule["frac"] * bankroll)
        # 絶対額キャップ (market impact 抑制)
        if rule.get("max_stake") is not None:
            stake = min(stake, float(rule["max_stake"]))
        stake = min(stake, bankroll)
        if stake < min_stake:
            continue

        bankroll, payout, rebate = apply_bet(
            bankroll, bet["bet_type"], float(bet["odds"]),
            bool(bet["hit"]), float(bet["payout_per_100yen"]),
            stake=stake, use_impact=use_impact,
        )

        if bool(bet["hit"]):
            hits += 1
            m_hits += 1
            total_paid += payout
        m_pnl += (payout - stake) + rebate
        total_staked += stake
        total_rebate += rebate
        placed += 1
        m_placed += 1
        m_stake += stake
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    if last_month is not None:
        close_month(last_month)

    return {
        "final": bankroll,
        "deposited": total_deposited,
        "profit": bankroll - total_deposited,
        "ratio": bankroll / total_deposited if total_deposited else 0,
        "staked": total_staked,
        "paid": total_paid,
        "rebate": total_rebate,
        "placed": placed,
        "hits": hits,
        "max_dd": max_dd,
        "monthly": pd.DataFrame(monthly_history),
    }


PLACE = "place_top1_ev>1.0"
SANRENPUKU = "sanrenpuku_top1_ev>1.0_p>=0.20"


def grid_search(
    bets_df: pd.DataFrame,
    *,
    place_fracs: list[float],
    san_fracs: list[float],
    initial: float = 100_000,
    monthly_inject: float = 100_000,
    use_impact: bool = True,
    place_cap: float | None = None,
    san_cap: float | None = None,
) -> pd.DataFrame:
    """Grid: (place_frac, sanrenpuku_frac) → 最終資金・DD"""
    rows = []
    for pf in place_fracs:
        for sf in san_fracs:
            strategies = {}
            if pf > 0:
                strategies[PLACE] = {"mode": "fraction", "frac": pf, "max_stake": place_cap}
            if sf > 0:
                strategies[SANRENPUKU] = {"mode": "fraction", "frac": sf, "max_stake": san_cap}
            if not strategies:
                continue
            res = simulate(
                bets_df, strategies=strategies,
                initial=initial, monthly_inject=monthly_inject,
                use_impact=use_impact,
            )
            rows.append({
                "place_frac": pf,
                "san_frac": sf,
                "place_cap": place_cap,
                "san_cap": san_cap,
                "final": int(res["final"]),
                "profit": int(res["profit"]),
                "ratio": round(res["ratio"], 3),
                "max_dd": round(res["max_dd"] * 100, 1),
                "staked": int(res["staked"]),
                "rebate": int(res["rebate"]),
            })
    return pd.DataFrame(rows)


def grid_search_with_caps(
    bets_df: pd.DataFrame,
    *,
    place_fracs: list[float],
    san_fracs: list[float],
    place_caps: list[float | None],
    san_caps: list[float | None],
    initial: float = 100_000,
    monthly_inject: float = 100_000,
    use_impact: bool = True,
) -> pd.DataFrame:
    """(frac, cap) 2軸 grid."""
    rows = []
    for pc in place_caps:
        for sc in san_caps:
            sub = grid_search(
                bets_df,
                place_fracs=place_fracs, san_fracs=san_fracs,
                initial=initial, monthly_inject=monthly_inject,
                use_impact=use_impact,
                place_cap=pc, san_cap=sc,
            )
            rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def fmt_yen(x):
    return f"¥{int(x):>13,}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bets-log", default="data/backtest/v4_bets_log.parquet")
    p.add_argument("--initial", type=float, default=100_000)
    p.add_argument("--monthly-inject", type=float, default=100_000)
    p.add_argument("--no-impact", action="store_true", help="market impact を無効化 (比較用)")
    args = p.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    df = pd.read_parquet(args.bets_log)
    print(f"=== Market Impact Sim: {len(df):,} bets ===")
    print(f"  対象: {PLACE}, {SANRENPUKU}")
    print(f"  初期 ¥{int(args.initial):,} / 毎月注入 ¥{int(args.monthly_inject):,}")
    print(f"  Market impact: {'OFF (理論値)' if args.no_impact else 'ON (現実値)'}")
    print(f"  SPAT4: 馬連/馬単/ワイド/三連複/三連単のみ 0.57%還元 (単複は対象外)")
    print(f"  プール想定 (¥/レース): 複勝 {POOL_ASSUMPTIONS_YEN['place']:,}, 三連複 {POOL_ASSUMPTIONS_YEN['sanrenpuku']:,}")
    print()

    # 絶対額キャップを軸に grid search
    fracs_place = [0.01, 0.02, 0.03, 0.05, 0.10]
    fracs_san   = [0.0, 0.005, 0.01, 0.02]
    place_caps  = [None, 5_000, 10_000, 20_000, 30_000, 50_000, 100_000]
    san_caps    = [None, 3_000, 5_000, 10_000, 30_000]

    print("=== Grid: 絶対額キャップ × フラクション (impact ON) ===")
    grid_caps = grid_search_with_caps(
        df,
        place_fracs=fracs_place, san_fracs=fracs_san,
        place_caps=place_caps, san_caps=san_caps,
        initial=args.initial, monthly_inject=args.monthly_inject,
        use_impact=True,
    )
    grid_top = grid_caps.sort_values("final", ascending=False).head(20)
    print("Top 20 (final 資金順):")
    print(grid_top.to_string(index=False))

    # Place 単独 (san_frac=0) で cap × frac の関係を見る
    print("\n=== Place 単独 (san_frac=0): cap × frac 効果 ===")
    place_only = grid_caps[grid_caps["san_frac"] == 0].copy()
    pivot = place_only.pivot_table(
        index="place_cap", columns="place_frac", values="profit", aggfunc="first"
    )
    pivot.index = pivot.index.map(lambda x: "no_cap" if x is None or pd.isna(x) else f"¥{int(x):,}")
    pivot.columns = [f"{c*100:.0f}%" for c in pivot.columns]
    print("(セル = profit, 単位 ¥)")
    print(pivot.round(0).to_string())

    # Best 抽出
    best = grid_caps.loc[grid_caps["final"].idxmax()]
    print(f"\n=== 最適設定 (impact ON) ===")
    print(f"place_frac = {best['place_frac']*100:.1f}%, place_cap = {best['place_cap']}")
    print(f"san_frac   = {best['san_frac']*100:.1f}%, san_cap   = {best['san_cap']}")
    print(f"final = ¥{best['final']:,}, profit = ¥{best['profit']:,}, ratio = {best['ratio']}, max_DD = {best['max_dd']}%")


if __name__ == "__main__":
    main()
