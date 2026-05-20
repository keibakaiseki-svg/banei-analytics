"""
Plackett-Luce 近似による組合せ確率計算。

LightGBM の二項分類器 (target_win) 出力 `pred_p_win` を「各馬の Plackett-Luce
スコア」とみなし、組合せ券種の確率を導出する。

仮定:
    P(rank_1 = i) = p_i / Σp_k                    (sum=1 に正規化)
    P(rank_1 = i, rank_2 = j) = p_i * p_j / (1 - p_i)
    P(rank_1 = i, rank_2 = j, rank_3 = k) = p_i * p_j / (1 - p_i) * p_k / (1 - p_i - p_j)

これは「最強の馬から順に決まる」モデル。実際は同枠相関等の効果があるため過小評価
気味だが、券種選択の比較では十分有用。

使用例:
    from analytics.plackett_luce import combo_probs_for_race
    probs = combo_probs_for_race(
        horse_nos=[1, 2, 3, 4, 5, 6, 7, 8],
        scores=[0.30, 0.20, 0.15, 0.12, 0.08, 0.07, 0.05, 0.03],
        bet_type="sanrentan",
    )
    # → {'1-2-3': 0.0234, '1-2-4': 0.0187, ...}
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Iterable, Optional

import numpy as np

EPS = 1e-12


def normalize(scores: Iterable[float]) -> np.ndarray:
    s = np.asarray(list(scores), dtype=float)
    total = s.sum()
    if total <= 0:
        return np.full(len(s), 1.0 / max(len(s), 1))
    return s / total


def prob_ordered(p_norm: np.ndarray, indices: tuple[int, ...]) -> float:
    """P(順位1=indices[0], 順位2=indices[1], ...) under Plackett-Luce.
    `p_norm` は sum=1 の正規化済スコア。"""
    if len(set(indices)) != len(indices):
        return 0.0
    used_sum = 0.0
    prob = 1.0
    for idx in indices:
        denom = 1.0 - used_sum
        if denom <= EPS:
            return 0.0
        prob *= p_norm[idx] / denom
        used_sum += p_norm[idx]
    return prob


def prob_unordered_top_k(p_norm: np.ndarray, indices: tuple[int, ...]) -> float:
    """P({indices} が同時に上位 len(indices) 位以内に入る・順序不問)。
    指定集合の全順列の prob_ordered の和。"""
    return sum(prob_ordered(p_norm, perm) for perm in permutations(indices))


def prob_pair_in_top_k(p_norm: np.ndarray, i: int, j: int, k: int) -> float:
    """P(i ∈ top k AND j ∈ top k)。ワイド計算用。
    i, j を「k 個の上位位置中に並ぶ」全パターンの和。"""
    if i == j:
        return 0.0
    n = len(p_norm)
    if k == 2:
        # 自分たちだけで top2 を独占する確率
        return prob_unordered_top_k(p_norm, (i, j))
    # k=3 (or larger): i,j に加えて任意の他馬 c が top k に入るパターン
    total = 0.0
    others = [c for c in range(n) if c != i and c != j]
    fillers_size = k - 2
    for fillers in combinations(others, fillers_size):
        members = (i, j, *fillers)
        # k 個全員の順列ぶん和を取る
        total += sum(prob_ordered(p_norm, perm) for perm in permutations(members))
    return total


# ============================================================================
# 馬番ベースの公開API
# ============================================================================

# 券種ごとの設定
BET_TYPE_META: dict[str, dict] = {
    "umaren":     {"size": 2, "ordered": False, "kind": "top_k_equal"},
    "umatan":     {"size": 2, "ordered": True,  "kind": "top_k_equal"},
    "wide":       {"size": 2, "ordered": False, "kind": "pair_in_top_k"},
    "sanrenpuku": {"size": 3, "ordered": False, "kind": "top_k_equal"},
    "sanrentan":  {"size": 3, "ordered": True,  "kind": "top_k_equal"},
}


def _format_combo(horse_nos: list[int], indices: tuple[int, ...], ordered: bool) -> str:
    hs = [horse_nos[i] for i in indices]
    if not ordered:
        hs = sorted(hs)
    return "-".join(str(h) for h in hs)


def combo_probs_for_race(
    horse_nos: list[int],
    scores: Iterable[float],
    bet_type: str,
    *,
    wide_top_k: Optional[int] = None,
) -> dict[str, float]:
    """1レース分の組合せ確率を計算。

    Args:
        horse_nos: 馬番リスト (整数)
        scores: 各馬のスコア (順序は horse_nos と一致, 例: pred_p_win)
        bet_type: "umaren"/"umatan"/"wide"/"sanrenpuku"/"sanrentan"
        wide_top_k: ワイドの「複勝範囲」 (8+: 3, 5-7: 2)。bet_type=wide で必須。

    Returns:
        {combination_str: prob}
    """
    if bet_type not in BET_TYPE_META:
        raise ValueError(f"unknown bet_type: {bet_type}")
    meta = BET_TYPE_META[bet_type]
    size = meta["size"]
    ordered = meta["ordered"]
    p_norm = normalize(scores)
    n = len(horse_nos)
    if n < size:
        return {}

    result: dict[str, float] = {}
    iter_fn = permutations if ordered else combinations
    for idx_combo in iter_fn(range(n), size):
        if meta["kind"] == "top_k_equal":
            if ordered:
                prob = prob_ordered(p_norm, idx_combo)
            else:
                prob = prob_unordered_top_k(p_norm, idx_combo)
        elif meta["kind"] == "pair_in_top_k":
            if wide_top_k is None:
                raise ValueError("wide_top_k must be specified for bet_type=wide")
            prob = prob_pair_in_top_k(p_norm, idx_combo[0], idx_combo[1], wide_top_k)
        else:
            raise ValueError(f"unknown kind: {meta['kind']}")
        combo_str = _format_combo(horse_nos, idx_combo, ordered=ordered)
        result[combo_str] = prob
    return result


def wide_top_k_from_entry(entry_count: int) -> Optional[int]:
    """ばんえいワイドのpayout対象 top_k。
    8+: top 3 / 5-7: top 2 / 4-: なし
    (一般的なJRA/NAR規程に準拠・要事実確認)
    """
    if entry_count >= 8:
        return 3
    if entry_count >= 5:
        return 2
    return None
