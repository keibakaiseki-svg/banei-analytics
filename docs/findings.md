# Banei Analytics: 発見と検証ロードマップ

最終更新: 2026-05-24
分析期間: 2019-10-01 〜 2026-04-30 (79ヶ月, 11,414レース, 4券種=馬連/ワイド/馬単/三連複)

---

## 1. 戦略パフォーマンス (全期間ベース)

### 最終ベースライン (water filter 適用後)

| 戦略 | bets | hit rate | ROI | 評価 |
|---|---|---|---|---|
| **place_top1_ev>1.0_water_filt** | 738 | 62.1% | **1.059** | 安定収益 |
| **sanrenpuku_top1_p>=0.20_water_filt** | 2,057 | 13.7% | **1.090** | 中分散・高ROI |
| place_top1_raw (フィルタなし) | 1,283 | 59.4% | 1.015 | |
| sanrenpuku_p20_raw | 2,563 | 13.8% | 1.077 | |
| win_top1_ev>1.0 | 1,283 | 28.5% | 0.899 | 不採用 |
| umatan_p>=0.10 | 1,645 | 6.5% | 0.886 | 不採用 |
| umaren_p>=0.10 | 3,135 | 10.1% | 0.882 | 不採用 |

### 最終推奨運用設定

```
資金: ¥100K 初期 + 月¥100K 注入
戦略 A: place_top1
    - water bin ∈ {light(1-2%), wet(3-4%)} のみ
    - frac 5% × bankroll, cap ¥30K
戦略 B: sanrenpuku_p20
    - water bin ∈ {light(1-2%), normal(2-3%)} のみ
    - frac 2% × bankroll, cap ¥10K
期待 (79ヶ月 simulation):
    - 終資金 ¥10.5M (deposit ¥8M, ratio 1.328)
    - 月平均 profit +¥33K
    - 最大DD 36%
    - SPAT4還元 +¥240K (三連複のみ対象、複勝は対象外)
```

---

## 2. 重要な発見 (傾向)

### 2.1 水分量 (track_water_pct) パターン

| 水分量 | 本命勝率 | 荒れ度 | place ROI | san ROI | 戦略 |
|---|---|---|---|---|---|
| dry (<1.0) | 38.3% | 低 | 0.945 ❌ | 1.029 △ | **罠** (本命強くオッズ低) |
| **light (1-2)** | 35.7% | 中 | **1.053** ✓ | **1.108** ✓ | 両戦略OK |
| normal (2-3) | 35.2% | 中 | 0.961 | **1.233** ★ | sanrenpuku 強い |
| **wet (3-4)** | 34.0% | 高 | **1.090** ★ | 0.988 | place 強い |
| heavy (4+) | 29.7% | 高 | 0.743 | n/a | **要警戒** (荒れすぎ) |

**含意**:
- dry は罠 (本命勝率高いが配当が見合わない)
- 戦略ごとに「美味しい水分量」が異なる → meta-filter 有効
- 水分量増 → 荒れ度増 (本命勝率 38→30%)

### 2.2 古馬重賞前後の「叩き」効果

**5/6 古馬重賞で確認** (Kent様仮説の決定的証拠):

| 重賞 | 月 | san 前 1-3日 ROI | place 前 4-7日 ROI |
|---|---|---|---|
| ばんえい記念 | 3月 | 0.850 | 0.962 |
| **帯広記念** | 1月 | **0.212** ❌❌ | **0.340** ❌❌ |
| **ヒロインズ** | 2月 | **0.000** ❌❌ | 0.883 |
| **北見記念** | 10月 | **0.464** ❌ | **0.615** ❌ |
| **岩見沢記念** | 9月 | **0.000** ❌❌ | 1.043 |
| **旭川記念 (例外)** | 7月 | **2.150** ★ | 1.700 ★ |

**含意**:
- 重賞前1-7日は主力馬が「叩き」入れて本気出さない → 本命の予測精度低下
- san は特に前1-3日に直撃 (組合せ系のため)
- 旭川記念 (夏季ナイター期) のみ逆 (周辺レースが好調)

### 2.3 若馬重賞は「お祭り効果」(古馬と真逆)

| 重賞 | 月 | san 前 1-3日 ROI | 評価 |
|---|---|---|---|
| **ナナカマド賞** | 10月 | **2.779** ⭐⭐ | 大化け |
| **プリンセス賞** | 1月 | **2.938** ⭐⭐⭐ | 超大化け |
| **ばんえいオークス** | 12月 | **2.489** ⭐⭐ | 大化け |
| ばんえい大賞典 | 8月 | 1.333 | 好調 |
| ばんえいダービー | 12月 | 1.400 ★ | 好調 |
| 黒ユリ賞 | 2月 | 1.314 | 好調 |
| イレネー記念 | 3月 | 0.814 | 弱 |
| とかち皐月賞 | 5月 | 0.681 ❌ | 5月不振 |
| **銀河賞** | 9月 | **0.000** ❌❌ | **9月不振 part 2** |

**含意**:
- 若馬は出走頻度低 → 叩き需要なし、主力馬が普通に走る
- 重賞前で人気と実力にギャップ生じやすい → モデル edge 大
- 銀河賞は例外 (4歳重賞だが古馬戦並みに荒れる)

### 2.4 9月不振の犯人特定

**9月 sanrenpuku ROI 0.479 の主因**:
1. **岩見沢記念** (9月中下旬・古馬重賞): 前14日全部 san 大失速
2. **銀河賞** (9月下旬・4歳重賞): 前14日 ROI 0.14-0.44
- 9月は **重賞のダブルアタック** で前半・後半とも san が罠

### 2.5 馬齢×性別パターン

| 性別 | 年齢 | place ROI | 評価 |
|---|---|---|---|
| **牝** | **2** | **1.326** ⭐⭐ | **成長期最強** |
| 牝 | 3 | 1.031 | 普通 |
| 牝 | 4 | 0.947 | 弱 |
| 牡 | 2 | 0.915 ❌ | 牡2歳は弱 |
| 牡 | 3 | 1.012 | 普通 |
| **牡** | **4** | **1.153** ★ | 完成期 |
| 牡 | 5-6 | 1.0-1.1 | 維持期 |
| **全** | **6** | **1.254** ★ | **ベテラン** |

**含意**:
- 牝馬は 2歳がピーク (成長カーブ早い)
- 牡馬は 4歳がピーク
- 6歳全般 (経験馬) が安定して高 ROI
- 体重微増 (+1〜+10kg) は順調成長サイン (ROI 1.10+)

### 2.6 popularity 偏重問題

| Variant | place ROI | place 穴率 (odds≥2.0) | san ROI |
|---|---|---|---|
| current (raw) | 1.086 | 5% (本命寄り) | 0.954 (16mo) / 1.090 (79mo) |
| **drop** | 1.027 | **68%** (穴狙い) | 0.908 (79mo) |
| conditional (中和) | 0.959 | 50% | 0.709 |

**含意**:
- popularity SHAP 0.827 = 他特徴量の **16倍** 偏重
- drop variant: 穴買い特化として独立に機能 (ROI 1.027)
- san は drop だと PL確率分散して機能不全 → current 必須
- **place は ensemble (current + drop) で穴本命併用余地あり**

### 2.7 騎手特徴量の威力 ⭐ (本セッション最大の発見)

| Variant | place ROI | san ROI |
|---|---|---|
| Baseline (騎手なし) | 1.086 | 0.954 |
| **+ Jockey features (7本)** | **1.108** (+2.2pp) | **1.202** (+24.8pp) ⭐⭐⭐ |

**重要な騎手特徴量** (gain 順):
- `jk_career_top3_rate` (騎手通算複勝率) → 全特徴中 **6位**
- `jk_career_win_rate` (騎手通算勝率) → 11位
- `jk_trainer_pair_win_rate` (**主戦コンビ**) → 13位 ⭐ Kent様仮説確認
- `jk_horse_pair_rides` (騎乗回数 = 主戦騎手検出)
- `jk_horse_pair_win_rate` (馬×騎手勝率)
- `jk_recent_win_rate` (直近100戦勝率)
- `is_main_jockey_for_horse` (binary) → 連続特徴量に置き換わり 使用されず

**主戦コンビ実例 (累計騎乗 win rate 平均13%)**:
- 阿部武×坂本東: 16.5% ★
- 藤本匠×松井浩: 15.2% ★
- 西謙一×西弘美: 13.3%
- 鈴木恵: 単独で 17.6% (突出した個人能力)

### 2.8 年別 ROI 推移 (regression to mean)

| Year | place_filt ROI | san_filt ROI | 評価 |
|---|---|---|---|
| 2019 Q4 | 0.796 | 1.519 ⭐ | 異常良好 |
| 2020 | 0.950 | 1.379 ⭐ | 良好 |
| 2021 | 1.134 ✓ | 0.908 | place 強い時期 |
| 2022 | 1.034 ✓ | 1.072 | 普通 |
| 2023 | 1.046 ✓ | 0.984 | break-even |
| 2024 | 1.081 ✓ | 0.996 | place 強い |
| 2025 | 1.185 ✓ | 1.090 | 両方良好 |
| **2026 Jan-Apr** | **0.926** ❌ | **0.780** ❌ | **要警戒** |

**含意**:
- 2019-2020 高ROI は再現性低い (regression)
- 2021-2025 で安定 (place 1.03-1.18, san 0.91-1.09)
- **2026 不調は警戒シグナル** (実運用初期で同様なら戦略停止検討)

### 2.9 市場影響 (pari-mutuel)

- 複勝プール想定 ¥3M, 三連複 ¥20M
- bet 額が プール の 5% を超えると odds 大幅圧縮
- 最適配分: place 5% + cap ¥30K, san 2% + cap ¥10K
- フラクションのみだと資金成長で自己破壊 → **絶対額キャップ必須**
- SPAT4 還元: 三連複/三連単 系のみ 0.57% (複勝・単勝は対象外)

---

## 3. 試行錯誤の記録 (失敗・没案)

| 試行 | 結果 | 学び |
|---|---|---|
| 広め買い (box top3-5) | ROI 0.73-0.85 全て損失 | bookmaker margin に拡散で勝てない |
| 馬連流し anti-fav | ROI 0.905 (改善するが負け) | edge 出すには更なる filter 必要 |
| popularity conditional (中和) | place 0.959 / san 0.709 (悪化) | feature 削除より重み付け改良が必要 |
| Kent様提案 10%/10% (両戦略) | profit -¥2.78M (破産) | 高分散戦略には Kelly フラクションが要 |

---

## 4. 今後すべき検証 (Roadmap)

### Phase 7 (即実装可能・高優先度)

1. **騎手特徴量の本実装 (v8)** ⭐⭐⭐
   - features.py に SQL window 統合
   - 79ヶ月 walk-forward で再ベンチ
   - 期待: sanrenpuku ROI 1.09 → **1.18-1.22**, compound +¥1M profit改善
   - 工数: 2-3時間

2. **重賞前7日スキップ filter 統合 (v8)**
   - 古馬重賞 (5種, 旭川除く) 前7日 全戦略skip
   - 9月の銀河賞前 san 特別 skip
   - 期待: 月¥10K profit 改善
   - 工数: 1時間

3. **若馬重賞前 san 強化フィルタ**
   - プリンセス・ナナカマド・オークス前 san を倍プッシュ
   - 工数: 1時間

### Phase 8 (中優先度)

4. **三連単データ取得 + 全7券種 EV比較** ⭐
   - Colab で 38時間 (4セッション) ← **進行中**
   - 完了後: walk_forward_v9 で全7券種 EV 比較
   - **「人気薄1着 三連単」(anti-favorite) 戦略** 検証
     - 仮説: 三連単は bet単位小→市場影響なし、人気薄一着は大穴で高EV
     - reference: [banei-anti-fav-sanrentan](banei_anti_fav_sanrentan.md)

5. **ensemble 戦略 (place: current + drop)**
   - 両モデルが同馬選択 → 高信頼大額bet
   - 別馬選択 → 穴本命併用
   - 工数: 半日

6. **成長指標特徴量** (馬の成長カーブ捕捉)
   - age × sex 交互作用 (2歳牝馬独立扱い)
   - SF trend (改善傾向の slope)
   - 連続体重増加カウント
   - 工数: 2-3時間

### Phase 9 (低優先度)

7. **走行コース実装** (内詰め/外詰め)
   - 開催日×R番号で交互。馬番固定でゲート位置のみ変更
   - 物理 lane を特徴量化 → 内/外有利の検定
   - reference: [banei-lane-fill-rule](banei_lane_fill_rule.md)

8. **実プールデータ取得**
   - netkeiba 払戻ページから「発売金額」スクレイプ
   - 現状の market_impact_sim 想定値を実値で校正
   - 工数: 半日

9. **モデル calibration** (longshot 過大評価補正)
   - isotonic regression / Platt scaling
   - EV計算の信頼性 UP
   - 工数: 1-2時間

10. **2026 年 不調 原因深掘り**
    - 競争激化? レジーム変化? 偶然?
    - 実運用 3ヶ月で edge 出ない場合の判定基準

### Phase 10 (運用)

11. **実マネー 試験運用**
    - Phase 1: Conservative 1% で 3ヶ月 (edge 確認)
    - Phase 2: Moderate 3% で 1年 (拡大)
    - Phase 3: Aggressive 5% (PlaceOnly or 複合) で本格運用
    - 月次 DD 監視 + 動的サイジング

12. **GitHub Actions 朝バッチ + Streamlit dashboard**
    - 当日推奨自動生成
    - 月次 ROI ダッシュボード
    - Slack/メール通知

---

## 5. 構築済みインフラ (再利用可能)

### スクレイパ
- `scrapers/netkeiba_combo_odds.py` (5券種, AJAX+jiku対応)
- `scripts/scrape_netkeiba_combo_odds.py` (auto-push retry付き)
- `notebooks/07_netkeiba_combo_odds_colab.ipynb` (Colab driver)

### 解析
- `analytics/walk_forward_v4.py` (確率フィルタ + ベットログ)
- `analytics/walk_forward_v5.py` (広め買い: 検証済 没)
- `analytics/walk_forward_v6.py` (water filter)
- `analytics/walk_forward_v7_drop.py` (popularity drop)
- `analytics/plackett_luce.py` (組合せ確率)

### 戦略シミュレーション
- `analytics/bankroll_sim.py` (Kelly比較)
- `analytics/composite_sim.py` (place+san並走)
- `analytics/market_impact_sim.py` (pari-mutuel + SPAT4)
- `analytics/popularity_experiment.py` (popularity 3変種比較)
- `analytics/jockey_experiment.py` (騎手特徴量実験)

### データ
- `data/parquet/odds_netkeiba_{umaren,wide,umatan,sanrenpuku}.parquet` (各 11,414レース)
- `data/parquet/odds_netkeiba.parquet` (単勝・複勝)
- `data/backtest/v*_bets_log.parquet` (戦略別ベット履歴)

---

## 6. メモリ参照

- [banei-lane-fill-rule.md](.claude/projects/-Users-keto-ito/memory/banei_lane_fill_rule.md) - 走行コース割当ルール
- [banei-anti-fav-sanrentan.md](.claude/projects/-Users-keto-ito/memory/banei_anti_fav_sanrentan.md) - 人気薄1着三連単戦略

---

**🎯 即時アクション**: Colab で三連単データ取得開始 + Phase 7 (騎手特徴量実装) を並行進行。
