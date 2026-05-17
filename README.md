# banei-analytics

ばんえい競馬（帯広競馬場）の予想・分析基盤。
回収率最大化を目的とした提案型アシスタントツールを目指す。

## ステータス

**Phase 3: マスタテーブル派生 + EDA基盤 — 完了**
(Phase 2 のフルバックフィル実行と並行して進められる構成)

## ゴール（最終形）

- 朝1回 + レース直前のバッチで当日全レースを取得・予測
- 馬場パターン分類による予想モデルの切替（当日序盤レース結果からベイズ更新）
- Streamlit ダッシュボードで購入判断補助（モデル勝率・損益分岐オッズ・推奨買い目）
- 完全無料運用（GitHub Actions + Streamlit Community Cloud + Google Colab）

## アーキテクチャ

```
GitHub Actions (朝バッチ)
  └→ スクレイピング → Parquet追記 → git commit
                                     ↓
Streamlit Community Cloud ← Git pull ← Repo
                                     ↑
Google Colab (週次手動)
  └→ モデル学習・再構築 → git commit

Mac launchd (直前バッチ・任意)
  └→ 直前水分量更新 + パターン事後更新 → git commit
```

## データソース

| ソース | 用途 |
|---|---|
| [keiba.go.jp](https://www.keiba.go.jp/) | メイン（出馬表・結果・水分量・払戻） |
| [baneidb.com](https://baneidb.com/) | 補完（条件別回収率の参考値） |
| [ばんえい十勝公式](https://www.banei-keiba.or.jp/) | 補完（格付け・賞金変動） |
| [netkeiba (NAR)](https://nar.netkeiba.com/) | 補完（馬個別の長期検索） |

## Phase 0 で確認した keiba.go.jp の HTML 構造

URL: `https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/RaceMarkTable?k_raceDate=YYYY/MM/DD&k_raceNo=N&k_babaCode=3`

### レースヘッダ部
プレーンテキストとしてレース情報が散在。以下の正規表現で抽出可能:

| 項目 | パターン | 例 |
|---|---|---|
| 開催日・レース番号 | `(\d{4})年(\d{1,2})月(\d{1,2})日.*?帯\s*広第(\S+?)競走` | "2026年5月4日 （月）帯 広第１競走" |
| 距離 | `(\d+)ｍ` | "200ｍ" |
| 天候・馬場水分量 | `天候：(\S+?)\s*馬場：([\d.]+)` | "天候：小雨 馬場：2.8" |
| レース名・クラス | `馬場：[\d.]+\s+(\S[^（]+?)\s+（` | "飛花琳＆崇獅 初誕生記念Ｃ２－１５" |
| 賞金 | `(\d)着\s+([\d,]+)円` | "1着 300,000円" |

**水分量 (`track_water_pct`)** : 「馬場：2.8」の形で記載。レース毎に取得すれば自動で時系列化される（一日の途中で 2.8 → 2.7 と変化する例を 2026-05-04 で確認）。

### 結果テーブル
class=`bs` の最内テーブルで、直下tdに「着順」「馬名」を含む行をヘッダとする。
入れ子テーブルが多いため、必ず `find_all(recursive=False)` で直下要素のみを取得すること。

15列構成:

| # | 列名 | 例 | 備考 |
|---|---|---|---|
| 1 | 着順 | `1` | イレギュラー時 `失/中/故/除/取/降` |
| 2 | 枠番 | `4` | |
| 3 | 馬番 | `4` | |
| 4 | 馬名 | `シオン` | リンク先 `k_lineageLoginCode` を馬IDとして使用 |
| 5 | 所属 | `ばんえい` | |
| 6 | 性齢 | `牝 3` | |
| 7 | 積載重量 | `540` | 規定積載重量(kg) |
| 8 | 騎手(所属) | `☆今井千(ばんえい)` | **先頭の☆等が減量マーカー** / リンク先 `k_riderLicenseNo` |
| 9 | 調教師 | `久田守` | リンク先 `k_trainerLicenseNo` |
| 10 | 馬体重 | `860` | 当日体重 (kg) |
| 11 | 差 | `12` | **馬体重前走比 (kg, 符号付き)** |
| 12 | タイム | `2:11.5` | |
| 13 | 着差 | (空) | **空欄。タイム差から逆算する** |
| 14 | 上り3F | (空) | **ばんえいでは概念なし。常に空欄** |
| 15 | 人気 | `2` | |

### 払戻テーブル
class=`None` の最内テーブルで、直下tdの先頭が「Ｒ」かつ「単勝」を含む行を持つ。
7券種: 単勝 / 複勝 / 馬連複 / 馬連単 / ワイド / 三連複 / 三連単

複勝・ワイド等は1セル内に複数組分の値が `<br>` 区切りで連結（または `円` 区切りの単一テキスト）。
`_split_br()` で展開する。

### 減量マーカー（騎手）
Phase 0 サンプル（2026-05-04 全12R中5Rを確認）では **☆** のみ観察。
今後確認すべきマーカー: `△ / ▲ / ◇ / ★`（女性騎手・見習い段階別で表記が変わる想定）。
具体の減量kgはサイトに明示されないため、騎手マスタ側で適用ルールを別途持つ必要あり（Phase 1）。

### イレギュラー対応
- 着差列は常時空欄 → タイム差から逆算（`time_diff_from_winner_sec`, `time_diff_from_prev_sec`）
- 同タイム別着順あり（R1で5位/6位が同 2:23.5） → タイム逆算では `time_diff_from_prev=0` で表現
- 真の同着（同一 finish_pos が複数）は `finish_status='dead_heat'` で識別
- 失格/中止/落馬/取消/降着は `raw_position_text` の文字から `finish_status` に変換

## ディレクトリ構成

```
banei-analytics/
├── scrapers/
│   ├── fetcher.py          # レート制限付きHTMLフェッチャ + ディスクキャッシュ
│   └── race_page.py        # keiba.go.jp RaceMarkTable パーサ
├── persist/
│   └── parquet_writer.py   # 月次パーティション Parquet writer (upsert対応)
├── data/
│   ├── raw_html/           # 生HTML キャッシュ (gitignore: 2025年以前は除外)
│   └── parquet/            # 永続化レイヤ (gitignore: 全除外・要再生成)
├── run_phase0.py           # Phase 0 単発検証
├── run_phase1.py           # Phase 1 日次パイプライン
├── pyproject.toml
└── README.md
```

## Phase 1 データスキーマ

Hiveパーティション形式の Parquet で保管:
```
data/parquet/<table>/year=YYYY/month=MM/data.parquet
```

### races テーブル (1行/レース)
| 列 | 型 | 備考 |
|---|---|---|
| race_id | str | PK. `YYYYMMDD_<baba>_NN` 例 `20260504_3_01` |
| race_date | str | YYYY-MM-DD |
| race_no | int | 1〜12 |
| baba_code | int | 帯広=3 |
| course_name | str | "帯広" |
| distance_m | int | 200 |
| weather, track_water_pct | str, float | スクレイプ時点の値 |
| race_name, race_class | str | クラス例: `C2-15`, `A2-1` |
| entry_count | int | 出走頭数 |
| prize_1st〜prize_5th | int | 円 |
| fetched_at | str | ISO8601 UTC |
| source_url | str | 取得元URL |

### entries テーブル (1行/(race_id, horse_no))
出馬情報と結果情報を統合。

| 列 | 備考 |
|---|---|
| race_id, horse_no | 複合PK |
| post_position, horse_id, horse_name | 馬IDは `k_lineageLoginCode` |
| affiliation, sex_age | "ばんえい" / "牡 3" |
| load_weight_kg | 積載重量 |
| jockey_id, jockey_name, allowance_marker | 騎手ID = `k_riderLicenseNo`, ☆等 |
| trainer_id, trainer_name | 調教師ID = `k_trainerLicenseNo` |
| body_weight_kg, body_weight_diff_kg | 当日体重・前走比 |
| finish_pos (nullable) | イレギュラー時 null |
| finish_status | `finished` / `dead_heat` / `disqualified` / `cancelled` / `fell` / `scratched_late` / `scratched_early` / `demoted` / `unknown` |
| finish_time_sec | float秒 |
| time_diff_from_winner_sec, time_diff_from_prev_sec | タイム逆算 |
| popularity | 人気 |
| raw_position_text, raw_margin_text | 生文字列(検証用) |

### payouts テーブル (1行/(race_id, bet_type, combination))
| 列 | 例 |
|---|---|
| race_id, bet_type, combination | `単勝`, `1-4`, `4-1-7` 等 |
| payout_yen | 払戻金 |
| popularity | 人気順位 |

## 永続化の冪等性

`persist.parquet_writer.write_race()` は **PK重複行を新値で上書き** する upsert を行う。
同一レースを再スクレイプして書き込んでも重複行は作られない。

## 実行

### 単日パイプライン (Phase 1)
```bash
uv run python run_phase1.py 2026-05-04
uv run python run_phase1.py 2026-05-04 --force-refresh  # 再取得
```

### 期間バックフィル (Phase 2)
```bash
# ローカルで2週間分
uv run python backfill.py --start 2026-05-01 --end 2026-05-14

# Colab用 (HTMLを使い捨て・ディスク節約)
uv run python backfill.py --start 2024-04-01 --end 2024-09-30 --no-html-cache

# チェックポイント無視してやり直し
uv run python backfill.py --start 2026-05-01 --end 2026-05-07 --no-resume
```

レート制限: デフォルト 2.5秒/リクエスト。HTMLキャッシュヒット時はスリープ省略。
チェックポイント: `data/checkpoints/backfill_progress.json` に進捗を記録し、中断後の再開を可能にする。

## Colab で 10年バックフィルを実行する手順

1. **GitHub PAT 発行**
   - [github.com/settings/personal-access-tokens/new](https://github.com/settings/personal-access-tokens/new) でfine-grained PATを作成
   - Repository access: `keibakaiseki-svg/banei-analytics` のみ
   - Permissions: `Contents` = **Read and write**
2. **Colab を開く**: `notebooks/02_backfill_colab.ipynb` を Google Colab で開く
3. **Secrets 登録**: Colab左サイドバーの🔑から `GITHUB_PAT` 名で発行したPATを登録、Notebook accessをオン
4. **セルを順に実行**: クローン → 依存導入 → バックフィル → 進捗確認 → push
5. **チャンク推奨**: 1セッションあたり半年〜1年分。各セッション後に必ず push してチェックポイントを保存

データ取得可能範囲: **2014年〜現在** （2014-05-04・2018-05-05・2020-05-04・2024-05-04で取得成功確認済）。
1日のキャッシュなし取得: 12レース × 2.5秒 ≈ 30秒/日。
1年(約160開催日) ≈ 80分のネット時間。10年で約13時間。

## ロードマップ

| Phase | 内容 | ステータス |
|---|---|---|
| 0 | HTML構造調査・最小スクレイパ | 完了 |
| 1 | 全項目スクレイパ・DuckDB保存 | 完了 |
| 2 | Colabで過去10年バックフィル | 基盤完了 (実行待ち) |
| 3 | 探索分析（馬番効果・水分量効果検証） | 完了 |
| 3.5 | 派生特徴量生成（脚質代替指標） | |
| 4 | 馬場パターンクラスタリング | |
| 5 | ベース予想モデル (LightGBM ranker) | |
| 6 | パターン別補正モデル | |
| 7 | Streamlit ダッシュボード MVP | |
| 8 | GitHub Actions 朝バッチ自動化 | |
| 9 | Mac launchd 直前バッチ | |
| 10 | 運用開始・モデル改善ループ | |
| 11 | YouTube動画解析（採算化後） | |

## Phase 1 完了サマリ（2026-05-04 全12R）

- 取得レース: 12 / 出走エントリ: 107 / 払戻レコード: 130
- 一日内の水分量変動を確認: R6まで 2.8 → R7以降 2.7
- 天候変動を確認: 小雨 → 曇
- 減量マーカー(☆)を全12Rで合計25騎手から検出
- PK重複ゼロ・冪等性検証済（再実行で全行 `replaced` 扱い）

## Phase 2 完了サマリ

- [backfill.py](backfill.py): 日付範囲指定・チェックポイント保存・再開可能なバックフィルスクリプト
- [notebooks/02_backfill_colab.ipynb](notebooks/02_backfill_colab.ipynb): Colab実行用ノートブック
- ローカル7日間テスト: 3開催日・36レース・322エントリを44KBのParquetに永続化
- 過去10年遡及可能性検証: 2014-05-04・2018-05-05・2020-05-04・2024-05-04で取得成功
- 開催なし日は `no_race_dates` に記録され再実行時にスキップ

## Phase 3 完了サマリ

### モジュール
- [analytics/masters.py](analytics/masters.py): 馬・騎手・調教師マスタを entries から派生生成（純粋関数・冪等）
- [analytics/eda.py](analytics/eda.py): 再利用可能な探索分析クエリ群
- [analytics/repeats.py](analytics/repeats.py): 連続2走ペアからのタイム変化要因分析
- [notebooks/03_eda.ipynb](notebooks/03_eda.ipynb): EDA ノートブック

### データ規模 (Phase 2 Chunk 2 完了時点・2026-05-17)
**10,578 レース・98,582 エントリ・2,121 ユニーク馬・連続2走ペア 94,902件**
(範囲: 2020-04-01 〜 2026-04-30, 残り 2014-04-01 〜 2020-03-31 を Phase 2 Chunk 3 で取得予定)

### 単独変数の影響度（タイム変化との Pearson 相関、n=94,902）

| 変数 | 30日サンプル(n=960) | フルデータ(n=94,902) | 解釈 |
|---|---|---|---|
| **馬場水分量変化** | -0.613 | **-0.294** | 最強の予測因子 |
| 着順変化 | +0.426 | +0.446 | 一貫 |
| 積載重量変化 | +0.213 | +0.283 | 中程度 |
| 馬体重変化 | -0.026 | -0.078 | 弱い (単独では効きにくい) |
| **中休み日数** | -0.390 | **+0.005** | **30日サンプルのバイアスだった・実際は無相関** |

### 水分量変化帯 × タイム変化 (200m基準・フルデータ)

| 変化幅 | n | 平均タイム変化 |
|---|---|---|
| 大きく湿る (+1.0以上) | 13,130 | **-11.5秒** |
| 少し湿る | 31,130 | -4.2秒 |
| 変化なし | 5,602 | -0.6秒 |
| 少し乾く | 33,210 | +3.8秒 |
| 大きく乾く (-1.0以上) | 11,830 | **+11.8秒** |

→ 水分量±1.0の変化で **±11秒** 動く。引き続き **モデル最重要特徴量**。

### 馬体重変化帯 × 複勝率 (フルデータ・前回のU字型から単調関係に修正)

| 帯 | n | 平均タイム変化 | 複勝率 |
|---|---|---|---|
| 減15kg以上 | 5,613 | +3.95秒 | 25.9% |
| 減5-15kg | 17,803 | +1.13秒 | 30.6% |
| ±5kg | 32,628 | +0.18秒 | 32.6% |
| 増5-15kg | 27,182 | -1.06秒 | 34.3% |
| 増15kg以上 | 11,480 | **-3.32秒** | **35.6%** |

→ **馬体重増加 → タイム短縮・複勝率上昇** の単調関係 (n=960 で観察されたU字型はノイズだった)。

### 騎手交代の効果 (n=34,536交代 / 60,366継続)

- 騎手交代: 平均タイム -0.9秒・着順変化 -0.07
- 騎手継続: 平均タイム +0.2秒・着順変化 +0.07

→ **1.1秒のタイム短縮効果** (30日サンプルとほぼ一致)。

### 過去N走平均着順 → 現走着順 (フルデータ)

- past_3: r=+0.204 (n=90,747)
- past_5: r=+0.194 (n=86,695)

→ 過去成績は **弱い予測因子**。LightGBM の補助特徴量として有用。

## Phase 4 への引き継ぎ事項

### 単独変数の追検証 (フルバックフィル後)
1. 馬番効果の有意性検定 (二項検定)
2. 1番人気勝率の長期分布
3. 単勝回収率が1.0を超える人気層の特定

### 交互作用分析 (フルバックフィル後)
データを増やしてから精緻に検証する論点:
- **中休み日数 × 水分量変化**: 休み明け馬は新馬場への適応にハンディがあるか
- **中休み日数 × 馬体重変化**: 休養中の体重変動と能力の関係
- **馬体重変化 × 水分量**: ±5kg内が最良という効果が水分量条件で変わるか
- **騎手交代 × 調教師パターン**: 特定調教師が起用する騎手による効果差
- **馬番 × 水分量**: 馬場が極端な時の内外差

### モデル設計準備
1. 馬体重派生特徴量: ±5kg内フラグ・+15kg以上特例・ベスト体重乖離・3走移動平均との差
2. 水分量派生特徴量: 絶対値・前走比変化・変化方向×絶対値の交互作用
3. 馬場パターンクラスタリング軸の確定 (5〜7パターン抽出)
4. パターン別 LightGBM ranker のベースライン構築準備
5. 減量マーカー☆以外のバリアント収集 (女性騎手騎乗レースを別日から取得)
6. マーカー → kg のマッピングルール整理
7. 失格・中止等イレギュラーレースの実例収集

## やらないこと

- 自動投票
- リアルタイムオッズ秒単位取得
- 24時間稼働サーバー
- 全国地方競馬対応（帯広限定）
- 動画解析（Phase 11 採算化後）

## 開発環境

- Python 3.12 (uv 管理)
- 主要依存: httpx, beautifulsoup4, lxml, pandas, pyarrow

## 実行

```bash
uv run python run_phase0.py
```

## ライセンス・利用上の注意

- スクレイピング対象サイトの利用規約・robots.txt を遵守
- リクエスト間隔は 2〜3 秒以上推奨
- データの二次配布は行わない
- 商用利用は別途検討
