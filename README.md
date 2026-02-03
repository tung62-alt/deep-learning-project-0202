# Low-Resource Akkadian Transliteration → English with ByT5 (Self-Learning Pipeline)

本專案針對**極低資源**的古代語言翻譯任務：將 **阿卡德語轉寫（transliteration）** 翻成 **英文**。資料具有**符號密集、格式多變、噪聲高**等特性，因此採用 **ByT5（byte-to-byte, token-free）** 作為核心模型，並設計一套「修補可疑標註 → 自我蒸餾 → 未標註偽標註 → 最終課程式收斂」的訓練管線，以提升翻譯品質與穩定性。

---

## 目錄
- [任務與資料](#任務與資料)
- [為什麼用 ByT5](#為什麼用-byt5)
- [方法總覽（Stage 2–5）](#方法總覽stage-25)
- [安裝環境](#安裝環境)
- [快速開始（Colab / 本機）](#快速開始colab--本機)
- [評估指標](#評估指標)
- [實驗結果](#實驗結果)
- [引用與致謝](#引用與致謝)
- [License](#license)

---

## 任務與資料

### 任務定義
- **Input**：阿卡德語轉寫（transliteration）文字（包含學術標註、括號、缺字、重建符號、數字下標等）
- **Output**：英文翻譯（English)

### 資料來源與特性（概述）
- 低資源：訓練對齊句對數量有限（約千級規模）
- 噪聲高：含可疑標註（suspicious）、錯配、缺漏、格式不一致
- 字串細節重要：符號、拼寫變形、括號與標記會影響語義

---

## 為什麼用 ByT5

### 1) Token-free（byte-to-byte）更適配轉寫輸入
轉寫資料混雜大量符號與不規則標記，傳統 subword tokenizer 容易產生不穩定切分與 OOV-like 行為；ByT5 以 bytes 建模，可降低 tokenization 對罕見符號與拼寫變形的敏感度。

### 2) 抗噪與字內部模式（within-word patterns）
轉寫對「字串內部細節」高度敏感；ByT5 能更直接學到字元/符號層級的規律，對拼寫擾動與符號噪聲通常更有韌性。

### 3) 對低資源更友善
在資料量受限時，byte-level 建模可避免詞表設計與切分帶來的額外稀疏性問題，讓模型把容量用在學習字串型態與對齊模式。

---

## 方法總覽（Stage 2–5）

本專案的訓練管線（pipeline）重點是：**先降低髒資料污染**，再用自我訓練與未標註偽標註擴增資料量，最後用 curriculum 方式回到高品質 gold 分佈收斂。

### Stage 2 — Repair suspicious（修補可疑標註）
目的：降低可疑樣本（suspicious）對模型的污染。

做法（概念）：
1. 先用較乾淨的 gold 訓練 **Teacher0**
2. Teacher0 對 suspicious 樣本重新生成 target（pseudo repair）
3. 用信心門檻 `conf_th` 過濾低信心樣本，只保留較可靠的修補結果
4. 輸出一份「已修補/已標記」的訓練資料（repaired / flagged）

常用參數（示意）：
- `conf_th`：信心門檻，越高表示越嚴格保留
- `gen_bs`：生成 batch size（inference）
- `--do_repair`：啟用修補流程

### Stage 3 — Self-distillation（自我蒸餾 / self-training）
目的：把 Teacher0 更平滑一致的輸出分佈蒸餾回去，提升泛化與穩定性。

做法（概念）：
1. Teacher0 只對 **train split** 產生 pseudo（避免 dev leakage）
2. 以 `pseudo_weight` 將 pseudo 與 gold 混合訓練 **Student**
3. 常搭配較小 learning rate、較少 epochs，並可同樣用 `conf_th` 或其他規則降低雜訊

常用參數（示意）：
- `pseudo_weight`：pseudo 的混合權重（例如 0.3～0.6）
- `--do_selfdistill`：啟用自我蒸餾

### Stage 4 — Unlabeled pseudo labeling（未標註資料偽標註）
目的：使用未標註資料（unlabeled）擴充訓練資料量。

做法（概念）：
1. 將 unlabeled 切分（例如 A/B）
2. 依 `u_part` + `u_take` 抽樣一部分資料產生 pseudo label
3. 以 `conf_th` 過濾（可選），輸出 `unlabeled_*_byt5_pseudo.csv`

常用參數（示意）：
- `u_part`：取哪個 partition（A 或 B）
- `u_take`：取多少條 unlabeled 產 pseudo
- `--do_label_unlabeled_byt5`：啟用 ByT5 對 unlabeled 產 pseudo

### Stage 5 — Final Training（Curriculum Fine-tuning）
目的：先吸收多樣性，再回到高品質 gold 分佈收斂，穩定提升最終指標。

做法（概念）：
1. **Mixed Training**：Repaired gold + unlabeled pseudo（提升覆蓋率與多樣性）
2. **Gold Refine**：最後只用高品質 gold + 低 LR 收斂到真實標註分佈
3. 產生最終 submission（推論 test）

常用參數（示意）：
- `--do_final_train`：啟用最終訓練
- `--do_infer`：輸出 submission
- `--bf16`：使用 bf16（視硬體支援）

---

## 安裝環境

### 建議版本（示意）
- Python 3.10+
- PyTorch（依 CUDA 版本）
- transformers / datasets / accelerate
- sacrebleu / evaluate
- （可選）peft / trl / bitsandbytes（若要 LoRA 產 pseudo）

### 安裝指令（示意）
```bash
pip install -U transformers datasets accelerate sacrebleu sentencepiece evaluate

```


## 快速開始（Colab / 本機）

### Step 0：資料清理與 suspicious 標記（輸出 flagged）

```bash
python pipeline_align_train.py \
  --train_path /content/train.csv \
  --sentences_path /content/Sentences_Oare_FirstWord_LinNum.csv \
  --out_dir /content/out \
  --filter_suspicious \
  --epochs 10
```

預期輸出（示意）：

/content/out/train_flagged.csv

### Step 1：跑完整 ByT5 Self-Learning Pipeline（Stage 2–5 + inference）
```bash
python byt5_pseudo_pipeline.py \
  --train_csv /content/train.csv \
  --flagged_csv /content/out/train_flagged.csv \
  --test_csv /content/test.csv \
  --sample_sub_csv /content/sample_submission.csv \
  --unlabeled_csv /content/published_texts.csv \
  --out_dir /content/pseudo_out \
  --bf16 \
  --gen_bs 8 \
  --do_repair \
  --do_selfdistill \
  --do_label_unlabeled_byt5 \
  --u_part A --u_take 2000 \
  --do_final_train \
  --do_infer
```

### Step 2（可選）：加入資料增強（Context Concat + Gap Mask）
```bash
python byt5_pseudo_pipeline_combined.py \
  --train_csv /content/train_cleaned.csv \
  --flagged_csv /content/out/train_flagged.csv \
  --test_csv /content/test.csv \
  --sample_sub_csv /content/sample_submission.csv \
  --out_dir runs/exp1 \
  --do_context_concat --ctx_ratio 0.25 \
  --do_gap_mask --gap_ratio 0.30
  ```

### Step 3（可選）：用 LLM（LoRA）產 unlabeled pseudo
```bash
python llm_lora_pseudo.py \
  --bf16 \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --gold_train_csv /content/train.csv \
  --unlabeled_csv /content/published_texts.csv \
  --out_dir /content/llm_out \
  --u_part B --u_take 2000 \
  --max_seq_len 1024 \
  --max_steps 800 \
  --batch_size 2 --grad_accum 8 \
  --gen_bs 8
   ```

### Step 4（可選）：把 LLM pseudo 回灌 ByT5 做最終訓練
```bash
python byt5_pseudo_pipeline_combined_v4d_teacher0_uses_traincsv.py \
  --train_csv /content/train_v2.csv \
  --flagged_csv /content/out/train_flagged.csv \
  --test_csv /content/test.csv \
  --sample_sub_csv /content/sample_submission.csv \
  --unlabeled_csv /content/published_texts.csv \
  --out_dir runs/exp_full \
  --do_context_concat --ctx_ratio 0.25 \
  --do_gap_mask --gap_ratio 0.30 \
  --do_repair --do_selfdistill --do_label_unlabeled_byt5 --do_final_train --do_infer \
  --export_pseudo_for_other_model
  ```

## 評估指標
- BLEU

- ChrF++（character n-gram F-score）

- Kaggle metric：BLEU 與 ChrF++ 的幾何平均（geometric mean）

## 實驗結果

> 以下以簡報/實驗紀錄中的結果為主。

### 整體方法比較（範例）

| 方法 | BLEU | ChrF++ | Final Metric |
|---|---:|---:|---:|
| Baseline (ByT5 Basic) | 2.55 | 15.28 | 6.25 |
| + Self-Learning (Pipeline) | 6.2 | 20.23 | 11.2 |
| + Context-Gap Augmentation | 7.91 | 23.59 | 13.66 |
| + LLM Pseudo Labeling (Best) | 9.29 | 28.23 | 16.2 |

### 參數掃描（建議放法）

- `u_take`（例如 1000 / 2000 / 4000）-> 2000效果最佳
- `conf_th`（例如 -2 / -1 / 0）-> 實驗目前都放2
- `pseudo_weight`（例如 0.3 / 0.4 / 0.5 / 0.6）-> 目前實驗下來0.4效果最佳


## 引用與致謝

- **資料來源**：Kaggle Deep Past Challenge-Translate Akkadian to English <https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/overview>
- **核心模型**：ByT5（`google/byt5-small`）
- **方法設計與結果整理參考**：
  - Semi-supervised approach for Transformers（Towards Data Science）：<https://towardsdatascience.com/semi-supervised-approach-for-transformers-38722f1b5ce3/>
  - ChatGPT 樣本提示指令：zero-shot / one-shot / few-shot（Medium）：<https://medium.com/seaniap/chatgpt%E4%B9%8B%E6%A8%A3%E6%9C%AC%E6%8F%90%E7%A4%BA%E6%8C%87%E4%BB%A4-zero-shot-one-shot-%E8%88%87few-shot-c5d3b91b02c4>
  - Transcending Language Boundaries: Harnessing LLMs for Low-Resource Language Translation：<https://arxiv.org/html/2411.11295v1>
  - ByT5: Towards a token-free future with pre-trained byte-to-byte models：<https://arxiv.org/abs/2105.13626>


