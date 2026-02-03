# -*- coding: utf-8 -*-
import os, re, json, math, random, argparse
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# -----------------------------
# cleaning（沿用你原檔）
# -----------------------------
CONTROL_CHARS = re.compile(r"[\u0000-\u001F\u007F-\u009F\u200B\u200C\u200D\uFEFF]")
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
DET_PATTERN = re.compile(r"\{([^}]+)\}")
DET_MAP_SHORT = {"d":"DG","mul":"ST","ki":"KI","lu₂":"LU","e₂":"E2","uru":"UR","kur":"KR","mi":"MI","m":"M","geš":"GS","ĝeš":"GS","tug₂":"TG","dub":"DB","id₂":"ID","mušen":"MS","na₄":"NA4","kuš":"KS","u₂":"U2"}

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def clean_x(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s)
    s = CONTROL_CHARS.sub("", s)
    s = re.sub(r"\[x\]", "<gap>", s)
    s = re.sub(r"\.{3,}", "<big_gap>", s)

    def repl(m):
        det = m.group(1).strip()
        tag = DET_MAP_SHORT.get(det, det)
        return f"<{tag}>"
    s = DET_PATTERN.sub(repl, s)

    s = re.sub(r"(<[^>]+>)", r" \1 ", s)
    s = re.sub(r"<([^>]+)>", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]", r"\1", s)

    s = s.translate(SUBSCRIPT_MAP)
    s = s.replace("/", " ")
    s = re.sub(r"[:.]", " ", s)
    s = re.sub(r"[⌜⌝!?]", "", s)
    return normalize_ws(s)

def clean_y(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s)
    s = CONTROL_CHARS.sub("", s)
    s = re.sub(r"<([^>]+)>", r"\1", s)
    return normalize_ws(s)

def is_bad_text(s: str) -> bool:
    if s is None: return True
    s = str(s).strip()
    if len(s) < 3: return True
    if re.fullmatch(r"[.\s…]+", s): return True
    return False

# -----------------------------
# prompt format
# -----------------------------
def build_prompt(x: str) -> str:
    return (
        "You are a careful translator.\n"
        "Task: Translate the following transliteration into fluent English.\n"
        "Rules:\n"
        "- Output English translation only.\n"
        "- Keep proper names as they appear.\n"
        "- Do not add commentary.\n\n"
        f"Transliteration:\n{x}\n\n"
        "English translation:\n"
    )

# -----------------------------
# generation
# -----------------------------
@torch.no_grad()
def generate_translations(model, tok, prompts, max_new_tokens=180, batch_size=8, temperature=0.0, max_prompt_len=1024):
    device = next(model.parameters()).device
    outs = []

    # 推薦：eval mode（避免 dropout）
    model.eval()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        ).to(device)

        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            num_beams=1,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        texts = tok.batch_decode(gen, skip_special_tokens=True)

        for p, t in zip(batch, texts):
            if "English translation:" in t:
                t = t.split("English translation:", 1)[-1].strip()
            else:
                t = t[len(p):].strip() if t.startswith(p) else t.strip()
            outs.append(normalize_ws(t))
    return outs

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--gold_train_csv", required=True)
    ap.add_argument("--unlabeled_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=1024)

    ap.add_argument("--u_part", choices=["A","B"], default="B")
    ap.add_argument("--u_take", type=int, default=2000)
    ap.add_argument("--u_seed", type=int, default=123)

    ap.add_argument("--gen_bs", type=int, default=8)
    ap.add_argument("--gen_max_new", type=int, default=180)
    ap.add_argument("--bf16", action="store_true", help="use bf16 (recommended on L4/A100)")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # -----------------------------
    # 1) Load gold and build SFT text
    # -----------------------------
    gold = pd.read_csv(args.gold_train_csv)
    if "transliteration" not in gold.columns or "translation" not in gold.columns:
        raise ValueError("gold_train_csv needs columns: transliteration, translation")

    gold["X_clean"] = gold["transliteration"].apply(clean_x)
    gold["Y_clean"] = gold["translation"].apply(clean_y)
    gold = gold[(gold["X_clean"].str.len()>0) & (gold["Y_clean"].str.len()>0)].reset_index(drop=True)

    gold["text"] = gold.apply(lambda r: build_prompt(r["X_clean"]) + r["Y_clean"], axis=1)
    train_ds = Dataset.from_pandas(gold[["text"]], preserve_index=False)

    # -----------------------------
    # 2) Load model in 4bit + LoRA
    # -----------------------------
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=compute_dtype,
    )

    # L4 省顯存：開 checkpointing + 關 cache（TRL 0.27.1 也預設 use_cache=False，但保險）
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # -----------------------------
    # 3) SFT train (TRL 0.27.1 正確用法)
    # -----------------------------
    sft_cfg = SFTConfig(
        output_dir=os.path.join(args.out_dir, "llm_lora_ckpt"),
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        fp16=(not args.bf16),
        bf16=args.bf16,
        report_to="none",
        # 關鍵：在 config 指定資料欄位 & max_length（你的 signature 裡就有）
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=False,
        use_cache=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        processing_class=tok,   # TRL 0.27.1 用 processing_class
    )
    trainer.train()

    # 存 LoRA adapter
    adapter_dir = os.path.join(args.out_dir, "llm_lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    # -----------------------------
    # 4) Generate pseudo-label for unlabeled part
    # -----------------------------
    U = pd.read_csv(args.unlabeled_csv)
    if "transliteration" not in U.columns:
        raise ValueError("unlabeled_csv needs column: transliteration")
    U["X_clean"] = U["transliteration"].apply(clean_x)
    U = U[U["X_clean"].str.len()>0].reset_index(drop=True)

    rng = np.random.default_rng(args.u_seed)
    idx = np.arange(len(U))
    rng.shuffle(idx)
    mid = len(U)//2
    pick = idx[:mid] if args.u_part=="A" else idx[mid:]
    pick = pick[:min(args.u_take, len(pick))]
    U_part = U.iloc[pick].copy().reset_index(drop=True)

    prompts = [build_prompt(x) for x in U_part["X_clean"].tolist()]
    pseudo = generate_translations(
        trainer.model, tok, prompts,
        max_new_tokens=args.gen_max_new,
        batch_size=args.gen_bs,
        temperature=0.0,
        max_prompt_len=args.max_seq_len,   # 跟訓練長度一致
    )

    U_part["translation"] = pseudo
    U_part = U_part[~U_part["translation"].apply(is_bad_text)].reset_index(drop=True)

    out_csv = os.path.join(args.out_dir, f"unlabeled_{args.u_part}_llm_pseudo.csv")
    U_part.to_csv(out_csv, index=False)
    print("[LLM pseudo] saved:", out_csv, "n=", len(U_part))

if __name__ == "__main__":
    main()
