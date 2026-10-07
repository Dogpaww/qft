"""LoRA SFT of Qwen3-4B-Instruct-2507 on the question/answer pairs built by data_prep.py.

  python train.py [--config config.yaml]

Loss is computed on assistant answer tokens only.
Adapter is saved on its own; the base model is never merged or modified.
"""
import argparse, json, os, random

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)

from data_prep import load_cfg, read_jsonl, CFG_PATH


def qa_example(tok, system, q, a, max_len):
    """Tokenise with the Qwen chat template; mask everything except the assistant answer."""
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(a + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}


class Pad:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids = torch.tensor([b["input_ids"] + [self.pad_id] * (m - len(b["input_ids"])) for b in batch])
        lab = torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch])
        att = torch.tensor([[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch])
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


class ListDS(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def plot_loss(log_history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = [(l["step"], l["loss"]) for l in log_history if "loss" in l]
    ev = [(l["step"], l["eval_loss"]) for l in log_history if "eval_loss" in l]
    plt.figure(figsize=(7, 4))
    plt.plot(*zip(*tr), label="train loss")
    if ev:
        plt.plot(*zip(*ev), "o-", label="eval loss (120 main questions)")
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(path)


def main(cfg):
    tc, P = cfg["train"], cfg["paths"]
    seed = cfg["seed"]
    random.seed(seed); torch.manual_seed(seed)
    rng = random.Random(seed)
    os.makedirs(P["logs_dir"], exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["models"]["base"])
    system = cfg["inference"]["system_prompt"]  # same lore-free prompt as inference

    qa = read_jsonl(P["qa_train"])
    rows = [qa_example(tok, system, r["question"], r["answer"], tc["max_seq_len"]) for r in qa]
    rng.shuffle(rows)
    ev_rows = [qa_example(tok, system, r["question"], r["answer"], tc["max_seq_len"]) for r in read_jsonl(P["qa_eval"])]
    print(f"train examples: {len(rows)}; eval examples: {len(ev_rows)}")

    kw = dict(torch_dtype=torch.bfloat16)
    if tc["qlora_fallback"]:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(cfg["models"]["base"], **kw)
    if tc["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lc = tc["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=P["adapter_dir"], num_train_epochs=tc["epochs"], learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler"], warmup_ratio=tc["warmup_ratio"], weight_decay=tc["weight_decay"],
        per_device_train_batch_size=tc["per_device_batch_size"], per_device_eval_batch_size=tc["per_device_batch_size"],
        gradient_accumulation_steps=tc["grad_accum_steps"], bf16=tc["bf16"],
        logging_steps=tc["logging_steps"], logging_dir=P["logs_dir"],
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=None,   # checkpoint every epoch
        seed=seed, data_seed=seed, report_to="tensorboard", remove_unused_columns=False,
        group_by_length=False)
    trainer = Trainer(model=model, args=args, train_dataset=ListDS(rows), eval_dataset=ListDS(ev_rows),
                      data_collator=Pad(tok.pad_token_id))
    trainer.train()
    model.save_pretrained(P["adapter_dir"])          # adapter only; base untouched, not merged
    tok.save_pretrained(P["adapter_dir"])
    json.dump(trainer.state.log_history, open(os.path.join(P["logs_dir"], "train_log.json"), "w"), indent=1)
    plot_loss(trainer.state.log_history, os.path.join(P["logs_dir"], "loss_curve.png"))
    print("adapter saved to", P["adapter_dir"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_PATH)
    main(load_cfg(ap.parse_args().config))
