#!/usr/bin/env python3
import argparse
import os
from dataclasses import asdict

from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoTokenizer

from trl import GKDConfig, GKDTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train TRL GKD on local GSM8K parquet files.")
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=6.0)
    parser.add_argument("--lmbda", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--dataset_num_proc", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="wandb", help="wandb or none")
    parser.add_argument("--project_name", type=str, default="trl_gkd_gsm8k_compare")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    return parser.parse_args()


def to_chat_messages(raw_prompt):
    # gsm8k_gkd parquet stores prompt as an array/object of chat dicts.
    if hasattr(raw_prompt, "tolist"):
        raw_prompt = raw_prompt.tolist()
    if isinstance(raw_prompt, dict):
        raw_prompt = [raw_prompt]
    if isinstance(raw_prompt, str):
        return [{"role": "user", "content": raw_prompt}]
    if isinstance(raw_prompt, list):
        return raw_prompt
    return [{"role": "user", "content": str(raw_prompt)}]


def normalize_for_trl_gkd(messages):
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        content = msg.get("content", "")
        if content is None:
            content = ""
        normalized.append({"role": role, "content": str(content)})

    if not normalized:
        normalized = [{"role": "user", "content": ""}]

    # TRL GKD's DataCollatorForChatML builds prompt from messages[:-1].
    # Ensure there is always a final assistant message as completion placeholder.
    if normalized[-1]["role"] != "assistant":
        normalized.append({"role": "assistant", "content": ""})

    return normalized


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if args.report_to.lower() == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.project_name)

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("parquet", data_files={"train": args.train_file, "test": args.eval_file})

    def preprocess(example):
        messages = normalize_for_trl_gkd(to_chat_messages(example["prompt"]))
        return {"messages": messages}

    with PartialState().local_main_process_first():
        dataset = dataset.map(
            preprocess,
            num_proc=args.dataset_num_proc,
            desc="Preprocessing prompt->messages for TRL GKD",
        )
        for split in list(dataset.keys()):
            drop_cols = [c for c in dataset[split].column_names if c not in {"messages"}]
            if drop_cols:
                dataset[split] = dataset[split].remove_columns(drop_cols)

    run_name = args.run_name or os.path.basename(args.output_dir.rstrip("/"))
    report_to = "none" if args.report_to.lower() == "none" else ["wandb"]
    eval_strategy = "no" if args.eval_steps <= 0 else "steps"

    gkd_args = GKDConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        teacher_model_name_or_path=args.teacher_model,
        do_train=True,
        do_eval=eval_strategy != "no",
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=4,
        report_to=report_to,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_drop_last=True,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        max_length=args.max_length,
        dataset_num_proc=args.dataset_num_proc,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        lmbda=args.lmbda,
        beta=args.beta,
    )

    if local_rank == 0:
        print(
            f"[trl-gkd] temperature={args.temperature}, gamma={args.gamma} "
            f"(gamma not used by TRL GKD; kept only for setting parity)",
            flush=True,
        )
        print(f"[trl-gkd] config: {asdict(gkd_args)}", flush=True)

    trainer = GKDTrainer(
        model=args.student_model,
        teacher_model=args.teacher_model,
        args=gkd_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if eval_strategy != "no" else None,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    if eval_strategy != "no":
        metrics = trainer.evaluate()
        print(f"[trl-gkd] final_eval={metrics}", flush=True)


if __name__ == "__main__":
    main()
