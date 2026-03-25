"""
QLoRA fine-tuning of Llama-3.2-3B-Instruct on NK risk assessment data.

Input:  data/training/{train,val}.jsonl
Output: models/nk-risk-analyst/ (LoRA adapter weights)
"""

import json
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from config import (
    BASE_MODEL,
    TRAINING_DIR,
    MODELS_DIR,
    LOAD_IN_4BIT,
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_USE_DOUBLE_QUANT,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    NUM_EPOCHS,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    WARMUP_STEPS,
    MAX_SEQ_LENGTH,
    MAX_GRAD_NORM,
)


def load_dataset(split: str) -> Dataset:
    """Load a JSONL dataset split."""
    path = TRAINING_DIR / f"{split}.jsonl"
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return Dataset.from_list(examples)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {BASE_MODEL}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )

    # Load model
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    # LoRA config
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = load_dataset("train")
    val_dataset = load_dataset("val")
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val: {len(val_dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        bf16=True,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=5,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    # Add max_seq_length to training args (newer trl versions)
    training_args.max_seq_length = MAX_SEQ_LENGTH

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max seq length: {MAX_SEQ_LENGTH}")

    trainer.train()

    # Save adapter
    print(f"\nSaving LoRA adapter to {MODELS_DIR}...")
    trainer.save_model(str(MODELS_DIR))
    tokenizer.save_pretrained(str(MODELS_DIR))

    # Save training metrics
    metrics = trainer.state.log_history
    with open(MODELS_DIR / "training_log.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTraining complete!")

    # Print final metrics
    train_losses = [m["loss"] for m in metrics if "loss" in m]
    eval_losses = [m["eval_loss"] for m in metrics if "eval_loss" in m]
    if train_losses:
        print(f"  Final train loss: {train_losses[-1]:.4f}")
    if eval_losses:
        print(f"  Final eval loss: {eval_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
