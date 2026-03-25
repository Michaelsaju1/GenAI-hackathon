"""
Before/after comparison: base model vs fine-tuned model on test set.

Input:  data/training/test.jsonl, models/nk-risk-analyst/
Output: data/evaluation_results.json
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config import BASE_MODEL, TRAINING_DIR, MODELS_DIR, DATA_DIR, MAX_SEQ_LENGTH


def load_test_set() -> list[dict]:
    """Load test examples."""
    examples = []
    with open(TRAINING_DIR / "test.jsonl") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def extract_input_prompt(example: dict) -> list[dict]:
    """Extract just the system + user messages (no assistant response)."""
    return [m for m in example["messages"] if m["role"] != "assistant"]


def generate_response(model, tokenizer, messages: list[dict], max_new_tokens: int = 1024) -> str:
    """Generate a response from the model given chat messages."""
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def try_parse_json(text: str) -> dict | None:
    """Try to parse JSON from model output, handling code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def evaluate_response(predicted: dict | None, ground_truth: dict) -> dict:
    """Compute metrics for a single prediction."""
    if predicted is None:
        return {"valid_json": False, "field_completeness": 0.0, "escalation_error": None}

    # Field completeness
    required = ["escalation_level", "escalation_rationale", "situation_summary",
                "key_actors", "watch_indicators", "confidence_level"]
    present = sum(1 for f in required if f in predicted and predicted[f])
    completeness = present / len(required)

    # Escalation error
    esc_error = None
    if "escalation_level" in predicted and "escalation_level" in ground_truth:
        try:
            esc_error = abs(int(predicted["escalation_level"]) - int(ground_truth["escalation_level"]))
        except (ValueError, TypeError):
            pass

    return {
        "valid_json": True,
        "field_completeness": completeness,
        "escalation_error": esc_error,
    }


def main():
    print("Loading test set...")
    test_examples = load_test_set()
    print(f"  {len(test_examples)} test examples")

    # Load tokenizer
    print(f"\nLoading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Load fine-tuned model (base + LoRA adapter)
    print(f"Loading LoRA adapter from {MODELS_DIR}...")
    finetuned_model = PeftModel.from_pretrained(base_model, str(MODELS_DIR))

    results = []

    for i, example in enumerate(test_examples):
        print(f"\n{'='*60}")
        print(f"Test example {i+1}/{len(test_examples)}: {example['messages'][1]['content'][:80]}...")

        messages = extract_input_prompt(example)
        ground_truth = json.loads(example["messages"][-1]["content"])

        # Base model response
        print("\n  [BASE MODEL] Generating...", flush=True)

        # Temporarily disable adapter for base model inference
        finetuned_model.disable_adapter_layers()
        base_response = generate_response(finetuned_model, tokenizer, messages)
        finetuned_model.enable_adapter_layers()

        base_parsed = try_parse_json(base_response)
        base_metrics = evaluate_response(base_parsed, ground_truth)

        # Fine-tuned model response
        print("  [FINE-TUNED] Generating...", flush=True)
        ft_response = generate_response(finetuned_model, tokenizer, messages)
        ft_parsed = try_parse_json(ft_response)
        ft_metrics = evaluate_response(ft_parsed, ground_truth)

        result = {
            "week_start": example["messages"][1]["content"].split("\n")[0] if example["messages"][1]["content"] else "",
            "ground_truth": ground_truth,
            "base_model": {
                "raw_output": base_response[:2000],
                "parsed_json": base_parsed,
                "metrics": base_metrics,
            },
            "finetuned_model": {
                "raw_output": ft_response[:2000],
                "parsed_json": ft_parsed,
                "metrics": ft_metrics,
            },
        }
        results.append(result)

        # Print comparison
        gt_esc = ground_truth.get("escalation_level", "?")
        base_esc = base_parsed.get("escalation_level", "N/A") if base_parsed else "INVALID JSON"
        ft_esc = ft_parsed.get("escalation_level", "N/A") if ft_parsed else "INVALID JSON"

        print(f"\n  Ground truth escalation: {gt_esc}")
        print(f"  Base model escalation:   {base_esc} (valid_json={base_metrics['valid_json']}, "
              f"completeness={base_metrics['field_completeness']:.0%})")
        print(f"  Fine-tuned escalation:   {ft_esc} (valid_json={ft_metrics['valid_json']}, "
              f"completeness={ft_metrics['field_completeness']:.0%})")

    # Save results
    output_path = DATA_DIR / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")

    # Aggregate metrics
    print(f"\n{'='*60}")
    print("AGGREGATE METRICS")
    print(f"{'='*60}")

    for model_name, key in [("Base Model", "base_model"), ("Fine-tuned Model", "finetuned_model")]:
        metrics = [r[key]["metrics"] for r in results]
        valid_json = sum(1 for m in metrics if m["valid_json"])
        avg_completeness = sum(m["field_completeness"] for m in metrics) / len(metrics)
        esc_errors = [m["escalation_error"] for m in metrics if m["escalation_error"] is not None]
        avg_esc_error = sum(esc_errors) / len(esc_errors) if esc_errors else float("nan")

        print(f"\n  {model_name}:")
        print(f"    Valid JSON: {valid_json}/{len(metrics)} ({100*valid_json/len(metrics):.0f}%)")
        print(f"    Avg field completeness: {avg_completeness:.0%}")
        print(f"    Escalation MAE: {avg_esc_error:.2f}" if esc_errors else "    Escalation MAE: N/A")


if __name__ == "__main__":
    main()
