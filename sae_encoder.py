import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sae_lens import SAE


def load_finetuned_model(model_path, base_model_id=None, adapter_path=None):
    """
    Load a finetuned model in one of two ways:
      1. Merged model  — provide model_path pointing to a merged checkpoint.
      2. Base + adapter — provide base_model_id + adapter_path for LoRA.
    Falls back to loading model_path as a standalone HF model.
    """
    if adapter_path and base_model_id:
        # LoRA adapter mode: load base model, then apply adapter
        print(f"🔄 Loading base model: {base_model_id}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        print(f"🔗 Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("✅ Adapter merged into base model")
    else:
        # Merged / standalone model mode
        print(f"🔄 Loading finetuned model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def analyze_output_features(
    model_path="./output/gemma-4b/merged",
    base_model_id=None,
    adapter_path=None,
    sae_id="gemma-scope-3-4b-res-12",
    sae_release="gemma-scope-3-4b-pt-res",
    hook_layer=12,
    prompt="List of scientists: Isaac Newton, Albert Einstein, ",
    max_new_tokens=30,
    temperature=0.7,
    top_k_features=3,
):
    """
    Generates text using a FINETUNED model and analyzes output-token
    activations with a pretrained SAE to find features that fired
    during repetitive moments.

    Args:
        model_path:     Path to a merged / finetuned model checkpoint.
        base_model_id:  (optional) HF base model id for LoRA adapter mode.
        adapter_path:   (optional) Path to LoRA adapter dir.
        sae_id:         SAE identifier inside the release.
        sae_release:    SAE release name on SAELens.
        hook_layer:     Transformer layer whose residual stream we tap.
        prompt:         Generation prompt.
        max_new_tokens: How many tokens to generate.
        temperature:    Sampling temperature.
        top_k_features: Number of top SAE features to report.
    """
    # ------------------------------------------------------------------
    # 1. Load finetuned model
    # ------------------------------------------------------------------
    model, tokenizer = load_finetuned_model(model_path, base_model_id, adapter_path)
    device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # 2. Load SAE
    # ------------------------------------------------------------------
    print(f"\n📦 Loading SAE: {sae_release} / {sae_id} ...")
    sae, _, _ = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=str(device),
    )

    # ------------------------------------------------------------------
    # 3. Generate text
    # ------------------------------------------------------------------
    print(f"\n--- Generating from prompt: '{prompt}' ---")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    print(f"Full Output : {full_text}")
    print(f"Generated   : {generated_text}")

    # ------------------------------------------------------------------
    # 4. Capture residual-stream activations via a forward hook
    # ------------------------------------------------------------------
    captured = {}

    def _hook_fn(module, input, output):
        # output is usually a tuple; first element is the hidden state
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden_states"] = hidden.detach()

    # Register hook on the target layer's output
    target_layer = model.model.layers[hook_layer]
    handle = target_layer.register_forward_hook(_hook_fn)

    with torch.no_grad():
        model(output_ids)  # full forward pass to populate the hook

    handle.remove()

    activations = captured["hidden_states"]  # [batch, seq, d_model]

    # Keep only the generated-token positions
    output_activations = activations[:, prompt_len:, :]

    # ------------------------------------------------------------------
    # 5. SAE encoding
    # ------------------------------------------------------------------
    feature_acts = sae.encode(output_activations)  # [batch, output_seq, n_features]

    # ------------------------------------------------------------------
    # 6. Report top features
    # ------------------------------------------------------------------
    print(f"\n--- Analyzing Output Features (layer {hook_layer}) ---")

    top_act_values, top_indices = torch.topk(
        feature_acts.max(dim=1).values[0], k=top_k_features
    )

    for i in range(len(top_indices)):
        feat_idx = top_indices[i].item()
        max_val = top_act_values[i].item()

        peak_pos = feature_acts[0, :, feat_idx].argmax().item()
        actual_token = tokenizer.decode(output_ids[0, prompt_len + peak_pos])

        print(f"  Feature {feat_idx}: Peak Activation {max_val:.4f} on token '{actual_token}'")

    # ------------------------------------------------------------------
    # 7. Mitigation suggestion
    # ------------------------------------------------------------------
    print("\n--- Mitigation Suggestion ---")
    print(
        f"To reduce repetition, try suppressing Feature {top_indices[0].item()} "
        f"at layer {hook_layer} using a steering hook."
    )


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SAE feature analysis on a finetuned model's output."
    )
    parser.add_argument(
        "--model-path",
        default="./output/gemma-4b/merged",
        help="Path to merged / finetuned model (default: ./output/gemma-4b/merged)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="HF base model id (only needed with --adapter-path for LoRA mode)",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Path to LoRA adapter dir (used together with --base-model)",
    )
    parser.add_argument("--sae-id", default="gemma-scope-3-4b-res-12")
    parser.add_argument("--sae-release", default="gemma-scope-3-4b-pt-res")
    parser.add_argument("--hook-layer", type=int, default=12)
    parser.add_argument("--prompt", default="List of scientists: Isaac Newton, Albert Einstein, ")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k-features", type=int, default=3)
    args = parser.parse_args()

    analyze_output_features(
        model_path=args.model_path,
        base_model_id=args.base_model,
        adapter_path=args.adapter_path,
        sae_id=args.sae_id,
        sae_release=args.sae_release,
        hook_layer=args.hook_layer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k_features=args.top_k_features,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()