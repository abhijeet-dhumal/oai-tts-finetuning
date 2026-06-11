#!/usr/bin/env python3
"""Merge a LoRA checkpoint into a standalone final/ model directory."""

import argparse
import logging
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("SOURCE_CHECKPOINT", ""),
        required=not os.environ.get("SOURCE_CHECKPOINT"),
    )
    p.add_argument(
        "--output",
        default=os.environ.get("FINAL_DIR", "/data/orpheus/checkpoints/turkish-v2/final"),
    )
    p.add_argument(
        "--base_model",
        default=os.environ.get("BASE_MODEL", "unsloth/orpheus-3b-0.1-pretrained"),
    )
    p.add_argument("--hf_cache", default=os.environ.get("HF_HOME", "/data/orpheus/hf-cache"))
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.output)
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading tokenizer from %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.hf_cache)
    tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading base model %s", args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.hf_cache,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    if (ckpt / "adapter_config.json").exists():
        log.info("Merging LoRA adapter from %s", ckpt)
        model = PeftModel.from_pretrained(base, str(ckpt))
        model = model.merge_and_unload()
    else:
        log.info("No adapter_config.json — loading full weights from %s", ckpt)
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt),
            cache_dir=args.hf_cache,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    log.info("Saving merged model → %s", out)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    log.info("Done. final/ ready at %s", out)


if __name__ == "__main__":
    main()
