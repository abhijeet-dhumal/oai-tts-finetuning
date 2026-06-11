#!/usr/bin/env python3
"""Upload merged Orpheus Turkish fine-tune + eval artifacts to Hugging Face Hub."""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ALL_PHRASE_IDS = (
    "flight_announce", "welcome", "safety", "farewell", "weather",
    "news_intro", "directions", "question", "tech", "emergency",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/data/orpheus/checkpoints/turkish-v2/final"))
    p.add_argument("--eval-dir", default=os.environ.get("EVAL_DIR", "/data/orpheus/eval-v2"))
    p.add_argument("--readme", default=os.environ.get("README_PATH", "/orpheus-scripts/hf-README.md"))
    p.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID", "AbDhumal/orpheus-3b-turkish-tts-v2"))
    p.add_argument("--private", action="store_true", default=os.environ.get("HF_PRIVATE", "").lower() == "true")
    p.add_argument("--skip-weights", action="store_true")
    p.add_argument("--readme-only", action="store_true", help="Upload model card only")
    p.add_argument("--eval-only", action="store_true", help="Upload eval/ tree only (no weights)")
    p.add_argument("--prune-audio", action="store_true",
                   help="Delete eval/audio/* phrase dirs not in included_phrases.json")
    p.add_argument("--included-phrases", default="",
                   help="JSON list of phrase ids to keep on HF (default: eval_dir/included_phrases.json)")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    eval_dir = Path(args.eval_dir)
    readme = Path(args.readme)

    if (
        not args.skip_weights
        and not args.readme_only
        and not args.eval_only
        and not (model_dir / "model.safetensors.index.json").exists()
    ):
        raise FileNotFoundError(f"No merged weights in {model_dir}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private, token=token)

    if readme.exists():
        print(f"Uploading model card → README.md", flush=True)
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Add model card with eval benchmarks",
        )

    if (not args.readme_only or args.eval_only) and eval_dir.is_dir() and any(eval_dir.iterdir()):
        print(f"Uploading eval artifacts from {eval_dir} …", flush=True)
        api.upload_folder(
            folder_path=str(eval_dir),
            path_in_repo="eval",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Update curated eval WAVs and results",
        )

    if args.prune_audio:
        import json as _json
        inc_path = Path(args.included_phrases) if args.included_phrases else eval_dir / "included_phrases.json"
        keep = set(_json.loads(inc_path.read_text())) if inc_path.exists() else set()
        for pid in ALL_PHRASE_IDS:
            if pid not in keep:
                repo_path = f"eval/audio/{pid}"
                try:
                    api.delete_folder(
                        repo_id=args.repo_id,
                        path_in_repo=repo_path,
                        repo_type="model",
                        commit_message=f"Remove low-quality sample {pid}",
                    )
                    print(f"Pruned {repo_path}", flush=True)
                except Exception as e:
                    print(f"Prune skip {repo_path}: {e}", flush=True)
        try:
            api.delete_folder(
                repo_id=args.repo_id,
                path_in_repo="eval/checkpoint-8000-run",
                repo_type="model",
                commit_message="Remove stale v2 eval artifacts",
            )
            print("Pruned eval/checkpoint-8000-run", flush=True)
        except Exception as e:
            print(f"Prune skip checkpoint-8000-run: {e}", flush=True)

    if not args.skip_weights and not args.readme_only and not args.eval_only:
        print(f"Uploading weights from {model_dir} → {args.repo_id} …", flush=True)
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload Turkish fine-tuned Orpheus-3B weights (merged LoRA, v2)",
            ignore_patterns=["optimizer.pt", "rng_state*", "scheduler.pt", "training_args.bin"],
        )

    print(f"Done: https://huggingface.co/{args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
