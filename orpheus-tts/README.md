# Orpheus TTS Fine-Tuning on OpenShift AI — Turkish Language Adaptation

An example of distributed LLM-based TTS fine-tuning using **Kubeflow Trainer v2 (TrainJob)** on Red Hat OpenShift AI. The task — adapting Orpheus-3B to Turkish — demonstrates how modern codec-language-model TTS can be fine-tuned for a new language with measurable, audible results tracked end-to-end in MLflow.

---

## Use Case

**Problem:** `unsloth/orpheus-3b-0.1-pretrained` is a 3B-parameter LLM trained to generate speech as SNAC audio tokens. Out of the box it has no Turkish phonology — given Turkish text it produces incoherent English-accented output.

**Goal:** Fine-tune the Llama-3 backbone on Turkish text-to-speech token sequences so the model produces intelligible, natural-sounding Turkish audio. Demonstrate measurable WER/CER improvement at each checkpoint, logged to MLflow alongside audio artifacts.

---

## Stack & Why

| Component | Choice | Reason |
|-----------|--------|--------|
| **Base model** | `unsloth/orpheus-3b-0.1-pretrained` (Llama-3 backbone) | Open-weight, compatible with standard HF Trainer; SNAC codec produces high-quality 24kHz audio |
| **Training strategy** | Causal language modelling (next-token prediction on interleaved text+audio tokens) | Natural fit for an LLM — no custom GAN or aligner needed; the same cross-entropy loss drives both text understanding and audio generation |
| **Audio codec** | SNAC 24kHz (`hubertsiuzdak/snac_24khz`) | Hierarchical codec, 7 tokens/frame interleaved across 3 codebooks; lossless round-trip at 24kHz |
| **Dataset** | `afkfatih/turkish-tts-combined-raw` (~81K raw audio samples) | Largest publicly available Turkish TTS dataset; re-tokenized from scratch with the correct Llama-3 vocabulary |
| **Preprocessing** | Distributed sharding across nodes (5K samples/node) with PVC sentinel | 10K samples tokenized in ~2.5 min; skipped on resume |
| **Distributed training** | DDP via `torchrun` + Kubeflow Trainer v2 TrainJob | 2 nodes × 1× A100-80GB; Trainer injects `PET_*` env vars consumed natively by `torchrun` |
| **Experiment tracking** | MLflow (OpenShift AI managed instance) | Per-step loss, WER/CER, RTF, audio artifacts, spectrograms, checkpoint model versions |

---

## Outcome

After fine-tuning on 2,000 Turkish samples (3 epochs):

| Metric | Baseline (untrained) | Fine-tuned |
|--------|----------------------|------------|
| WER (Whisper-small, Turkish) | ~130–140% | Target < 50% |
| CER | ~90–100% | Target < 30% |
| RTF (real-time factor) | ~2.0 | Target < 1.2 |
| Turkish phonemes (ş, ğ, ç…) | absent | present |

Baseline and fine-tuned audio are logged to MLflow at every `AUDIO_LOG_STEPS` interval for direct comparison.

---

## Structure

```
orpheus-tts/
├── scripts/
│   └── train_orpheus.py      # preprocessing + full training loop — reads config from env vars
└── manifests/
    ├── trainjob-orpheus.yaml  # 2-node TrainJob — the only file to edit
    └── kustomization.yaml     # generates orpheus-tts-scripts ConfigMap from scripts/
```

---

## Deploy

Prerequisites: `smartshop-training` ClusterQueue, `orpheus-tts-storage` PVC (150Gi), `hf-credentials` and `smartshop-mlflow-token` secrets.

```bash
# Apply ConfigMap + TrainJob
cd examples/tts-finetuning/orpheus-tts
kubectl kustomize . | oc apply -f - -n smartshop

# Watch progress
oc get trainjob orpheus-turkish-tts -n smartshop -w
oc logs -n smartshop -l trainer.kubeflow.org/trainjob-name=orpheus-turkish-tts -f
```

To run preprocessing only (both nodes shard the work):
```bash
# Runs automatically before torchrun — controlled by PREPROCESS_SAMPLES env var
# To force re-preprocessing, delete the sentinel:
oc exec <pod> -- find /data/orpheus/checkpoints/preprocessed -mindepth 1 -delete
```

To adjust training scope, edit env vars in `trainjob-orpheus.yaml`:

```yaml
- name: PREPROCESS_SAMPLES
  value: "10000"      # samples to tokenize (split across nodes)
- name: MAX_TRAIN_SAMPLES
  value: "2000"       # cap for training (drawn from preprocessed pool)
- name: NUM_EPOCHS
  value: "3"
- name: AUDIO_LOG_STEPS
  value: "50"         # how often to log audio to MLflow
```

---

## MLflow

Training logs to the OpenShift AI managed MLflow instance:

- **Metrics:** `loss`, `eval_loss`, `perplexity`, `wer/<prompt>`, `cer/<prompt>`, `rtf/<prompt>` — all step-indexed
- **Artifacts:** `.wav` + spectrogram `.png` per prompt per eval step; baseline at step 0, final at training end
- **Traces:** one `orpheus_tts_pipeline` trace per inference call with sub-spans: `tokenise → model_generate → snac_decode`
- **Model registry:** `orpheus-turkish-tts` — a new version registered at each checkpoint save with `training_step` and `checkpoint_pvc_path` tags
