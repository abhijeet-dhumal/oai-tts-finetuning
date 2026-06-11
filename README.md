# TTS Fine-Tuning on Red Hat OpenShift AI

Distributed Text-to-Speech fine-tuning examples using **Kubeflow Trainer v2** on Red Hat OpenShift AI. Two complementary approaches for adapting TTS models to a new language (Turkish), each showcasing a different generation paradigm.

---

## Examples

### [`orpheus-tts/`](./orpheus-tts/)

Fine-tunes **Orpheus-3B** — a codec language model (Llama-3 backbone) that generates speech as SNAC audio tokens. Training is standard next-token prediction with HuggingFace `Trainer`. No custom GAN or aligner needed.

- **Model:** `unsloth/orpheus-3b-0.1-pretrained` (3.3B params, bfloat16)
- **Output:** 24kHz natural-sounding speech via SNAC codec
- **Training:** Causal LM cross-entropy, DDP across 2× A100-80GB
- **Preprocessing:** Distributed SNAC tokenization (5K samples/node in parallel)

→ [README](./orpheus-tts/README.md) · [Architecture](./orpheus-tts/ARCHITECTURE.md) · [HF model](https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2)

---

### [`vits-tts/`](./vits-tts/)

Fine-tunes **MMS-TTS** (VITS architecture) — a GAN-based end-to-end TTS model with an integrated HiFi-GAN vocoder. Trains text encoder, posterior encoder, and normalizing flow against both reconstruction and adversarial losses.

- **Model:** `facebook/mms-tts-eng` (36M params, full GAN)
- **Output:** 16kHz speech with naturalness from adversarial training
- **Training:** Generator + discriminator with separate GradScalers, DDP across 2× GPU
- **Preprocessing:** Tokenizer vocabulary expansion + audio resampling (sentinel-guarded)

→ [README](./vits-tts/README.md) · [Architecture](./vits-tts/ARCHITECTURE.md)

---

## Common Infrastructure

Both examples share the same Kubeflow Trainer v2 pattern:

| Layer | Detail |
|-------|--------|
| **Orchestration** | `TrainJob` CRD — provisions pods, headless service, injects `PET_*` env vars for `torchrun` |
| **GPU quota** | Kueue `ClusterQueue` enforces per-namespace limits |
| **Scripts** | Served from a `ConfigMap` built by Kustomize — no image rebuild on script changes |
| **Storage** | Shared PVC for model cache, preprocessed dataset, and checkpoints |
| **Tracking** | OpenShift AI managed MLflow — metrics, audio artifacts, spectrograms, model registry |
| **Secrets** | HuggingFace token + MLflow token from cluster secrets via `secretKeyRef` |

## Deploy Either Example

```bash
# Orpheus TTS
cd orpheus-tts
kubectl kustomize . | oc apply -f - -n smartshop

# VITS / MMS TTS
cd vits-tts
kubectl kustomize . | oc apply -f - -n smartshop
```
