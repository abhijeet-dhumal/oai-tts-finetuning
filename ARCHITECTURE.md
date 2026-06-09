# Architecture — Turkish TTS Fine-Tuning on Red Hat OpenShift AI

## Overview

This example demonstrates how to run distributed model fine-tuning on **Red Hat OpenShift AI** using the **Kubeflow Trainer v2 TrainJob** primitive. The workload is a full GAN fine-tuning of a Text-to-Speech model for Turkish language adaptation — a task that is concrete, measurable, and directly representative of real-world language localisation use cases.

The pattern is general-purpose: swap the base model and dataset and the same manifest and scripts template applies to any distributed fine-tuning workload.

---

## Use Case

`facebook/mms-tts-eng` (VITS architecture, Apache-2.0) produces English phonetics when given Turkish text. Turkish has sounds absent in English — ş, ğ, ç, ü, ö, ı — and distinct prosody. The goal is to adapt the model's internal linguistic representations to Turkish using native Turkish speech data, producing intelligible, natural-sounding Turkish audio.

---

## Model Architecture

The base model is **VITS** (Variational Inference with adversarial learning for end-to-end Text-to-Speech, Kim et al., ICML 2021). It is a fully end-to-end neural TTS model that learns phoneme alignment, duration, acoustics, and waveform synthesis in a single model with no external aligner.

```
Turkish text
      │
      ▼
 Text Encoder                ← fine-tuned on Turkish
      │
      ▼
 Stochastic Duration Predictor ← fine-tuned
      │
      ▼
 Normalizing Flow (GlowTTS)   ← fine-tuned
      │
      ▼
 HiFi-GAN Vocoder             ← frozen (weights preserved)
      │
      ▼
 Audio output @ 16 kHz

 ─── training only ───────────────────────────
 Real audio → Posterior Encoder → latent z_q  ← fine-tuned
```

The HiFi-GAN vocoder is frozen throughout training. It is entirely language-agnostic — it maps latent acoustic representations to waveform samples. Freezing it preserves the pretrained synthesis quality and halves the number of trainable parameters.

---

## Training Pipeline

Training runs as a **full GAN**: the text encoder, posterior encoder, and normalizing flow are optimised jointly against both a reconstruction loss and an adversarial discriminator (Multi-Period + Multi-Scale). The discriminator is randomly initialised and trained from scratch on Turkish data.

### Loss function

| Term | Weight | Role |
|------|--------|------|
| Mel reconstruction L1 | ×45 | Perceptual audio fidelity |
| KL divergence | ×1.0 | Aligns posterior with text prior |
| Generator adversarial | ×1.0 | Forces natural waveform statistics |
| Feature matching | ×1.0 | Stabilises GAN convergence |

Weights match the original VITS paper exactly (Kim et al. 2021, Table 1).

### Optimiser

AdamW with `lr=2e-4`, `eps=1e-9`, `ExponentialLR(gamma=0.999875)` stepped once per epoch — the schedule from the original VITS codebase.

Separate `GradScaler` instances for the generator and discriminator enable stable FP16 mixed-precision training with two independent optimisers.

### Monotonic Alignment Search (MAS)

Text-to-spectrogram alignment is computed via MAS (from Glow-TTS, Jeong et al. 2020) on the CPU as a vectorised NumPy operation per batch. Batches where audio is shorter than the text sequence are skipped automatically.

---

## Data Pipeline

**Dataset:** `afkfatih/turkish-tts-combined-raw` — ~81,000 samples, 7 speakers, native Turkish studio recordings (HuggingFace).

**Preprocessing** (`preprocess_mms.py`):
1. Expand the VITS tokenizer vocabulary with Turkish characters (ş, ğ, ç, ü, ö, ı, â, î, û)
2. Resample all audio to 16,000 Hz (model's native sample rate)
3. Write a sentinel file to the shared PVC — preprocessing is skipped on all subsequent job restarts

---

## Distributed Training Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Red Hat OpenShift AI                                                │
│                                                                      │
│  ┌────────────────────────┐    ┌────────────────────────┐            │
│  │  TrainJob Pod node-0   │    │  TrainJob Pod node-1   │            │
│  │  1× GPU                │    │  1× GPU                │            │
│  │                        │    │                        │            │
│  │  pip install (deps)    │    │  pip install (deps)    │            │
│  │  torchrun  rank 0      │◄──►│  torchrun  rank 1      │  NCCL DDP  │
│  │    train_vits.py       │    │    train_vits.py       │            │
│  └──────────┬─────────────┘    └──────────┬─────────────┘            │
│             └──────────────┬──────────────┘                          │
│                            │ shared PVC                              │
│               ┌────────────▼────────────┐                            │
│               │  /data/tts/              │                            │
│               │  ├─ mms-dataset/         │  preprocessed Arrow shards │
│               │  ├─ hf-cache/            │  model weights cache       │
│               │  └─ checkpoints/         │  best / final safetensors  │
│               └─────────────────────────┘                            │
│                                                                      │
│  Kubeflow Trainer v2 operator                                        │
│  · Provisions pods + headless service for DNS rendezvous             │
│  · Injects PET_MASTER_ADDR / PET_MASTER_PORT / PET_NNODES /         │
│    PET_NODE_RANK — consumed natively by torchrun                     │
│  · Kueue ClusterQueue enforces GPU quota per namespace               │
│                                                                      │
│  MLflow (managed instance)                                           │
│  · Metrics per step: mel_loss, kl_loss, gen_adv, disc_loss          │
│  · Audio artifacts every eval interval (baseline / fine-tuned /     │
│    reference), mel spectrograms, loss curves                         │
│  · Final checkpoint registered in Model Registry                     │
└──────────────────────────────────────────────────────────────────────┘
```

### Kubeflow Trainer v2 — TrainJob

TrainJob is the Kubeflow Trainer v2 CRD for running distributed training on Kubernetes. It handles:

- Multi-node pod provisioning with a headless service for DNS-based rendezvous
- Environment injection (`PET_*` vars) read natively by `torchrun` — no custom launcher wrapper needed
- Integration with **Kueue** for GPU quota enforcement and job queuing
- Failure policy and pod restart management

Training scripts are served from a **ConfigMap** (`tts-mms-scripts`) built by Kustomize from the local `scripts/` directory. Scripts are updated with a single `oc apply -k` without rebuilding any container image.

---

## Evaluation

`evaluate.py` runs on rank 0 after training completes. It generates audio for a fixed set of Turkish test sentences and computes:

| Metric | Tool | What it measures |
|--------|------|-----------------|
| WER — Word Error Rate | Whisper large-v3 (Turkish) | Intelligibility: can a Turkish ASR model understand the output? |
| CER — Character Error Rate | same | Character-level accuracy (more stable for agglutinative Turkish) |
| MCD — Mel Cepstral Distortion | librosa MFCC | Spectral proximity to the reference Turkish model |
| RTF — Real-Time Factor | wall-clock / audio duration | Inference speed; < 1.0 = real-time capable |

Three audio tracks are logged to MLflow per test sentence: **baseline** (unmodified `mms-tts-eng`), **fine-tuned** (current run), and **reference** (`mms-tts-tur`, the native Turkish model).

---

## File Structure

```
examples/tts-finetuning/
├── scripts/
│   ├── preprocess_mms.py     # tokenizer expansion + waveform extraction
│   ├── train_vits.py         # full VITS GAN training loop
│   └── evaluate.py           # WER / CER / MCD / RTF + MLflow artifacts
├── manifests/
│   ├── trainjob-tts.yaml     # 2-node TrainJob — the only file to edit
│   └── kustomization.yaml    # builds tts-mms-scripts ConfigMap from scripts/
└── ARCHITECTURE.md
```

---

## References

1. Kim, J. et al. — "VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech," ICML 2021.
   [https://arxiv.org/abs/2106.06103](https://arxiv.org/abs/2106.06103)

2. Pratap, V. et al. — "Scaling Speech Technology to 1,000+ Languages" (MMS), Meta AI, 2023.
   [https://arxiv.org/abs/2305.13516](https://arxiv.org/abs/2305.13516)

3. Lacombe, Y. — "Fine-tune VITS / MMS models for language adaptation," HuggingFace, 2023.
   [https://huggingface.co/blog/ylacombe/finetune-hf-vits](https://huggingface.co/blog/ylacombe/finetune-hf-vits)

4. Kong, J. et al. — "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis," NeurIPS 2020.
   [https://arxiv.org/abs/2010.05646](https://arxiv.org/abs/2010.05646)

5. Jeong, J. et al. — "Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search," NeurIPS 2020.
   [https://arxiv.org/abs/2005.11129](https://arxiv.org/abs/2005.11129)

6. Radford, A. et al. — "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), OpenAI, 2022.
   [https://arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)

7. Kubeflow Trainer v2 — TrainJob API reference.
   [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
