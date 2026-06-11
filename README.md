# Orpheus Turkish TTS — OpenShift AI / Kubeflow Trainer v2

Fine-tune [unsloth/orpheus-3b-0.1-pretrained](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained) for **Turkish text-to-speech** using LoRA on a SNAC 24 kHz codec vocabulary. Trained on Red Hat OpenShift AI with **Kubeflow Trainer v2 (TrainJob)**.

| Deliverable | Link |
|-------------|------|
| **Fine-tuned weights** | [AbDhumal/orpheus-3b-turkish-tts-v2](https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2) |
| **Model card source** | [`huggingface/README.md`](huggingface/README.md) |
| **Architecture deep-dive** | [`ARCHITECTURE.md`](ARCHITECTURE.md) |

---

## Results (v2 training · eval-v4)

| Metric | Baseline (English pretrained) | Fine-tuned (`final/`) | Δ |
|--------|-------------------------------|------------------------|---|
| **WER mean** | 1.58 | **0.72** | −0.85 |
| **CER mean** | 1.22 | **0.41** | −0.81 |
| **eval_loss** (training) | — | **4.35** | best @ step 8,800 |

Training: 2× A100-80GB · 20K samples · 8 epochs · 9,504 steps · LoRA r=32 α=64 · merged from checkpoint-8800.

Audio samples and charts live on the [HF model card](https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2#audio-samples). Pushing this repo to GitHub does **not** upload weights — use `manifests/job-hf-upload.yaml` on the cluster.

---

## Repository layout

```
├── scripts/
├── manifests/
├── huggingface/          # model card + eval charts + MLflow UI screenshots
├── kustomization.yaml
├── ARCHITECTURE.md
└── README.md
```

> **Archive:** prior multi-example layout (`orpheus-tts/` + `vits-tts/`) → [`archive/multi-example-layout`](https://github.com/abhijeet-dhumal/oai-tts-finetuning/tree/archive/multi-example-layout)

---

## Prerequisites

- OpenShift AI cluster with Kubeflow Trainer v2 (`TrainJob` CRD)
- Kueue `ClusterQueue` (example uses `smartshop-training`)
- GPU nodes with `nvidia.com/gpu`
- Secrets: `hf-credentials` (HuggingFace token), `smartshop-mlflow-token`
- Runtime: `torch-distributed-cuda130-torch210-py312`

Adjust `namespace`, queue name, and MLflow URI in manifests for your environment.

---

## Deploy

### 1. ConfigMap + PVC

```bash
git clone https://github.com/abhijeet-dhumal/oai-tts-finetuning.git
cd oai-tts-finetuning
kubectl kustomize . | oc apply -f - -n <namespace>
```

### 2. Train

```bash
oc apply -f manifests/trainjob-orpheus.yaml -n <namespace>
oc get trainjob orpheus-turkish-tts-v2 -n <namespace> -w
```

Key env vars are in `trainjob-orpheus.yaml` (`LEARNING_RATE=2e-5`, `NUM_EPOCHS=8`, `MAX_TRAIN_SAMPLES=20000`, `LORA_R=32`, …).

### 3. Merge best checkpoint

```bash
# On a GPU pod with PVC mounted:
python /orpheus-scripts/save_final_orpheus.py
# → /data/orpheus/checkpoints/turkish-v2/final/
```

### 4. Post-train eval

```bash
oc apply -f manifests/trainjob-orpheus-eval.yaml -n <namespace>
# Output: /data/orpheus/eval-v4-final/
```

### 5. Publish to Hugging Face (optional)

```bash
oc create configmap orpheus-hf-card -n <namespace> \
  --from-file=hf-README.md=huggingface/README.md \
  --from-file=results.json=huggingface/eval/results.json \
  --dry-run=client -o yaml | oc apply -f -

oc apply -f manifests/job-hf-upload.yaml -n <namespace>
```

Set `UPLOAD_MODE` in the job manifest: `eval` (default), `full` (weights + eval), or `readme`. Requires `HF_TOKEN` with write access (`HF_REPO_ID` env).

---

## MLflow

Experiment: `orpheus-turkish-tts`

| Run | ID | Notes |
|-----|-----|-------|
| Training v2 | `6804b44335f347849f26da2736aa73df` | loss, in-train audio, WER/CER |
| Eval v4 | `50c177d696164b4a83169292b2109b4e` | 10-sentence benchmark |

Static chart exports + MLflow UI screenshots: [`huggingface/eval/mlflow/`](huggingface/eval/mlflow/)

| Screenshot | What it shows |
|------------|---------------|
| [ui_traces_pipeline](huggingface/eval/mlflow/ui_traces_pipeline.png) | Traces: `orpheus_tts_pipeline` spans, RTF, duration |
| [ui_artifacts_audio](huggingface/eval/mlflow/ui_artifacts_audio.png) | Artifacts: step audio + waveform preview |

---

## Updating scripts on the cluster

After editing Python files locally:

```bash
kubectl kustomize . | oc apply -f - -n <namespace>
# Re-apply TrainJob / Job to pick up new ConfigMap
```

---

## Related

- Hugging Face model: [AbDhumal/orpheus-3b-turkish-tts-v2](https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2)
