---
title: CRISPR Guide RNA Design Environment
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# CRISPR Guide RNA Design Environment

An AI agent benchmark that simulates the workflow of a computational biologist designing CRISPR gene edits — the same process used in real gene therapy research.

> CRISPR won the 2020 Nobel Prize in Chemistry. This environment turns that science into a fully deterministic RL training benchmark.

---

## Overview

The agent receives a DNA sequence containing a known pathogenic mutation. It must identify valid cut sites, design a guide RNA, and assess whether the edit is safe — mirroring exactly what biologists do in the lab before a clinical trial.

Three tasks of increasing difficulty:

| Task | What the agent does | Max Score |
|------|---------------------|-----------|
| **Easy** | Find all NGG PAM sites (valid cut anchors) in a 90-bp sequence | 1.0 |
| **Medium** | Design a 20-nt guide RNA near a mutation + score on-target efficiency | 1.0 |
| **Hard** | Check off-target binding risk for 3 candidate guides, rank them, pick the safest | 1.0 |

All scoring is **fully deterministic** — PAM sites are rule-based (NGG), efficiency uses Doench Rule Set 2 formulas, off-targets use mismatch counting with seed-region checks. No LLM-as-judge.

---

## Project Structure

```
crispr_env/
├── __init__.py
├── models.py              # CRISPRAction, CRISPRObservation, CRISPRState (Pydantic)
├── client.py              # OpenEnv WebSocket client
├── openenv.yaml           # Submission manifest
├── pyproject.toml
├── requirements.txt
└── server/
    ├── environment.py     # All task logic + scoring functions
    ├── app.py             # FastAPI server (one line)
    └── Dockerfile
inference.py               # LLM agent script (OpenAI-compatible)
```

---

## Available Actions

```
scan_sequence(pam_positions)      →  submit found NGG PAM sites
design_guide(position)            →  get 20-nt guide RNA at a PAM position
score_ontarget(guide)             →  Doench RS2 efficiency score [0–1]
check_offtarget(guide_index)      →  off-target hit count in genome excerpt
select_best(ranking, selected)    →  final safety ranking + recommendation
```

---

## Reward Structure

### Easy — PAM Scanning
```
reward = correct_found / total_true_pams   (recall, max 1.0)
```

### Medium — Guide Design
```
reward = 0.4 (valid guide near mutation) + 0.6 × efficiency_score
```

### Hard — Safety Assessment
```
reward = 0.3 (all 3 guides checked)
       + 0.4 (correct safety ranking)
       + 0.3 (correct guide selected)
```

---

## Running Locally

```bash
# Install dependencies
pip install openenv-core fastapi uvicorn pydantic

# Start the environment server
uvicorn crispr_env.server.app:app --host 0.0.0.0 --port 8000 --reload

# Run the agent (in a separate terminal)
export HF_TOKEN=your_huggingface_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export CRISPR_TASK=easy          # easy | medium | hard
export CRISPR_ENV_URL=http://localhost:8000

python inference.py
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | Hugging Face API token |
| `API_BASE_URL` | Yes | LLM endpoint (default: HF Router) |
| `MODEL_NAME` | Yes | Model identifier |
| `CRISPR_TASK` | No | Task to run: `easy`, `medium`, `hard` (default: `easy`) |
| `CRISPR_ENV_URL` | No | Environment server URL (default: `http://localhost:8000`) |
| `LOCAL_IMAGE_NAME` | No | Docker image name if using `from_docker_image()` |

---

## Docker

```bash
# Build
docker build -t crispr-guide-env -f crispr_env/server/Dockerfile .

# Run
docker run -p 8000:8000 crispr-guide-env
```

---

## Biology Background

| Term | Meaning |
|------|---------|
| **PAM site** | NGG motif where Cas9 anchors before cutting |
| **Guide RNA (gRNA)** | 20-nt sequence that directs Cas9 to the target |
| **On-target efficiency** | How reliably the guide cuts at the intended site |
| **Off-target binding** | Accidental cuts at similar sequences elsewhere in the genome |
| **Doench Rule Set 2** | Published scoring model for gRNA efficiency (Nature Biotech 2016) |

---

## References

- Doench et al. (2016) *Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9*. Nature Biotechnology.
- Hsu et al. (2013) *DNA targeting specificity of RNA-guided Cas9 nucleases*. Nature Biotechnology.
- [OpenEnv Framework](https://github.com/meta-pytorch/OpenEnv)

---

## Author

**Vidhaan Khare** — [github.com/Vidhaankhare16](https://github.com/Vidhaankhare16)
