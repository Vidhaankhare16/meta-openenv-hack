"""
CRISPR Guide RNA Design Environment — Inference Script
=======================================================
MANDATORY environment variables:
    API_BASE_URL      LLM API endpoint (OpenAI-compatible)
    MODEL_NAME        Model identifier
    HF_TOKEN or API_KEY  API key for LLM
    IMAGE_NAME        Docker image name (set by hackathon validator)
    CRISPR_TASK       Task: easy | medium | hard  (default: easy)

STDOUT FORMAT
-------------
[START] task=<task_name> env=crispr_env model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import re
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Path setup — works whether run from repo root or /tmp/workspace/
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from client import CRISPREnv
from models import CRISPRAction

# ---------------------------------------------------------------------------
# Configuration  (matches hackathon sample script variable names exactly)
# ---------------------------------------------------------------------------

IMAGE_NAME   = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.getenv("CRISPR_TASK",  "easy")
BENCHMARK    = "crispr_env"
MAX_STEPS    = 6
TEMPERATURE  = 0.2
MAX_TOKENS   = 512
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Logging helpers (exact format required by hackathon)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    err_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "easy": textwrap.dedent("""
        You are a computational biologist assistant in a CRISPR guide-RNA
        design environment.

        Task EASY: find all NGG PAM sites on the forward strand of a DNA
        sequence. A PAM site is position i where sequence[i+1]=='G' AND
        sequence[i+2]=='G' (0-based, N can be any nucleotide).

        Scan the ENTIRE sequence and list every such position.

        Reply with ONE JSON object — no markdown, no prose:
        {"action_type": "scan_sequence", "pam_positions": [<list of ints>]}
    """).strip(),

    "medium": textwrap.dedent("""
        You are a computational biologist assistant in a CRISPR design
        environment.

        Task MEDIUM: design a guide RNA near a pathogenic mutation.

        STEP 1 — choose a PAM site within ~22 bp UPSTREAM of the mutation:
          {"action_type": "design_guide", "position": <int>}

        STEP 2 — score the returned guide:
          {"action_type": "score_ontarget", "guide": "<20-nt string>"}

        Reply with ONE JSON object per turn. No markdown, no prose.
        The PAM is the NGG triplet; position is the index of N (i+1 and i+2
        are both G). Pick the PAM closest to but not past the mutation.
    """).strip(),

    "hard": textwrap.dedent("""
        You are a computational biologist assistant in a CRISPR design
        environment.

        Task HARD: assess off-target risk for 3 candidate guides.

        STEPS 1-3 — check each guide (index 0, 1, 2):
          {"action_type": "check_offtarget", "guide_index": <0|1|2>}

        STEP 4 — after checking all 3, select the safest:
          {
            "action_type": "select_best",
            "ranking": [<safest>, <middle>, <most_dangerous>],
            "selected_guide_index": <safest>
          }
        Ranking is by off-target count ascending (fewest = safest).

        Reply with ONE JSON object per turn. No markdown, no prose.
    """).strip(),
}

# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Optional[CRISPRAction]:
    """Extract the first valid JSON object from the model response."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    try:
        return CRISPRAction(**json.loads(text))
    except Exception:
        pass

    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return CRISPRAction(**json.loads(match.group()))
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_model_action(client: OpenAI, system_prompt: str,
                     history: List[dict], obs_message: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": obs_message})
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return ""


# ---------------------------------------------------------------------------
# Fallback actions (when LLM output is unparseable)
# ---------------------------------------------------------------------------

def fallback_action(task: str, step: int, obs_message: str) -> CRISPRAction:
    if task == "easy":
        return CRISPRAction(action_type="scan_sequence", pam_positions=[])
    elif task == "medium":
        if step <= 1:
            return CRISPRAction(action_type="design_guide", position=30)
        return CRISPRAction(action_type="score_ontarget",
                            guide="ATCGATCGATCGATCGATCG")
    else:
        if step <= 3:
            return CRISPRAction(action_type="check_offtarget",
                                guide_index=step - 1)
        return CRISPRAction(action_type="select_best",
                            ranking=[0, 1, 2],
                            selected_guide_index=0)


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

async def run_episode(task: str) -> None:
    client    = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    system_prompt = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["easy"])

    # ── Environment connection ───────────────────────────────────────────
    # Hackathon sets IMAGE_NAME; from_docker_image() pulls + starts the
    # container and connects automatically.
    if IMAGE_NAME:
        env = await CRISPREnv.from_docker_image(IMAGE_NAME)
    else:
        # Fallback: connect to an already-running server
        env_url = os.getenv("CRISPR_ENV_URL",
                            "https://vidhaan16-meta-openenv-hack.hf.space")
        env = CRISPREnv(base_url=env_url)

    rewards: List[float] = []
    history: List[dict]  = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result      = await env.reset(task=task)
        obs_message = result.observation.message
        done        = result.done

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            raw_response = get_model_action(
                client, system_prompt, history, obs_message
            )

            action    = parse_action(raw_response)
            error_msg = None
            if action is None:
                error_msg = f"unparseable: {raw_response[:60]!r}"
                action    = fallback_action(task, step, obs_message)
                print(f"[DEBUG] Fallback at step {step}: {action.action_type}",
                      flush=True)

            action_str = json.dumps(action.model_dump(exclude_none=True))
            result     = await env.step(action)

            reward      = result.reward or 0.0
            done        = result.done
            obs_message = result.observation.message

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward,
                     done=done, error=error_msg)

            history.append({"role": "user",      "content": obs_message})
            history.append({"role": "assistant",  "content": raw_response})

            if done:
                break

        score   = min(max(sum(rewards), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score,
                rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_episode(TASK_NAME))
