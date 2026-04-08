"""
CRISPR Guide RNA Design Environment — Inference Script
=======================================================
Required environment variables (set by hackathon validator):
    IMAGE_NAME    Docker image for the environment container
    API_BASE_URL  LLM endpoint  (default: HF Router)
    MODEL_NAME    Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      API key

Optional:
    CRISPR_TASK        easy | medium | hard  (default: easy)
    CRISPR_ENV_URL     Override server URL when IMAGE_NAME is not set

STDOUT FORMAT (required by validator):
    [START] task=<t> env=<e> model=<m>
    [STEP]  step=<n> action=<a> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
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
# Path setup — works from /tmp/workspace/ (hackathon) or local repo root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from client import CRISPREnv          # noqa: E402
from models import CRISPRAction        # noqa: E402

# ---------------------------------------------------------------------------
# Config  (variable names match the hackathon sample script exactly)
# ---------------------------------------------------------------------------
IMAGE_NAME   = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN")  or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.getenv("CRISPR_TASK",  "easy")
BENCHMARK    = "crispr_env"
MAX_STEPS    = 6
TEMPERATURE  = 0.2
MAX_TOKENS   = 512
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Required stdout logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "easy": textwrap.dedent("""
        You are a computational biologist in a CRISPR guide-RNA design tool.

        TASK EASY: find all NGG PAM sites on the forward strand.
        A PAM site is position i where sequence[i+1]=='G' AND sequence[i+2]=='G'.
        (N = any nucleotide; i is 0-based index of N in the NGG triplet.)

        Scan the ENTIRE sequence and list every position.
        Reply with ONE JSON object only — no markdown, no text:
        {"action_type": "scan_sequence", "pam_positions": [<list of ints>]}
    """).strip(),

    "medium": textwrap.dedent("""
        You are a computational biologist in a CRISPR design tool.

        TASK MEDIUM: design a guide RNA near a pathogenic mutation.

        Step 1 — choose a PAM position within ~22 bp upstream of the mutation:
          {"action_type": "design_guide", "position": <int>}

        Step 2 — score the returned guide with:
          {"action_type": "score_ontarget", "guide": "<20-nt string>"}

        Reply with ONE JSON object per turn. No markdown, no prose.
        A PAM position i is valid when sequence[i+1]=='G' and sequence[i+2]=='G'.
    """).strip(),

    "hard": textwrap.dedent("""
        You are a computational biologist in a CRISPR design tool.

        TASK HARD: find the safest guide RNA from 3 candidates.

        Steps 1–3 — check each guide (index 0, 1, 2) for off-targets:
          {"action_type": "check_offtarget", "guide_index": <0|1|2>}

        Step 4 — rank by safety (fewest off-targets = safest) and select:
          {"action_type": "select_best",
           "ranking": [<safest_index>, <mid_index>, <dangerous_index>],
           "selected_guide_index": <safest_index>}

        Reply with ONE JSON object per turn. No markdown, no prose.
    """).strip(),
}

# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Optional[CRISPRAction]:
    text = re.sub(r"```(?:json)?\s*", "", text.strip()).strip().rstrip("`").strip()
    try:
        return CRISPRAction(**json.loads(text))
    except Exception:
        pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return CRISPRAction(**json.loads(m.group()))
        except Exception:
            pass
    return None


def fallback_action(task: str, step: int) -> CRISPRAction:
    if task == "easy":
        return CRISPRAction(action_type="scan_sequence", pam_positions=[])
    if task == "medium":
        if step <= 1:
            return CRISPRAction(action_type="design_guide", position=30)
        return CRISPRAction(action_type="score_ontarget",
                            guide="GCATCGATCGATCGATCGAT")
    # hard
    if step <= 3:
        return CRISPRAction(action_type="check_offtarget", guide_index=step - 1)
    return CRISPRAction(action_type="select_best",
                        ranking=[1, 2, 0], selected_guide_index=1)

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_llm_action(client: OpenAI, system: str, history: List[dict],
                   obs: str) -> str:
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": obs})
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS, stream=False,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return ""

# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

async def run_episode(task: str) -> None:
    rewards:     List[float] = []
    history:     List[dict]  = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    env                      = None

    # [START] must be emitted before any other output
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        sys_prompt = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["easy"])

        # ── Connect to environment ───────────────────────────────────────
        if IMAGE_NAME:
            # Hackathon path: validator sets IMAGE_NAME, we start the container
            env = await CRISPREnv.from_docker_image(IMAGE_NAME)
        else:
            # Dev/fallback path: connect to a live Space
            env_url = os.getenv(
                "CRISPR_ENV_URL",
                "https://vidhaan16-meta-openenv-hack.hf.space",
            )
            env = CRISPREnv(base_url=env_url)
            await env.connect()   # ← must be called explicitly for URL path

        # ── Reset ────────────────────────────────────────────────────────
        result      = await env.reset(task=task)
        obs_message = result.observation.message
        done        = result.done

        # ── Step loop ────────────────────────────────────────────────────
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            raw      = get_llm_action(llm, sys_prompt, history, obs_message)
            action   = parse_action(raw)
            err_msg  = None

            if action is None:
                err_msg = f"parse_error:{raw[:40]!r}"
                action  = fallback_action(task, step)
                print(f"[DEBUG] fallback at step {step}: {action.action_type}",
                      flush=True)

            action_str = json.dumps(action.model_dump(exclude_none=True))

            result      = await env.step(action)
            reward      = result.reward or 0.0
            done        = result.done
            obs_message = result.observation.message

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward,
                     done=done, error=err_msg)

            history.append({"role": "user",      "content": obs_message})
            history.append({"role": "assistant",  "content": raw})

            if done:
                break

        score   = min(max(sum(rewards), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        # Catch ALL exceptions so [END] is always emitted
        print(f"[DEBUG] Episode exception: {type(exc).__name__}: {exc}",
              flush=True)

    finally:
        # [END] must ALWAYS be emitted, even on crash
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score,
                rewards=rewards)

# ---------------------------------------------------------------------------
# Entry point — always exits with code 0
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(run_episode(TASK_NAME))
    except Exception as e:
        # Should never reach here, but ensure clean exit regardless
        print(f"[DEBUG] Top-level exception: {e}", flush=True)
    sys.exit(0)
