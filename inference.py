"""
CRISPR Guide RNA Design Environment — Inference Script
=======================================================
MANDATORY environment variables:
    API_BASE_URL      LLM API endpoint (OpenAI-compatible)
    MODEL_NAME        Model identifier
    HF_TOKEN          API key (or API_KEY)
    CRISPR_TASK       Task name: easy | medium | hard  (default: easy)
    LOCAL_IMAGE_NAME  Docker image name (if using from_docker_image)

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
import textwrap
from typing import List, Optional

from openai import OpenAI

from crispr_env import CRISPRAction, CRISPREnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.getenv("CRISPR_TASK",  "easy")
BENCHMARK    = "crispr_env"
MAX_STEPS    = 6
TEMPERATURE  = 0.2   # Low temp for deterministic tool calls
MAX_TOKENS   = 512
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Logging helpers
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
        You are a computational biologist assistant operating in a CRISPR
        guide-RNA design tool environment.

        Your current task is EASY: find all NGG PAM sites in a DNA sequence.

        A PAM site is an NGG triplet on the DNA forward strand.
        Position i is a PAM site when sequence[i+1]=='G' AND sequence[i+2]=='G'.
        (N = any nucleotide; i is the 0-based index of N.)

        You must respond with a single JSON object — no markdown, no prose:
        {"action_type": "scan_sequence", "pam_positions": [<list of ints>]}

        Scan the entire sequence carefully and list every position.
    """).strip(),

    "medium": textwrap.dedent("""
        You are a computational biologist assistant in a CRISPR design environment.

        Your task is MEDIUM: design a guide RNA near a pathogenic mutation.

        STEP SEQUENCE:
          Step 1 — Call design_guide to get the 20-nt guide at a valid PAM
                   position within ~25 bp of the mutation.
            {"action_type": "design_guide", "position": <int>}

          Step 2 — Score the returned guide with score_ontarget.
            {"action_type": "score_ontarget", "guide": "<20-nt string>"}

        Reply with ONE JSON object per turn. No markdown, no prose.
        Choose the PAM site closest to (but upstream of) the mutation position.
    """).strip(),

    "hard": textwrap.dedent("""
        You are a computational biologist assistant in a CRISPR design environment.

        Your task is HARD: assess off-target risk for 3 candidate guides and
        recommend the safest one.

        STEP SEQUENCE:
          Steps 1-3 — Check each guide (index 0, 1, 2):
            {"action_type": "check_offtarget", "guide_index": <0|1|2>}

          Step 4 — After checking all 3, select the safest guide:
            {
              "action_type": "select_best",
              "ranking": [<safest_index>, <middle_index>, <dangerous_index>],
              "selected_guide_index": <safest_index>
            }
          Ranking = sorted by off-target count ascending (fewest = safest).

        Reply with ONE JSON object per turn. No markdown, no prose.
    """).strip(),
}

# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Optional[CRISPRAction]:
    """Extract the first JSON object from the model's text response."""
    # Try to find a JSON block
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Try direct parse
    try:
        data = json.loads(text)
        return CRISPRAction(**data)
    except Exception:
        pass

    # Try to extract first {...} blob
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return CRISPRAction(**data)
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_model_action(
    client: OpenAI,
    system_prompt: str,
    history: List[dict],
    obs_message: str,
) -> str:
    """Call the LLM and return its raw text response."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])   # keep last 6 turns for context
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
# Fallback actions (heuristic, for when LLM output is unparseable)
# ---------------------------------------------------------------------------

def fallback_action(task: str, step: int, obs_message: str) -> CRISPRAction:
    """Return a reasonable fallback action if the LLM response is unparseable."""
    if task == "easy":
        # Return empty list — zero reward but valid action
        return CRISPRAction(action_type="scan_sequence", pam_positions=[])
    elif task == "medium":
        if step <= 1:
            return CRISPRAction(action_type="design_guide", position=30)
        else:
            return CRISPRAction(action_type="score_ontarget",
                                guide="ATCGATCGATCGATCGATCG")
    else:  # hard
        if step <= 3:
            return CRISPRAction(action_type="check_offtarget",
                                guide_index=step - 1)
        else:
            return CRISPRAction(action_type="select_best",
                                ranking=[0, 1, 2],
                                selected_guide_index=0)


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

async def run_episode(task: str) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    system_prompt = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["easy"])

    if IMAGE_NAME:
        env = await CRISPREnv.from_docker_image(IMAGE_NAME)
    else:
        # Connect to a running server (set API_BASE_URL to env endpoint)
        env_url = os.getenv("CRISPR_ENV_URL", "http://localhost:8000")
        env = CRISPREnv(base_url=env_url)

    rewards: List[float] = []
    history: List[dict] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task)
        obs_message = result.observation.message
        done = result.done

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Get action from LLM
            raw_response = get_model_action(
                client, system_prompt, history, obs_message
            )

            # Parse action
            action = parse_action(raw_response)
            error_msg = None
            if action is None:
                error_msg = f"unparseable_response: {raw_response[:80]!r}"
                action = fallback_action(task, step, obs_message)
                print(f"[DEBUG] Fallback action used at step {step}: "
                      f"{action.action_type}", flush=True)

            # Execute action
            action_str = json.dumps(action.model_dump(exclude_none=True))
            result = await env.step(action)

            reward = result.reward or 0.0
            done = result.done
            obs_message = result.observation.message

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward,
                     done=done, error=error_msg)

            # Update conversation history
            history.append({"role": "user",    "content": obs_message})
            history.append({"role": "assistant","content": raw_response})

            if done:
                break

        # Score = sum of rewards clamped to [0, 1]
        score = min(max(sum(rewards), 0.0), 1.0)
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
