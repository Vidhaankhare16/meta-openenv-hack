"""
CRISPR Guide RNA Design Environment — Server Logic
===================================================
Implements the three tasks described in the hackathon problem statement.
All scoring is fully deterministic — no LLM-as-judge needed.

Biology primer (brief)
-----------------------
CRISPR-Cas9 cuts DNA at a site defined by:
  1. A ~20-nucleotide guide RNA (gRNA) complementary to the target.
  2. A PAM sequence (NGG on the non-template strand) immediately 3' of the
     cut site.
Finding valid cut sites = finding all NGG motifs in the sequence.
"""

import random
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

from ..models import CRISPRAction, CRISPRObservation, CRISPRState


# ============================================================================
# Synthetic biology data
# ============================================================================

# --- Easy task ---------------------------------------------------------------
# 90-bp synthetic sequence with several NGG PAM sites embedded at known
# positions.  The true PAM sites are computed at module load time by
# find_pam_sites(), so they are always consistent with the algorithm.
EASY_SEQUENCE = (
    "ATCGATCGGCAATCGATCGATCGGCATCGATCGGCATCGATCGATCGGCATCGATCGATCGG"
    "CATCGATCGATCGG"
)

# --- Medium task -------------------------------------------------------------
# 80-bp synthetic sequence representing a gene region with a pathogenic
# point mutation (C→T at position 38, creating a premature stop codon in
# this synthetic scenario).  The closest upstream PAM for a repair guide
# is at position 30 (NGG at 30-32).
MEDIUM_SEQUENCE = (
    "GCATCGATCGATCGATCGATCGATCGATCGATCGG"   # 35 bp  (PAM at 32: seq[32:35]="GGC" → N=G at 32)
    "CAT"                                    # mutation site (C at 38 → pathogenic)
    "CGATCGATCGATCGGCATCGATCGATCGG"          # 29 bp trailing
)
MUTATION_POSITION = 38   # pathogenic C→T change

# --- Hard task ---------------------------------------------------------------
# Three candidate guides with deliberately different off-target profiles.
# Guide 0 is DANGEROUS (many off-targets), Guide 1 is SAFEST (fewest),
# Guide 2 is MODERATE.
CANDIDATE_GUIDES = [
    "ATCGATCGATCGATCGATCG",   # Guide 0 – HIGH off-targets
    "GCTAGCTAGCTAGCTAGCTA",   # Guide 1 – SAFEST (true best)
    "TTCGATCGATCGATCGATCG",   # Guide 2 – MODERATE off-targets
]

# 500-bp genome excerpt with controlled off-target sites embedded.
# Constructed deterministically with random seed 42.
_GENOME_SEED = 42


def _build_genome_excerpt() -> str:
    """
    Build a reproducible 500-bp genome excerpt where:
      • Guide 0 has 6 off-target sites (≤3 mismatches, seed ≤1 mismatch).
      • Guide 1 has 1 off-target site (3 mismatches — borderline, seed OK).
      • Guide 2 has 3 off-target sites.
    """
    rng = random.Random(_GENOME_SEED)
    bases = "ACGT"
    genome = list(rng.choices(bases, k=500))

    def embed(guide: str, positions: List[int], n_mismatches: int) -> None:
        for pos in positions:
            variant = list(guide)
            # Introduce n_mismatches at non-seed positions (first 8 nt)
            mm_done = 0
            for idx in range(8):
                if mm_done >= n_mismatches:
                    break
                orig = variant[idx]
                replacement = rng.choice([b for b in bases if b != orig])
                variant[idx] = replacement
                mm_done += 1
            for k, nt in enumerate(variant):
                if pos + k < len(genome):
                    genome[pos + k] = nt

    # Guide 0 – 6 sites with 0-2 mismatches
    embed(CANDIDATE_GUIDES[0], [10, 55, 110, 170, 230, 300], 1)
    # One exact match for Guide 0 at position 10 (0 mismatches)
    for k, nt in enumerate(CANDIDATE_GUIDES[0]):
        genome[10 + k] = nt

    # Guide 1 – 1 site with 3 mismatches (seed still OK)
    embed(CANDIDATE_GUIDES[1], [360], 3)

    # Guide 2 – 3 sites with 1-2 mismatches
    embed(CANDIDATE_GUIDES[2], [390, 430, 465], 1)

    return "".join(genome[:500])


GENOME_EXCERPT = _build_genome_excerpt()


# ============================================================================
# Pure biology helper functions
# ============================================================================

def find_pam_sites(sequence: str) -> List[int]:
    """
    Return positions of all NGG PAM sites on the forward strand.

    Position i is a PAM site when sequence[i+1]=='G' and sequence[i+2]=='G'
    (i.e., the NGG triplet starts at i, where N = sequence[i]).

    Also scans the reverse-complement strand by looking for CCN (= NGG on
    the minus strand) on the forward representation.
    """
    seq = sequence.upper()
    sites: List[int] = []

    # Forward strand: NGG
    for i in range(len(seq) - 2):
        if seq[i + 1] == "G" and seq[i + 2] == "G":
            sites.append(i)

    # Reverse-complement strand: CCN on forward = NGG on minus strand
    for i in range(len(seq) - 2):
        if seq[i] == "C" and seq[i + 1] == "C":
            sites.append(-(i + 2))   # negative index convention for minus strand

    return sorted(set(sites))


def extract_guide(sequence: str, pam_position: int) -> Optional[str]:
    """
    Extract the 20-nt protospacer immediately upstream of the PAM at
    pam_position on the forward strand.  Returns None if the position
    is out of range.
    """
    seq = sequence.upper()
    start = pam_position - 20
    if start < 0 or pam_position + 2 >= len(seq):
        return None
    return seq[start:pam_position]


def compute_efficiency_score(guide: str) -> float:
    """
    Simplified on-target efficiency score inspired by Doench Rule Set 2
    (Doench et al., Nature Biotechnology 2016).

    Factors considered:
      • GC content  (ideal 40–70 %)
      • Poly-nucleotide stretches (≥4 same nt → penalty)
      • Seed-region T content (positions 15–20, adjacent to PAM)
      • Terminal nucleotide preferences

    Returns a float in [0.0, 1.0].
    """
    if len(guide) != 20:
        return 0.0
    guide = guide.upper()
    if not all(c in "ACGT" for c in guide):
        return 0.0

    score = 0.5  # base

    # GC content
    gc = (guide.count("G") + guide.count("C")) / 20.0
    if 0.40 <= gc <= 0.70:
        score += 0.20
    elif gc < 0.20 or gc > 0.90:
        score -= 0.30
    else:
        score += 0.10 * (1 - abs(gc - 0.55) / 0.35)

    # Poly-nucleotide penalty
    for nt in "ACGT":
        if nt * 4 in guide:
            score -= 0.15

    # Seed region (positions 15–19, 0-indexed): penalise T
    t_in_seed = guide[15:20].count("T")
    score -= t_in_seed * 0.02

    # Preferred terminal nucleotide: G at position 20 (index 19)
    if guide[19] == "G":
        score += 0.05

    # 5' end: prefer purine (A/G) at position 1
    if guide[0] in "AG":
        score += 0.03

    return round(max(0.0, min(1.0, score)), 4)


def find_offtargets(
    guide: str,
    genome: str,
    max_mismatches: int = 3,
    max_seed_mismatches: int = 1,
) -> List[Dict[str, Any]]:
    """
    Slide a window of len(guide) across the genome and record every window
    where:
      • Total mismatches ≤ max_mismatches
      • Seed-region mismatches (last 12 nt of guide, adjacent to PAM) ≤ 1

    Returns a list of hit dicts: {position, mismatches, seed_mismatches, sequence}.
    """
    guide = guide.upper()
    genome = genome.upper()
    n = len(guide)
    hits: List[Dict[str, Any]] = []

    for i in range(len(genome) - n + 1):
        window = genome[i : i + n]
        if not all(c in "ACGT" for c in window):
            continue
        total_mm = sum(1 for a, b in zip(guide, window) if a != b)
        seed_mm = sum(
            1 for a, b in zip(guide[-12:], window[-12:]) if a != b
        )
        if total_mm <= max_mismatches and seed_mm <= max_seed_mismatches:
            hits.append(
                {
                    "position": i,
                    "mismatches": total_mm,
                    "seed_mismatches": seed_mm,
                    "sequence": window,
                }
            )

    return hits


# ============================================================================
# Pre-compute ground-truth values
# ============================================================================

_TRUE_PAM_SITES_EASY: List[int] = [
    s for s in find_pam_sites(EASY_SEQUENCE) if s >= 0
]  # forward-strand only for easy task

_TRUE_PAM_SITES_MEDIUM: List[int] = [
    s for s in find_pam_sites(MEDIUM_SEQUENCE) if s >= 0
]

# Closest upstream PAM site to the mutation (must be ≤20 nt upstream so
# a guide can span it)
_BEST_PAM_MEDIUM: int = max(
    (p for p in _TRUE_PAM_SITES_MEDIUM if 0 < MUTATION_POSITION - p <= 22),
    default=_TRUE_PAM_SITES_MEDIUM[0] if _TRUE_PAM_SITES_MEDIUM else 0,
)

_TRUE_OFFTARGET_COUNTS: List[int] = [
    len(find_offtargets(g, GENOME_EXCERPT)) for g in CANDIDATE_GUIDES
]

_TRUE_BEST_GUIDE_INDEX: int = _TRUE_OFFTARGET_COUNTS.index(
    min(_TRUE_OFFTARGET_COUNTS)
)

# Safety ranking: sorted by off-target count ascending (safest first)
_TRUE_SAFETY_RANKING: List[int] = sorted(
    range(3), key=lambda i: _TRUE_OFFTARGET_COUNTS[i]
)


# ============================================================================
# Environment
# ============================================================================

class CRISPREnvironment(Environment):
    """
    CRISPR Guide RNA Design Environment.

    Three tasks are supported (set CRISPR_TASK env var or pass task= to reset):
      easy   – PAM site scanning
      medium – Guide RNA design + on-target efficiency scoring
      hard   – Off-target safety assessment + guide recommendation
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._state = CRISPRState()

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> CRISPRObservation:
        import os

        task_name = (
            task
            or os.getenv("CRISPR_TASK", "easy")
        ).lower()
        if task_name not in ("easy", "medium", "hard"):
            task_name = "easy"

        eid = episode_id or str(uuid.uuid4())

        if task_name == "easy":
            self._state = CRISPRState(
                episode_id=eid,
                step_count=0,
                task_name="easy",
                sequence=EASY_SEQUENCE,
                true_pam_sites=_TRUE_PAM_SITES_EASY,
            )
            return CRISPRObservation(
                done=False,
                reward=None,
                task="easy",
                sequence=EASY_SEQUENCE,
                message=(
                    "TASK (easy): Find all NGG PAM sites on the forward strand "
                    "of the sequence below.\n"
                    "Submit your answer with action_type='scan_sequence' and a "
                    "'pam_positions' list containing the 0-based index of each "
                    "N in the NGG triplet.\n\n"
                    f"Sequence ({len(EASY_SEQUENCE)} bp):\n{EASY_SEQUENCE}"
                ),
            )

        elif task_name == "medium":
            self._state = CRISPRState(
                episode_id=eid,
                step_count=0,
                task_name="medium",
                sequence=MEDIUM_SEQUENCE,
                true_pam_sites=_TRUE_PAM_SITES_MEDIUM,
                mutation_position=MUTATION_POSITION,
                best_pam_position=_BEST_PAM_MEDIUM,
            )
            return CRISPRObservation(
                done=False,
                reward=None,
                task="medium",
                sequence=MEDIUM_SEQUENCE,
                mutation_position=MUTATION_POSITION,
                mutation_info=(
                    f"Pathogenic C→T mutation at position {MUTATION_POSITION}. "
                    "Design a guide RNA to direct Cas9 to cut near this site."
                ),
                message=(
                    "TASK (medium): Design a 20-nt guide RNA to target the "
                    f"pathogenic mutation at position {MUTATION_POSITION}.\n\n"
                    "Steps:\n"
                    "  1. Call action_type='design_guide' with 'position' set to "
                    "     a valid NGG PAM position upstream of the mutation.\n"
                    "     The environment will return the 20-nt guide sequence.\n"
                    "  2. Call action_type='score_ontarget' with the returned "
                    "     'guide' sequence to get the efficiency score.\n\n"
                    f"Sequence ({len(MEDIUM_SEQUENCE)} bp):\n{MEDIUM_SEQUENCE}\n\n"
                    f"Mutation info: C→T at position {MUTATION_POSITION}."
                ),
            )

        else:  # hard
            self._state = CRISPRState(
                episode_id=eid,
                step_count=0,
                task_name="hard",
                sequence=MEDIUM_SEQUENCE,  # same gene region
                candidate_guides=CANDIDATE_GUIDES,
                genome_excerpt=GENOME_EXCERPT,
                true_offtarget_counts=_TRUE_OFFTARGET_COUNTS,
                true_best_guide_index=_TRUE_BEST_GUIDE_INDEX,
                checked_guides={},
            )
            return CRISPRObservation(
                done=False,
                reward=None,
                task="hard",
                sequence=MEDIUM_SEQUENCE,
                candidate_guides=CANDIDATE_GUIDES,
                genome_excerpt=GENOME_EXCERPT,
                checked_so_far={},
                message=(
                    "TASK (hard): Assess off-target binding risk for the 3 "
                    "candidate guides below against the provided genome excerpt, "
                    "then recommend the safest guide.\n\n"
                    "Steps:\n"
                    "  1. For each guide (index 0, 1, 2), call "
                    "     action_type='check_offtarget' with 'guide_index'.\n"
                    "  2. After checking all 3, call action_type='select_best' "
                    "     with 'ranking' (list of indices safest→most dangerous) "
                    "     and 'selected_guide_index' (the safest guide).\n\n"
                    "Candidate guides:\n"
                    + "\n".join(
                        f"  [{i}] {g}" for i, g in enumerate(CANDIDATE_GUIDES)
                    )
                    + f"\n\nGenome excerpt ({len(GENOME_EXCERPT)} bp) provided "
                    "in 'genome_excerpt' field."
                ),
            )

    # ------------------------------------------------------------------- step
    def step(
        self,
        action: CRISPRAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CRISPRObservation:
        self._state.step_count += 1
        task = self._state.task_name

        if task == "easy":
            return self._step_easy(action)
        elif task == "medium":
            return self._step_medium(action)
        else:
            return self._step_hard(action)

    # --------------------------------------------------------- easy task step
    def _step_easy(self, action: CRISPRAction) -> CRISPRObservation:
        if action.action_type != "scan_sequence":
            return CRISPRObservation(
                done=False,
                reward=0.0,
                task="easy",
                sequence=self._state.sequence,
                message=(
                    f"Invalid action '{action.action_type}' for easy task. "
                    "Use action_type='scan_sequence'."
                ),
            )

        predicted = set(action.pam_positions or [])
        true_set = set(self._state.true_pam_sites)

        correct = predicted & true_set
        recall = len(correct) / len(true_set) if true_set else 1.0
        # Small penalty for false positives to discourage guessing everything
        fp_penalty = max(0.0, (len(predicted - true_set) / max(len(true_set), 1)) * 0.1)
        reward = round(max(0.0, recall - fp_penalty), 4)

        self._state.cumulative_reward += reward

        return CRISPRObservation(
            done=True,
            reward=reward,
            task="easy",
            sequence=self._state.sequence,
            message=(
                f"Found {len(correct)}/{len(true_set)} PAM sites correctly. "
                f"False positives: {len(predicted - true_set)}. "
                f"Reward: {reward:.4f}\n"
                f"True PAM sites: {sorted(true_set)}\n"
                f"Your answer:    {sorted(predicted)}"
            ),
        )

    # ------------------------------------------------------- medium task step
    def _step_medium(self, action: CRISPRAction) -> CRISPRObservation:
        atype = action.action_type

        # ---- design_guide ----
        if atype == "design_guide":
            pos = action.position
            if pos is None:
                return CRISPRObservation(
                    done=False, reward=0.0, task="medium",
                    sequence=self._state.sequence,
                    mutation_position=self._state.mutation_position,
                    message="design_guide requires a 'position' integer.",
                )

            # Valid PAM nearby?
            true_pams = set(self._state.true_pam_sites)
            dist = abs(self._state.mutation_position - pos)
            valid_pam = pos in true_pams
            near_mutation = dist <= 25

            guide = extract_guide(self._state.sequence, pos)
            reward = 0.0
            if valid_pam and near_mutation and guide:
                reward = 0.4
                self._state.designed_guide = guide
                msg = (
                    f"Valid PAM at position {pos} (distance {dist} bp from mutation). "
                    f"Extracted guide: {guide}\n"
                    f"Now call score_ontarget with this guide."
                )
            else:
                reasons = []
                if not valid_pam:
                    reasons.append(f"position {pos} is not a valid NGG PAM site")
                if not near_mutation:
                    reasons.append(f"too far from mutation (dist={dist}, max=25)")
                if not guide:
                    reasons.append("could not extract 20-nt protospacer (too close to edge)")
                msg = "Invalid design_guide: " + "; ".join(reasons) + "."

            self._state.cumulative_reward += reward
            done = False  # Still need to score
            return CRISPRObservation(
                done=done, reward=reward, task="medium",
                sequence=self._state.sequence,
                mutation_position=self._state.mutation_position,
                returned_guide=guide if (valid_pam and near_mutation) else None,
                message=msg,
            )

        # ---- score_ontarget ----
        elif atype == "score_ontarget":
            guide_seq = (action.guide or "").upper().strip()
            if not guide_seq:
                return CRISPRObservation(
                    done=False, reward=0.0, task="medium",
                    sequence=self._state.sequence,
                    message="score_ontarget requires a 'guide' string.",
                )

            eff = compute_efficiency_score(guide_seq)
            reward = round(0.6 * eff, 4)
            self._state.ontarget_score = eff
            self._state.cumulative_reward += reward

            return CRISPRObservation(
                done=True, reward=reward, task="medium",
                sequence=self._state.sequence,
                mutation_position=self._state.mutation_position,
                returned_guide=guide_seq,
                efficiency_score=eff,
                message=(
                    f"On-target efficiency score: {eff:.4f}  "
                    f"(Doench Rule Set 2 simplified)\n"
                    f"Step reward: {reward:.4f} (= 0.6 × {eff:.4f})\n"
                    f"Cumulative reward: {self._state.cumulative_reward:.4f}"
                ),
            )

        else:
            return CRISPRObservation(
                done=False, reward=0.0, task="medium",
                sequence=self._state.sequence,
                message=(
                    f"Invalid action '{atype}' for medium task. "
                    "Use 'design_guide' then 'score_ontarget'."
                ),
            )

    # --------------------------------------------------------- hard task step
    def _step_hard(self, action: CRISPRAction) -> CRISPRObservation:
        atype = action.action_type

        # ---- check_offtarget ----
        if atype == "check_offtarget":
            idx = action.guide_index
            if idx is None or idx not in (0, 1, 2):
                return CRISPRObservation(
                    done=False, reward=0.0, task="hard",
                    sequence=self._state.sequence,
                    candidate_guides=self._state.candidate_guides,
                    genome_excerpt=self._state.genome_excerpt,
                    checked_so_far=dict(self._state.checked_guides),
                    message="check_offtarget requires 'guide_index' in {0, 1, 2}.",
                )

            guide_seq = self._state.candidate_guides[idx]
            hits = find_offtargets(guide_seq, self._state.genome_excerpt)
            count = len(hits)

            # Store result; reward 0.1 for each guide checked (max 3 × 0.1 = 0.3)
            key = str(idx)
            first_time = key not in self._state.checked_guides
            self._state.checked_guides[key] = count
            reward = 0.1 if first_time else 0.0
            self._state.cumulative_reward += reward

            return CRISPRObservation(
                done=False, reward=reward, task="hard",
                sequence=self._state.sequence,
                candidate_guides=self._state.candidate_guides,
                genome_excerpt=self._state.genome_excerpt,
                offtarget_result={
                    "guide_index": idx,
                    "guide_sequence": guide_seq,
                    "off_target_count": count,
                    "hits": hits[:5],          # first 5 hits shown
                    "total_hits": count,
                },
                checked_so_far=dict(self._state.checked_guides),
                message=(
                    f"Guide [{idx}] ({guide_seq}): {count} off-target site(s) found.\n"
                    f"Guides checked so far: {list(self._state.checked_guides.keys())}\n"
                    + ("Check the remaining guides, then call select_best."
                       if len(self._state.checked_guides) < 3
                       else "All 3 guides checked. Now call select_best.")
                ),
            )

        # ---- select_best ----
        elif atype == "select_best":
            checked = self._state.checked_guides
            all_checked = len(checked) == 3
            ranking = action.ranking or []
            sel_idx = action.selected_guide_index

            reward = 0.0

            # 0.3 for having called check_offtarget on all 3 guides
            if all_checked:
                reward += 0.3

            # 0.4 for correct safety ranking
            if ranking and list(ranking) == _TRUE_SAFETY_RANKING:
                reward += 0.4

            # 0.3 for selecting the correct safest guide
            if sel_idx == _TRUE_BEST_GUIDE_INDEX:
                reward += 0.3

            self._state.cumulative_reward += reward

            true_counts = self._state.true_offtarget_counts
            return CRISPRObservation(
                done=True, reward=reward, task="hard",
                sequence=self._state.sequence,
                candidate_guides=self._state.candidate_guides,
                checked_so_far=dict(checked),
                message=(
                    f"=== Hard Task Complete ===\n"
                    f"True off-target counts: {true_counts}\n"
                    f"True safety ranking (safest→dangerous): {_TRUE_SAFETY_RANKING}\n"
                    f"True safest guide: [{_TRUE_BEST_GUIDE_INDEX}] "
                    f"{CANDIDATE_GUIDES[_TRUE_BEST_GUIDE_INDEX]}\n\n"
                    f"Your ranking:     {ranking}\n"
                    f"Your selection:   [{sel_idx}]\n\n"
                    f"Step reward breakdown:\n"
                    f"  All 3 checked:   {'✓ +0.3' if all_checked else '✗ +0.0'}\n"
                    f"  Correct ranking: {'✓ +0.4' if list(ranking)==_TRUE_SAFETY_RANKING else '✗ +0.0'}\n"
                    f"  Correct pick:    {'✓ +0.3' if sel_idx==_TRUE_BEST_GUIDE_INDEX else '✗ +0.0'}\n"
                    f"  Step reward:     {reward:.2f}\n"
                    f"  Cumulative:      {self._state.cumulative_reward:.4f}"
                ),
            )

        else:
            return CRISPRObservation(
                done=False, reward=0.0, task="hard",
                sequence=self._state.sequence,
                candidate_guides=self._state.candidate_guides,
                genome_excerpt=self._state.genome_excerpt,
                checked_so_far=dict(self._state.checked_guides),
                message=(
                    f"Invalid action '{atype}' for hard task. "
                    "Use 'check_offtarget' (×3) then 'select_best'."
                ),
            )

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> CRISPRState:
        return self._state
