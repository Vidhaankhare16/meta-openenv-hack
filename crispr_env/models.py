"""
CRISPR Guide RNA Design Environment — Data Models
==================================================
Pydantic models defining the action, observation, and state for the
three-task CRISPR environment.

Tasks
-----
easy   : Find all NGG PAM sites in a DNA sequence.
medium : Design a 20-nt guide RNA and score its on-target efficiency.
hard   : Assess off-target binding risk for 3 candidate guides and
         recommend the safest one.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class CRISPRAction(Action):
    """
    A single step taken by the agent in the CRISPR environment.

    action_type must be one of:
        scan_sequence   – (easy)   Submit predicted PAM-site positions.
        design_guide    – (medium) Request the 20-nt guide at a PAM position.
        score_ontarget  – (medium) Score a guide RNA for on-target efficiency.
        check_offtarget – (hard)   Check off-target sites for one candidate guide.
        select_best     – (hard)   Submit final guide ranking + recommendation.
    """

    action_type: str

    # --- scan_sequence (easy) -------------------------------------------
    # Agent submits the list of PAM positions it found.
    pam_positions: Optional[List[int]] = None

    # --- design_guide (medium) ------------------------------------------
    # Agent selects which PAM position to target.
    position: Optional[int] = None

    # --- score_ontarget (medium) ----------------------------------------
    # Agent provides the guide sequence to evaluate.
    guide: Optional[str] = None

    # --- check_offtarget (hard) -----------------------------------------
    # Agent specifies which candidate guide (0, 1, or 2) to check.
    guide_index: Optional[int] = None

    # --- select_best (hard) ---------------------------------------------
    # Agent's safety ranking (safest → most dangerous) and final pick.
    ranking: Optional[List[int]] = None          # e.g. [1, 2, 0]
    selected_guide_index: Optional[int] = None   # index of chosen guide


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class CRISPRObservation(Observation):
    """
    What the agent sees after each step.

    Common fields (all tasks)
    -------------------------
    task            : "easy" | "medium" | "hard"
    sequence        : The DNA sequence being worked on.
    message         : Human-readable feedback / instructions.

    Easy-task fields
    ----------------
    (No extra context needed — full sequence is the task.)

    Medium-task fields
    ------------------
    mutation_position : Index of the pathogenic mutation.
    mutation_info     : Text description of the mutation.
    returned_guide    : 20-nt guide sequence returned by design_guide().
    efficiency_score  : Float [0,1] returned by score_ontarget().

    Hard-task fields
    ----------------
    candidate_guides  : List of 3 guide sequences to evaluate.
    genome_excerpt    : 500-bp reference genome excerpt.
    offtarget_result  : Off-target data returned by check_offtarget().
    checked_so_far    : Dict mapping guide index → off-target count.
    """

    task: str
    sequence: str
    message: str

    # Medium
    mutation_position: Optional[int] = None
    mutation_info: Optional[str] = None
    returned_guide: Optional[str] = None
    efficiency_score: Optional[float] = None

    # Hard
    candidate_guides: Optional[List[str]] = None
    genome_excerpt: Optional[str] = None
    offtarget_result: Optional[Dict[str, Any]] = None
    checked_so_far: Optional[Dict[str, int]] = None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CRISPRState(State):
    """
    Internal episode metadata (not exposed to the agent directly).
    """

    task_name: str = "easy"
    sequence: str = ""

    # Easy
    true_pam_sites: List[int] = []

    # Medium
    mutation_position: int = 0
    best_pam_position: int = 0
    designed_guide: Optional[str] = None
    ontarget_score: Optional[float] = None

    # Hard
    candidate_guides: List[str] = []
    genome_excerpt: str = ""
    true_offtarget_counts: List[int] = []   # [count_g0, count_g1, count_g2]
    true_best_guide_index: int = 0           # index of safest guide
    checked_guides: Dict[str, int] = {}      # guide_index_str → count discovered

    # Reward tracking
    cumulative_reward: float = 0.0
