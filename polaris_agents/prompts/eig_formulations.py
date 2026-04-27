"""
EIG (Expected Information Gain) formulation text generators.

Different formulations for how to present the EIG objective to the agent:
- JOINT: Maximize joint information gain about D and θ
- ADDITIVE: Explicit decomposition into D and θ components  
- IDS_RATIO: Information-Directed Sampling ratio formulation
"""

from .base import EIGFormulation


def get_eig_objective_text(formulation: EIGFormulation) -> str:
    """
    Get the EIG objective description text for the given formulation.
    
    Args:
        formulation: The EIG formulation to use
        
    Returns:
        Formatted string describing the EIG objective for prompts
    """
    if formulation == EIGFormulation.JOINT:
        return _get_joint_formulation_text()
    elif formulation == EIGFormulation.ADDITIVE:
        return _get_additive_formulation_text()
    elif formulation == EIGFormulation.IDS_RATIO:
        return _get_ids_ratio_formulation_text()
    else:
        raise ValueError(f"Unknown EIG formulation: {formulation}")


def _get_joint_formulation_text() -> str:
    """JOINT formulation: Maximize EIG(θ,D | action)"""
    return """Choose the action that maximizes Expected Information Gain (EIG) about D and θ.

**Expected Information Gain (EIG)** measures how much the observation from taking an action is expected to reduce uncertainty about D and θ:

EIG(action; history) = I((D,θ); observation | history, action)
                     = H[p(D,θ | history)] − E[H[p(D,θ | history, action, observation)]]

Where:
- H[p(D,θ | history)] is the current entropy (uncertainty) about D and θ given history so far
- E[H[p(D,θ | history, action, observation)]] is the expected entropy after taking action and receiving observation
- The difference is how much entropy about D and θ the action is expected to reduce

**Key principles:**
- Consider how the expected observation returned by the environment for each action will inform p(D) and p(θ)
- An action whose observation tells you little new provides low information gain
- Consider which uncertainty is currently most limiting your progress
- Prefer actions whose observations are expected to resolve the most impactful uncertainties for making an accurate prediction

**Action Selection Process:**
Consider plausible candidate actions, estimate their joint EIG, and select the action with the highest expected information gain."""


def _get_additive_formulation_text() -> str:
    """ADDITIVE formulation: Chain-rule decomposition of joint EIG"""
    return """Choose the action that maximizes Expected Information Gain (EIG) about D and θ.

**EIG can be decomposed using the chain rule:**

EIG(θ,D | action) = EIG(D | action) + EIG(θ | action, D)

Interpretation:
- EIG(D | action): Expected reduction in uncertainty about p(D) – how much you learn about which strategies, sources, or patterns are effective.
- EIG(θ | action, D): Given what you learn about D, the additional reduction in uncertainty about p(θ) for the specific instance.

When using this formulation, think in two steps:
1. How much will this action reduce uncertainty in p(D) (which strategies, sources, or patterns are effective)?
2. Given what you learn about D, how much additional reduction in p(θ) uncertainty will you gain (the instance-specific evidence)?

Some actions will mostly affect one term; others may contribute to both. You are optimizing the **total** uncertainty reduction about the joint (θ,D), expressed in this additive form.

**Key principles:**
- Consider how the expected observation for each action will inform p(D) and p(θ)
- An action whose observation tells you little new provides low information gain
- Prefer actions whose observations are expected to resolve the most impactful uncertainties

**Action Selection Process:**
Consider plausible candidate actions, estimate their joint EIG (sum of both terms), and select the action with the highest expected information gain."""


def _get_ids_ratio_formulation_text() -> str:
    return """Choose the next action using information-directed sampling, which scores each action by an information ratio that balances:
- Regret relative to having full knowledge of θ: the expected performance gap between taking this action now versus what you would achieve if you already knew the task-specific parameters.
- Learning how to act well in the future: captured by information gain about D (design effectiveness) – how much you learn about which actions, strategies, and information‑gathering patterns tend to work well for this kind of task.

The information-directed ratio expresses this tradeoff:
IDR(action) = ExpectedRegret(action))² / EIG(D | action)

Where:
- ExpectedRegret(action): Your expected performance gap between (a) taking this action and (b) taking the action you would choose if you already had perfect information about the task parameters θ for this task instance.
- EIG(D | action): Expected information gain about design effectiveness D – how much this action helps you learn which strategies, filters, or sources will make future actions better.

Conceptually:
- ExpectedRegret is driven by uncertainty about θ: How much worse your overall performance on this task could be, in expectation, compared to what you would achieve if you already knew the true θ and could choose your actions with that perfect information.
- EIG(D | action) captures learning about which actions will be good going forward: D encodes which strategies, filters, and information sources tend to lead to good decisions. Gaining information about D helps you choose better follow‑up actions as you continue working on this task (and on similar tasks).

The ratio favours actions that either:
- Keep the expected gap to the perfect-information performance small (low expected regret, by reducing current uncertainty about θ that matters for solving the current task), or
- Provide high information gain about D (clarifying which strategies, filters, or sources you should rely on when choosing later actions).

**Key principles:**
- Balance exploitation (reducing regret now) with exploration (learning for future actions)
- Actions with low regret AND high D-learning have excellent ratios
- Avoid actions that neither help the current task nor teach you anything useful

**Action Selection Process:**
Consider plausible candidate actions, estimate their information ratio, and select the action with the lowest ratio (best balance of regret and learning)."""

