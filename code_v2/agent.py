# # agent.py (The Definitive and Final Version with Clearer Structure)

# def create_prompt(diagnosis_summary: str, config: str, has_shap: bool, has_darshan: bool, options: dict, glossary: dict) -> str:
#     """
#     Creates the definitive, most advanced prompt using a unified reasoning framework
#     that forces a deep, two-stage analysis and produces a final, ranked list of
#     recommendations for all pipeline configurations.
#     """
#     options_str = ""
#     for key, value in options.items():
#         description = glossary.get(key, "No strategic description available.")
#         options_str += f"- **{key}**: {value}\n  - *Strategic Impact:* {description}\n"

#     # --- 1. Define the Core Context ---
#     prompt_header = f"""
# **ROLE AND GOAL:**
# You are an expert HPC I/O diagnostician. Your goal is to systematically evaluate all potential optimizations for a given performance issue, score their impact and risk, and synthesize these findings into a final, ranked list of the most beneficial recommendations.

# **CONTEXT:**
# You have performance data (the symptoms), the current configuration (the environment), and a strategic guide for all possible parameter changes (the Optimization Levers).

# **PERFORMANCE DIAGNOSIS DATA (The Symptoms):**
# {diagnosis_summary}

# **CURRENT I/O CONFIGURATION FILE (The Environment):**
# {config}

# **OPTIMIZATION LEVERS AND STRATEGIC IMPACT (Your Only Choices):**
# {options_str}
# """

#     # --- 2. Build the Definitive Two-Stage Reasoning Framework ---
#     # The framework is built piece by piece to ensure clarity and correctness.

#     # This part of Stage 1 is always present.
#     stage1_header = """
# **YOUR TASK: You MUST complete the following two stages in order.**

# ---
# ### **STAGE 1: Systematic Evaluation and Scoring**
# For **EACH** parameter in the "OPTIMIZATION LEVERS" list, perform the following analysis:

# 1.  **Assess Relevance and Justify:**
# """

#     # This is the DYNAMIC part. We insert a special instruction ONLY for the combined pipeline.
#     if has_shap and has_darshan:
#         stage1_relevance_instruction = """    - First, determine if changing this parameter is relevant by finding **corroborating evidence in BOTH** the SHAP data and the Darshan counters. If it's not supported by both, it is not relevant. State this corroborated evidence explicitly.
# """
#     else:
#         stage1_relevance_instruction = """    - First, determine if changing this parameter is relevant to addressing any of the bottlenecks listed in the "PERFORMANCE DIAGNOSIS DATA". Justify its relevance with specific data points.
# """

#     # This part of Stage 1 is common to ALL pipelines and is appended AFTER the dynamic instruction.
#     stage1_scoring_and_justification = """2.  **Assign Scores:** If the parameter is relevant, assign two scores:
#     - **Impact Score (1-10):** How much positive impact will the best change for this parameter have? (10 = massive improvement)
#     - **Risk Score (1-10):** How high is the risk of this change creating a new, significant bottleneck? (10 = very high risk)
# 3.  **Justify Scores:** Briefly explain your reasoning for both the Impact and Risk scores.
# """

#     # Stage 2 is also common to ALL pipelines.
#     stage2_synthesis_and_ranking = """
# ---
# ### **STAGE 2: Synthesized and Ranked Recommendations**
# Now, synthesize your findings from Stage 1 into a final, consolidated action plan.

# 1.  **Create Final Ranked List:** Present a numbered list of the optimization changes you analyzed in Stage 1 that have an Impact Score greater than 5. You **MUST** order this list from most to least beneficial (highest Impact, lowest Risk).
# 2.  **Format Each Item:** Each item in the final ranked list must clearly state:
#     - The recommended parameter change (e.g., "Change `fsync` from `1` to `0`").
#     - The **Impact Score** and **Risk Score** you assigned.
#     - A final justification for its rank in the list.
# """
    
#     # --- Assemble all the pieces into the final prompt ---
#     full_prompt = (
#         prompt_header 
#         + stage1_header 
#         + stage1_relevance_instruction 
#         + stage1_scoring_and_justification 
#         + stage2_synthesis_and_ranking
#     )
    
#     return full_prompt

def create_prompt(diagnosis_summary: str, config: str, has_shap: bool, has_darshan: bool, options: dict, glossary: dict) -> str:
    """
    Creates the definitive, most advanced prompt using a unified reasoning framework
    that forces a deep, three-stage analysis and produces a final, consolidated
    recommendation.
    """
    options_str = ""
    for key, value in options.items():
        description = glossary.get(key, "No strategic description available.")
        options_str += f"- **{key}**: {value}\n  - *Strategic Impact:* {description}\n"

    # --- 1. Define the Core Context for the LLM ---
    prompt_header = f"""
**ROLE AND GOAL:**
You are an expert HPC I/O diagnostician. Your goal is to systematically evaluate all potential optimizations for a given performance issue, score them, rank them, and finally synthesize your findings into a single, actionable configuration recommendation.

**CONTEXT:**
You have performance data listing top bottlenecks (the symptoms), the current configuration (the environment), and a strategic guide for all possible parameter changes (the Optimization Levers).

**PERFORMANCE DIAGNOSIS DATA (The Symptoms):**
{diagnosis_summary}

**CURRENT I/O CONFIGURATION FILE (The Environment):**
{config}

**OPTIMIZATION LEVERS AND STRATEGIC IMPACT (Your Only Choices):**
{options_str}
"""

    # --- 2. Build the Definitive Three-Stage Reasoning Framework ---
    # This entire block of text is sent to the LLM as its instructions.

    reasoning_framework = """
**YOUR TASK: You MUST complete the following three stages in order.**

---
### **STAGE 1: Systematic Evaluation and Scoring**
For **EACH** parameter in the "OPTIMIZATION LEVERS" list that is relevant to the bottlenecks provided, perform the following complete analysis:

1.  **Assess Relevance and Justify:**
"""
    # This is the dynamic instruction that makes the agent smarter for the combined pipeline.
    if has_shap and has_darshan:
        reasoning_framework += """    - First, determine if changing this parameter is relevant by finding **corroborating evidence in BOTH** the SHAP data and the Darshan counters. If it's not supported by both, it is not relevant. State this corroborated evidence explicitly.
"""
    else:
        reasoning_framework += """    - First, determine if changing this parameter is relevant to addressing any of the bottlenecks listed in the "PERFORMANCE DIAGNOSIS DATA". Justify its relevance with specific data points.
"""

    # This part of Stage 1 is common to all pipelines.
    reasoning_framework += """2.  **Assign Scores:** If the parameter is relevant, assign two scores:
    - **Impact Score (1-10):** How much positive impact will the best change for this parameter have? (10 = massive improvement)
    - **Risk Score (1-10):** How high is the risk of this change creating a new, significant bottleneck? (10 = very high risk)
3.  **Justify Scores:** Briefly explain your reasoning for both the Impact and Risk scores.

---
### **STAGE 2: Ranked Recommendations**
Now, synthesize your findings from Stage 1 into a final, ranked list.

1.  **Create Final Ranked List:** Present a numbered list of the optimization changes you analyzed in Stage 1 that have an Impact Score greater than 5. You **MUST** order this list from most to least beneficial (highest Impact, lowest Risk).
2.  **Format Each Item:** Each item in the final ranked list must clearly state the recommended parameter change, its **Impact Score**, its **Risk Score**, and a final justification.

---
### **STAGE 3: Final Consolidated Configuration**
Finally, create a single, actionable recommendation based on your ranked list from Stage 2.

1.  **Select Top Changes:** From your ranked list, select the top 4 or 5 non-contradictory changes that provide the best overall improvement.
2.  **Create "Before and After" Block:** Present the final configuration in the following format. In the "After" block, include a concise note about the expected benefit, referencing the scores. **Only include parameters that are actually being changed.**

**Final Recommendation:**
**Before:**
parameter1 = value
parameter2 = value


**After:**
parameter1 = new_value  (Impact: X, Risk: Y - Expected to solve the primary bottleneck)
parameter2 = new_value  (Impact: A, Risk: B - A complementary change to improve throughput)

"""
    
    # Assemble all the pieces into the final prompt
    full_prompt = prompt_header + reasoning_framework
    
    return full_prompt