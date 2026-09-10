"""
graph_engine.py
LangGraph orchestration layer for the CV <-> Job Description <-> Question
Generation pipeline.

This wraps the functions already validated in ai_engine.py (extract_skills,
extract_jd_requirements, compute_skill_gap, generate_questions) as nodes in
an explicit state graph, instead of calling them as a plain sequential
script. CV extraction and JD extraction have no dependency on each other,
so they run as parallel branches that both feed into the gap-computation
node (fan-out / fan-in). Question generation then loops back on itself
(a real conditional edge, not just a sequential call) if the generated
questions don't clear the Answerability Score gate.

This is a structural/orchestration change only — the prompts, the model,
and the matching logic are untouched, so Phase 1/2 results are identical to
calling the functions directly.

This file is LangGraph WITHOUT ReAct: every path through the graph is fixed in
advance and the code decides which one is taken. The one conditional edge here
is a retry on a failed quality gate, not a choice made by a model.

The ReAct agent — where the MODEL decides — lives in react_agent.py. The two
were one file until they were separated, because sharing a module made them
look like one mechanism when they are two.
"""

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END

from ai_engine import (
    extract_skills,
    extract_jd_requirements,
    compute_skill_gap,
    generate_questions,
    detect_language,
)


MAX_QUESTION_GEN_ATTEMPTS = 2


class PipelineState(TypedDict, total=False):
    cv_text: str
    jd_text: str
    cv_skills: dict
    jd_skills: dict
    skill_gap: dict
    questions: dict
    question_gen_attempts: int
    # "ar" or "en". Detected once from the CV and carried through the graph, so
    # every downstream node speaks the candidate's language without re-guessing.
    language: str
    error: Optional[str]


def _extract_cv_node(state: PipelineState) -> dict:
    # Language is resolved on every path, not only the extraction path. The UI
    # passes cv_skills already extracted at upload time and no cv_text at all,
    # so detecting it only inside the extraction branch would silently leave an
    # Arabic interview running in English.
    updates = {}
    if not state.get("language"):
        updates["language"] = detect_language(state.get("cv_text") or "")

    if state.get("cv_skills"):
        # Already extracted upstream (e.g. at CV-upload time) — skip the
        # redundant LLM call rather than re-extracting on every JD comparison.
        return updates

    result = extract_skills(state["cv_text"], document_type="CV")
    if "error" in result:
        return {"error": f"CV extraction failed: {result['error']}"}
    updates["cv_skills"] = result
    return updates


def _extract_jd_node(state: PipelineState) -> dict:
    result = extract_jd_requirements(state["jd_text"])
    if "error" in result:
        return {"error": f"JD extraction failed: {result['error']}"}
    return {"jd_skills": result}


def _compute_gap_node(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    if not state.get("cv_skills") or not state.get("jd_skills"):
        return {"error": "Missing cv_skills or jd_skills before gap computation."}
    gap = compute_skill_gap(state["cv_skills"], state["jd_skills"])
    return {"skill_gap": gap}


def _generate_questions_node(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    attempts = state.get("question_gen_attempts", 0) + 1
    result = generate_questions(state["jd_skills"], state["cv_skills"],
                                state["skill_gap"],
                                language=state.get("language", "en"))
    if "error" in result:
        return {"error": f"Question generation failed: {result['error']}", "question_gen_attempts": attempts}
    return {"questions": result, "question_gen_attempts": attempts}


def _gate_router(state: PipelineState) -> str:
    """Conditional edge: if any generated question fails the Answerability
    Score gate and we haven't exhausted retries, loop back and regenerate.
    Otherwise (all pass, or out of attempts, or an upstream error), finish."""
    if state.get("error"):
        return "end"
    questions = state.get("questions", {}).get("questions", [])
    all_pass = all(q.get("passes_gate") for q in questions) if questions else False
    if all_pass or state.get("question_gen_attempts", 0) >= MAX_QUESTION_GEN_ATTEMPTS:
        return "end"
    return "retry"


def _build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("extract_cv", _extract_cv_node)
    graph.add_node("extract_jd", _extract_jd_node)
    graph.add_node("compute_gap", _compute_gap_node)
    graph.add_node("generate_questions", _generate_questions_node)

    # Fan-out: both extraction nodes start in parallel from the graph's entry point.
    graph.add_edge(START, "extract_cv")
    graph.add_edge(START, "extract_jd")

    # Fan-in: gap computation only runs once both extractions have completed.
    graph.add_edge("extract_cv", "compute_gap")
    graph.add_edge("extract_jd", "compute_gap")
    graph.add_edge("compute_gap", "generate_questions")

    # Conditional loop: retry question generation until the Answerability
    # Score gate passes, or MAX_QUESTION_GEN_ATTEMPTS is reached.
    graph.add_conditional_edges(
        "generate_questions",
        _gate_router,
        {"retry": "generate_questions", "end": END},
    )

    return graph.compile()


_COMPILED_GRAPH = _build_graph()


def run_cv_jd_pipeline(jd_text: str, cv_text: str = "",
                       cv_skills: Optional[dict] = None,
                       language: Optional[str] = None) -> dict:
    """
    Runs the full CV <-> Job Description <-> Question Generation graph:
    extract CV skills (or reuse already-extracted ones), extract JD
    requirements, compute the skill gap, then generate Answerability-scored
    interview questions prioritized by that gap.

    Args:
        jd_text: raw Job Description text.
        cv_text: raw CV text — required only if cv_skills isn't already provided.
        cv_skills: pre-extracted CV skills (e.g., from the CV-upload step) to
            avoid a redundant extract_skills() call on every gap analysis.
        language: "ar" or "en". Pass it when the CV was extracted upstream and
            its text is no longer available here; otherwise it is detected from
            cv_text. Everything the candidate reads is produced in it.

    Returns the final pipeline state as a dict with keys:
        cv_skills, jd_skills, skill_gap, questions, language,
        error (error only present on failure)
    """
    initial_state: PipelineState = {"cv_text": cv_text, "jd_text": jd_text}
    if cv_skills:
        initial_state["cv_skills"] = cv_skills
    if language:
        initial_state["language"] = language
    final_state = _COMPILED_GRAPH.invoke(initial_state)
    return dict(final_state)


if __name__ == "__main__":
    import json

    sample_cv = "Experienced in Python, Docker, and Git. Strong communication skills."
    sample_jd = "Requirements: Python, Kubernetes, AWS. Excellent communication skills."

    print("Running CV/JD/Question-Generation pipeline via LangGraph...\n")
    result = run_cv_jd_pipeline(jd_text=sample_jd, cv_text=sample_cv)
    print(json.dumps(result, indent=2, ensure_ascii=False))
