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
"""

from typing import Annotated, TypedDict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from ai_engine import (
    extract_skills,
    extract_jd_requirements,
    compute_skill_gap,
    generate_questions,
    evaluate_answer,
    generate_followup,
    generate_off_plan_question,
    _build_skill_vocabulary,
    _is_soft_skill_target,
    detect_language,
    agent_client,
    AGENT_SYSTEM_PROMPT,
    MAX_PROBE_DEPTH,
    MAX_OFF_PLAN_QUESTIONS,
    MAX_TOTAL_QUESTIONS,
    ANSWER_WEAK_THRESHOLD,
    ANSWER_STRONG_THRESHOLD,
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


# ==========================================
# Adaptive interview agent (ReAct)
# ==========================================
# Unlike the pipeline above — where the path is fixed and the code decides —
# here the MODEL decides. It reads the answer evaluation, may query the
# interview state, and then calls one of four decision tools. That satisfies
# the ReAct pattern: an explicit Thought, a model-chosen Action, a real
# Observation returned from outside the model, and a loop back so the next
# Thought is grounded in what was observed.


class AnswerCycleState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    evaluation: dict
    error: Optional[str]


def _build_agent_tools(ctx: dict) -> list:
    """Tools are built per-invocation as closures over `ctx`, so they can read
    the live interview state and record the decision without global state."""

    @tool
    def get_interview_state() -> dict:
        """Read the current interview state: which skills are still queued,
        which have been covered, how many questions have been asked, and how
        much of the probe/off-plan budget remains."""
        return {
            "remaining_skills": ctx["remaining_skills"],
            "covered_skills": ctx["covered_skills"],
            "questions_asked": ctx["asked_count"],
            "questions_budget_left": ctx["budget_left"],
            "probe_depth_on_current_skill": ctx["probe_depth"],
            "max_probe_depth": MAX_PROBE_DEPTH,
            "off_plan_questions_used": ctx["off_plan_used"],
            "max_off_plan_questions": MAX_OFF_PLAN_QUESTIONS,
        }

    @tool
    def move_to_another_skill(reason: str) -> dict:
        """Skip the remaining planned questions on the CURRENT skill and move
        to the next skill. Use when the candidate has shown they lack the
        fundamentals of this skill, so pressing further yields no new signal.
        `reason` must cite the evaluation numbers behind the decision."""
        ctx["decision"] = {"route": "skip_skill", "reason": reason}
        return {"status": "will skip remaining questions on "
                          f"'{ctx['current_skill']}'"}

    @tool
    def probe_deeper(aspect: str) -> dict:
        """Ask a deeper follow-up on the SAME skill. Use when the answer was
        accurate but shallow, to find the limit of what the candidate knows.
        `aspect` is the specific thing to explore that they did not cover."""
        ctx["decision"] = {"route": "probe", "reason": aspect}
        return {"status": f"will generate a follow-up on '{ctx['current_skill']}'",
                "aspect": aspect}

    @tool
    def ask_about_skill(skill_name: str, reason: str) -> dict:
        """Ask about a skill the candidate raised themselves that was not in
        the planned set. Use when they mention a role-relevant technology
        while answering something else — a thread worth following."""
        ctx["decision"] = {"route": "ask_about_skill",
                           "skill": skill_name, "reason": reason}
        return {"status": f"will ask about '{skill_name}' (off-plan)"}

    @tool
    def continue_as_planned() -> dict:
        """Move to the next question in the plan without changing anything.
        Use when the skill is adequately covered and nothing warrants a
        detour."""
        ctx["decision"] = {"route": "next", "reason": "skill adequately covered"}
        return {"status": "will continue with the planned questions"}

    return [get_interview_state, move_to_another_skill, probe_deeper,
            ask_about_skill, continue_as_planned]


def _fallback_route(evaluation: dict, probe_depth: int) -> dict:
    """Deterministic safety net. Alaa (2025) documented small models producing
    fluent replies WITHOUT invoking any tool; if that happens here the
    interview must still advance sensibly rather than stall."""
    score = evaluation.get("final_score", 0.0)
    if score < ANSWER_WEAK_THRESHOLD:
        return {"route": "skip_skill",
                "reason": f"[fallback] weak answer (score {score:.2f})"}
    if score > ANSWER_STRONG_THRESHOLD and probe_depth < MAX_PROBE_DEPTH:
        return {"route": "probe",
                "reason": f"[fallback] strong answer (score {score:.2f})"}
    return {"route": "next", "reason": f"[fallback] score {score:.2f}"}


def _build_answer_cycle_graph(tools: list):
    agent_model = agent_client.bind_tools(tools)

    def _agent_node(state: AnswerCycleState) -> dict:
        return {"messages": [agent_model.invoke(state["messages"])]}

    def _should_continue(state: AnswerCycleState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    graph = StateGraph(AnswerCycleState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue,
                                {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")     # ← the ReAct loop
    return graph.compile()


def run_answer_cycle(question: dict, answer: str, cv_skills: dict,
                     jd_skills: dict, interview_state: dict,
                     language: str = "en") -> dict:
    """
    One adaptive interview step: evaluate the candidate's answer, then let the
    agent decide what to ask next.

    Args:
        question: the question just asked ({"question", "targets_skill", ...}).
        answer: what the candidate wrote.
        cv_skills / jd_skills: extraction results, used for the vocabulary.
        interview_state: {"remaining_skills", "covered_skills", "asked_count",
            "budget_left", "probe_depth", "off_plan_used"}.

    Returns:
        {"evaluation", "route", "reason", "thought", "new_question",
         "used_fallback", "tool_calls"} or {"error": ...}
    """
    targets_skill = question.get("targets_skill", "")
    vocabulary = _build_skill_vocabulary(jd_skills, cv_skills)
    is_soft = _is_soft_skill_target(targets_skill, jd_skills, cv_skills)

    evaluation = evaluate_answer(question.get("question", ""), answer,
                                 targets_skill, vocabulary,
                                 is_soft_skill=is_soft, language=language)
    if "error" in evaluation:
        return {"error": f"Answer evaluation failed: {evaluation['error']}"}

    ctx = dict(interview_state)
    ctx["current_skill"] = targets_skill
    ctx["decision"] = None

    tools = _build_agent_tools(ctx)
    graph = _build_answer_cycle_graph(tools)

    # A behavioural answer was graded on a different rubric, so the agent must
    # be shown that rubric — briefing it with "technical_accuracy = None" would
    # invite it to read a missing number as a failed one.
    if evaluation.get("is_soft_skill"):
        star = evaluation.get("star", {})
        criteria = (
            f"  [behavioural question — graded on STAR, not technical accuracy]\n"
            f"  situation         = {star.get('situation')}\n"
            f"  task              = {star.get('task')}\n"
            f"  action            = {star.get('action')}\n"
            f"  result            = {star.get('result')}\n"
            f"  star_completeness = {evaluation.get('star_completeness')}\n"
            f"  relevance         = {evaluation['relevance']}\n"
            f"  depth             = {evaluation['depth']}\n"
        )
    else:
        criteria = (
            f"  technical_accuracy= {evaluation['technical_accuracy']}\n"
            f"  relevance         = {evaluation['relevance']}\n"
            f"  depth             = {evaluation['depth']}\n"
        )

    # A zero means one of two different things, and the agent cannot act well
    # on the number alone. Before the evaluator zeroed off-topic answers, an
    # expert reply to the wrong question scored 0.45 and the agent read the
    # partial credit correctly — it re-asked the question, reasoning that there
    # was "no signal on this skill yet". Zeroing the score made the report
    # honest but took that signal away: the same answer now reads as 0.0, which
    # looks like "does not know", and the agent abandons a skill that was never
    # actually tested. So the reason is stated in words.
    untested_note = ("\n  NOTE: the answer did not address the skill at all "
                     "(relevance 0.0). This is NOT evidence that the candidate "
                     "lacks the skill — it was never tested. Re-asking the "
                     "question is usually better than moving on.\n"
                     if evaluation.get("skill_untested") else "")

    briefing = (
        f"Skill just probed: {targets_skill}\n"
        f"Question asked: {question.get('question', '')}\n"
        f"Candidate's answer: {answer}\n\n"
        f"Evaluation:\n"
        f"  final_score       = {evaluation['final_score']}\n"
        f"{criteria}"
        f"  filler_words      = {evaluation['filler_count']}\n"
        f"{untested_note}\n"
        f"Interview so far: {ctx['asked_count']} asked, "
        f"{ctx['budget_left']} left, probe depth on this skill "
        f"{ctx['probe_depth']}/{MAX_PROBE_DEPTH}, "
        f"off-plan used {ctx['off_plan_used']}/{MAX_OFF_PLAN_QUESTIONS}.\n"
        f"Skills still queued: {ctx['remaining_skills']}\n\n"
        f"Decide the next step."
    )

    try:
        final_state = graph.invoke({"messages": [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=briefing),
        ]})
    except Exception as e:
        return {"error": f"Agent step failed: {e}", "evaluation": evaluation}

    # The agent's written reasoning — this is what makes an adaptive decision
    # auditable instead of silent, and it is surfaced in the final report.
    thought = " ".join(
        m.content for m in final_state["messages"]
        if getattr(m, "type", "") == "ai" and isinstance(m.content, str) and m.content.strip()
    ).strip()

    tool_calls = [tc["name"] for m in final_state["messages"]
                  for tc in (getattr(m, "tool_calls", None) or [])]

    decision = ctx["decision"]
    used_fallback = decision is None
    if used_fallback:
        decision = _fallback_route(evaluation, ctx["probe_depth"])

    # --- budget enforcement ------------------------------------------------
    # The limits were stated to the agent in its briefing and nowhere else, so
    # they were a request rather than a rule: a real interview ran to 13
    # questions against a stated ceiling of 12, because nothing in the code
    # ever checked. An adaptive interview that can extend itself needs a stop
    # that does not depend on the model choosing to obey.
    #
    # Enforced after the decision rather than before it, so the agent's own
    # reasoning is still recorded — the report then shows both what it wanted
    # to do and why it was not allowed to.
    budget_note = None
    if decision["route"] == "probe" and ctx["probe_depth"] >= MAX_PROBE_DEPTH:
        budget_note = (f"probe budget exhausted "
                       f"({ctx['probe_depth']}/{MAX_PROBE_DEPTH} on this skill)")
    elif (decision["route"] == "ask_about_skill"
          and ctx["off_plan_used"] >= MAX_OFF_PLAN_QUESTIONS):
        budget_note = (f"off-plan budget exhausted "
                       f"({ctx['off_plan_used']}/{MAX_OFF_PLAN_QUESTIONS})")
    elif (decision["route"] in ("probe", "ask_about_skill")
          and ctx["asked_count"] >= MAX_TOTAL_QUESTIONS):
        budget_note = (f"interview length ceiling reached "
                       f"({ctx['asked_count']}/{MAX_TOTAL_QUESTIONS} asked)")

    if budget_note:
        decision = {"route": "next",
                    "reason": f"[budget] {decision['reason']} — overridden: {budget_note}"}

    # Dynamically generated questions clear the SAME Answerability gate as the
    # planned ones; if none does, the caller falls back to the planned queue.
    new_question = None
    if decision["route"] == "probe":
        # For a behavioural probe the STAR component scores steer the follow-up
        # towards whatever the candidate left out, so they are passed through.
        new_question = generate_followup(
            question.get("question", ""), answer, targets_skill,
            vocabulary, is_soft, star=evaluation.get("star"), language=language)
        if new_question is None:
            decision = {"route": "next",
                        "reason": "follow-up failed the Answerability gate"}
    elif decision["route"] == "ask_about_skill":
        skill_name = decision.get("skill", "")
        new_question = generate_off_plan_question(
            skill_name, answer, vocabulary,
            _is_soft_skill_target(skill_name, jd_skills, cv_skills),
            language=language)
        if new_question is None:
            decision = {"route": "next",
                        "reason": "off-plan question failed the Answerability gate"}

    return {
        "evaluation": evaluation,
        "route": decision["route"],
        "reason": decision.get("reason", ""),
        "thought": thought,
        "new_question": new_question,
        "used_fallback": used_fallback,
        "budget_capped": budget_note is not None,
        "tool_calls": tool_calls,
    }


if __name__ == "__main__":
    import json

    sample_cv = "Experienced in Python, Docker, and Git. Strong communication skills."
    sample_jd = "Requirements: Python, Kubernetes, AWS. Excellent communication skills."

    print("Running CV/JD/Question-Generation pipeline via LangGraph...\n")
    result = run_cv_jd_pipeline(jd_text=sample_jd, cv_text=sample_cv)
    print(json.dumps(result, indent=2, ensure_ascii=False))
