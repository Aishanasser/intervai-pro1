
import os
import re
import json
import unicodedata
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ==========================================
# 1. Configuration
# ==========================================
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
# Pinned to the dated snapshot, not the rolling alias.
#
# The alias `deepseek-v4-pro` started returning 404 ("Model not found,
# inaccessible, and/or not deployed") mid-project while still being listed by
# the /models endpoint — the whole platform stopped working with no code
# change on our side. The dated build serves the same model and was the
# fastest of the alternatives measured (3.1s vs 25-30s for kimi-k2p6 and
# deepseek-v4-flash).
#
# Pinning also matters for the evaluation itself: an F1 figure is only
# reproducible if the model behind it cannot be swapped underneath the name.
MODEL_NAME = "accounts/fireworks/models/deepseek-v4-pro-0813"

if not FIREWORKS_API_KEY:
    raise RuntimeError(
        "FIREWORKS_API_KEY environment variable is not set. "
        "Add it to your .env file or export it before running."
    )

client = ChatOpenAI(
    model=MODEL_NAME,
    api_key=FIREWORKS_API_KEY,
    base_url=FIREWORKS_BASE_URL,
    temperature=0.0,  # deterministic: extraction task, not creative generation
    max_tokens=16384,
    model_kwargs={"response_format": {"type": "json_object"}},
)

# A second client for the interview agent. It is identical except that it does
# NOT force a JSON-object response, because forced JSON mode and tool calling
# are mutually exclusive: JSON mode requires the reply to be a JSON document,
# while tool calling requires it to be a tool_calls payload. Binding tools to
# the JSON-mode client fails with "not strict. Only strict function tools can
# be auto-parsed".
agent_client = ChatOpenAI(
    model=MODEL_NAME,
    api_key=FIREWORKS_API_KEY,
    base_url=FIREWORKS_BASE_URL,
    temperature=0.0,
    max_tokens=16384,
)

# ==========================================
# 2. System Prompt — Phase 1 (CV Skill Extraction)
# ==========================================
CV_SKILL_EXTRACTION_PROMPT = """You are an expert AI system for resume skill extraction.

Your task is to analyze the provided CV and identify all skills mentioned.
The text may be in English, Arabic, or a mix of both.

Before answering, scan the CV section by section (Personal Skills, Experience,
Training/Certificates, Honors/Achievements, Activities/Memberships, etc.) —
a skill can appear in any section, not just one literally named "Skills". Do
not skip any section. However, scanning every section does NOT mean every
sentence contains a skill: a job duty, responsibility, task, or achievement
described in prose (e.g., "documented procedures", "asset & account
management", "provided IT support for 1,000+ staff", "secured sponsorships")
is NOT itself a skill — only extract something from a narrative sentence if
it names a concrete, specific tool, technology, programming language,
framework, methodology, or technique (e.g., "Docker", "Deep Learning",
"Python"). Do not turn a whole task/responsibility description into a skill
entry just because it appeared while scanning.

Rules:

1. Extract only skills — concrete named tools, technologies, programming
   languages, frameworks, methodologies, techniques, or competencies. Not
   general job duties, responsibilities, or achievements described in prose.
2. Keep multi-word skills together. If a longer skill phrase contains a shorter
   skill inside it (e.g., "Microsoft SQL Server" contains "SQL Server" and "SQL"),
   extract only the longest/complete form — do not also list the shorter
   sub-phrase separately.
3. Separate Technical Skills and Soft Skills.
4. Languages (e.g., Arabic, English, Turkish) always go in their own
   "languages" list — never in technical_skills or soft_skills. Extract each
   language exactly as it appears including any proficiency qualifier, per
   rule 7 (e.g., if the CV says "Arabic (Native)" or "English — Advanced (C1)",
   extract that full string, not just "Arabic" or "English" alone).
5. Ignore names, companies, universities, projects, and job titles.
6. Remove duplicates.
7. Preserve the original wording exactly as it appears in the text. Never
   paraphrase, summarize, or convert a descriptive sentence into a generic
   label — extract the exact phrase as written, word for word, including any
   qualifiers (e.g., "Proven ability to work under pressure" must stay exactly
   that, not "Ability to work under pressure"; "Social person with lots of
   connections" must stay exactly that, not "Networking"). The only exception
   is a trailing sentence-ending punctuation mark (e.g. a final "." or ".."),
   which must be dropped — everything else stays untouched.
8. A certificate, training course, or award mentioning a specific skill (e.g.,
   "First aid certificate from the World First Aid Organization", "Email
   etiquette course with [name]") is a valid skill source — extract the named
   skill from it (e.g., "First aid", "Email etiquette"), ignoring the issuing
   organization and any person's name per rule 5.
9. Return only JSON — no preamble, no explanation, no markdown code fences.

Output format:

{
  "technical_skills": [],
  "soft_skills": [],
  "languages": []
}
"""

# ==========================================
# 2b. System Prompt — Phase 2 (Job Description Requirement Extraction)
# ==========================================
JD_REQUIREMENT_EXTRACTION_PROMPT = """You are an expert AI system for extracting required
skills from a company's Job Description (JD).

Your task is to analyze the provided Job Description and identify all skills
it asks the candidate to have — whether listed under "Requirements",
"Qualifications", "Nice to have", "Preferred", "Responsibilities", or written
in plain prose anywhere in the text. The text may be in English, Arabic, or a
mix of both.

Rules:

1. Extract only skills — concrete named tools, technologies, programming
   languages, frameworks, methodologies, techniques, or competencies. Not
   generic phrases like "team player mindset" unless they name an actual
   competency (e.g., "communication skills" is fine; "fast-paced
   environment" is not a skill).
2. Keep multi-word skills together. If a longer skill phrase contains a shorter
   skill inside it (e.g., "Microsoft SQL Server" contains "SQL Server" and "SQL"),
   extract only the longest/complete form — do not also list the shorter
   sub-phrase separately.
3. Separate Technical Skills and Soft Skills.
4. Languages (e.g., Arabic, English, Turkish) always go in their own
   "languages" list — never in technical_skills or soft_skills. Extract each
   language exactly as it appears including any proficiency qualifier (e.g.,
   "Fluent in English" -> "Fluent in English", not just "English").
5. Ignore the company name, job title, location, salary, and benefits.
6. Remove duplicates.
7. Preserve the original wording exactly as it appears in the text — do not
   paraphrase or generalize a requirement into a shorter label. The only
   exception is a trailing sentence-ending punctuation mark, which must be
   dropped.
8. Treat "required" and "nice to have"/"preferred" skills the same way —
   extract both into the same lists (no separate priority tier).
9. Return only JSON — no preamble, no explanation, no markdown code fences.

Output format:

{
  "technical_skills": [],
  "soft_skills": [],
  "languages": []
}
"""

# ==========================================
# 3. LLM call wrapper (swap this when moving providers)
# ==========================================
# DeepSeek v4 Pro is a reasoning model: it spends completion tokens thinking
# before it writes a single character of the answer, and that thinking is
# billed against the same max_tokens budget as the answer itself. Extraction
# normally costs about 1,300 reasoning tokens, but a job description padded
# with benefits, equal-opportunity boilerplate and application instructions has
# been observed to make it deliberate until all 16,384 tokens were gone and
# emit no content at all. The call then fails inside the OpenAI client with a
# raw CompletionUsage dump, which is what the user sees.
#
# reasoning_effort="low" does not help — measured on this model, reasoning went
# up (1,375 tokens) rather than down (1,310). What does help is room: the
# model's context window is 1,048,576 tokens, so the 16,384 ceiling is ours,
# not the model's, and a truncated call can simply be retried with more of it.
_RETRY_MAX_TOKENS = 49_152
_TRUNCATION_MARKER = "length limit was reached"


def _call_llm(system_prompt: str, user_prompt: str,
              max_tokens: int | None = None) -> str:
    target = client if max_tokens is None else client.bind(max_tokens=max_tokens)
    response = target.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return response.content


# ==========================================
# 4. Helpers: JSON cleanup & schema validation
# ==========================================
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_code_fences(raw_output: str) -> str:
    return _FENCE_RE.sub("", raw_output.strip()).strip()


def _validate_schema(parsed: dict) -> str | None:
    """Return an error message if the parsed JSON doesn't match the expected
    skill-extraction schema, or None if it's valid."""
    if not isinstance(parsed, dict):
        return "Top-level JSON must be an object."
    for key in ("technical_skills", "soft_skills", "languages"):
        values = parsed.get(key)
        if not isinstance(values, list):
            return f"Missing or invalid '{key}' list."
        if not all(isinstance(v, str) for v in values):
            return f"'{key}' must contain only strings."
    return None


def _call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    """Call the LLM and parse its response as JSON (stripping code fences if
    needed). Returns {"error": ...} on any failure, otherwise the parsed dict.
    Shared by every prompt in this module — schema validation is the caller's
    responsibility, since each phase has a different expected shape."""
    try:
        raw_output = _call_llm(system_prompt, user_prompt)
    except Exception as e:
        if _TRUNCATION_MARKER not in str(e):
            return {"error": f"LLM call failed: {str(e)}"}
        # Ran out of budget while still thinking. Give it three times the room
        # once before giving up; the failure is intermittent, not deterministic.
        try:
            raw_output = _call_llm(system_prompt, user_prompt,
                                   max_tokens=_RETRY_MAX_TOKENS)
        except Exception as retry_error:
            if _TRUNCATION_MARKER in str(retry_error):
                return {"error": "The model spent its whole budget reasoning "
                                 "and never wrote an answer, twice in a row. "
                                 "Please try again — and if it keeps failing, "
                                 "trim the text to the responsibilities and "
                                 "requirements, since benefits and application "
                                 "instructions are what set this off."}
            return {"error": f"LLM call failed: {str(retry_error)}"}

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        cleaned = _strip_code_fences(raw_output)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"error": "Model did not return valid JSON.", "raw_output": raw_output}


# ==========================================
# 5. Public functions: extract_skills (Phase 1) / extract_jd_requirements (Phase 2)
# ==========================================
def _run_extraction(system_prompt: str, user_prompt: str) -> dict:
    """Shared LLM-call + JSON-parse + schema-validation pipeline used by both
    extract_skills() and extract_jd_requirements()."""
    parsed = _call_llm_json(system_prompt, user_prompt)
    if "error" in parsed:
        return parsed

    schema_error = _validate_schema(parsed)
    if schema_error:
        return {"error": f"Model output failed schema validation: {schema_error}", "raw_output": parsed}

    return parsed


# ==========================================
# 5b. Text normalisation (Arabic ingestion)
# ==========================================
# Arabic PDFs do not store the letters you see. They store *presentation
# forms* — the contextual glyph variants a font uses to join letters — from
# the Unicode blocks U+FB50-FDFF and U+FE70-FEFF. `pypdf` returns those
# codepoints verbatim, so the text looks correct on screen but is a different
# string entirely: the seen in "اسم" comes back as U+FEB3, not U+0633.
#
# This was previously recorded as `pypdf` garbling Arabic and was the stated
# reason an Arabic CV could not be added to the F1 gold set. Measured on two
# real Arabic CVs, the text is not garbled at all — word order is correct, and
# NFKC normalisation converts every presentation form back to its base letter
# (1391 forms -> 0 on the larger CV). What NFKC leaves behind is a handful of
# Persian-range letters that fonts substitute for Arabic ones, mapped below.
#
# Normalising here rather than in the scoring layer matters: exact-match gold
# annotation, entity counting and skill matching all compare strings, and all
# three break silently if two spellings of the same letter reach them.
_PRESENTATION_FORMS = re.compile(r"[ﭐ-﷿ﹰ-﻿]")

# Font substitutions that NFKC does not undo, because these are legitimately
# distinct codepoints in Persian/Urdu — they are simply the wrong letter here.
_ARABIC_LETTER_FIXES = str.maketrans({
    "ی": "ي",   # U+06CC Farsi yeh      -> Arabic yeh
    "ک": "ك",   # U+06A9 Keheh          -> Arabic kaf
    "ھ": "ه",   # U+06BE Heh doachashmee-> Arabic heh
    "ە": "ه",   # U+06D5 Ae             -> Arabic heh
})

# Tatweel is decoration that stretches a joining line; harakat are optional
# vowel marks. Both are invisible to meaning and fatal to string matching.
_ARABIC_DECORATION = re.compile(r"[ـً-ْٰۖ-ۭ]")


def normalize_text(text: str) -> str:
    """Make text safe to compare, without changing what it says.

    Applied at ingestion so every later stage — extraction, gap analysis,
    entity counting, gold-set matching — sees one spelling of each letter.
    Latin text is unaffected beyond NFKC's own ligature folding (fi -> fi).

    Note this is the conservative pass: it does NOT fold hamza forms
    (أ/إ/آ -> ا) or ta marbuta, because those change the spelling of a word
    rather than its encoding. Aggressive folding belongs in the matching
    layer, not here.
    """
    if not text:
        return text
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ARABIC_LETTER_FIXES)
    return _ARABIC_DECORATION.sub("", text)


# A CV written in Arabic must produce an Arabic interview. Detection is done
# once, here, and the answer is threaded through every later stage rather than
# re-guessed — so the questions, the evaluation and the feedback cannot end up
# disagreeing about what language the candidate is being interviewed in.
#
# Counted rather than modelled: no library, no API call, no failure mode. The
# ratio is taken over *letters only*, because a CV is full of technology names,
# emails and dates in Latin script even when it is written in Arabic — counting
# every character would misread an Arabic CV as an English one.
_ARABIC_RANGE = re.compile(r"[؀-ۿ]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

# Chosen low on purpose. An Arabic CV that lists its skills in English can
# easily be only a third Arabic by letter count, while an English CV is
# essentially 0% Arabic — so the two populations are nowhere near this line.
_ARABIC_LANGUAGE_THRESHOLD = 0.20


def detect_language(text: str) -> str:
    """Return "ar" or "en" for a document. Defaults to "en" on empty input."""
    if not text:
        return "en"
    letters = _LETTER.findall(text)
    if not letters:
        return "en"
    arabic = sum(1 for c in letters if _ARABIC_RANGE.match(c))
    return "ar" if arabic / len(letters) >= _ARABIC_LANGUAGE_THRESHOLD else "en"


def extract_skills(document_text: str, document_type: str = "CV") -> dict:
    """
    Phase 1: Extract the skills a candidate has, from their CV text.

    Args:
        document_text: raw CV text content (already extracted from PDF, plain text, etc.)
        document_type: kept for backward compatibility; this function always
            uses the CV-specific prompt (CV_SKILL_EXTRACTION_PROMPT).

    Returns:
        dict matching the schema in CV_SKILL_EXTRACTION_PROMPT, or an
        error dict if parsing/calling the LLM fails.
    """
    if not document_text or not document_text.strip():
        return {"error": "Empty document_text provided."}

    user_prompt = f"Text:\n{normalize_text(document_text)}"
    return _run_extraction(CV_SKILL_EXTRACTION_PROMPT, user_prompt)


def extract_jd_requirements(jd_text: str) -> dict:
    """
    Phase 2: Extract the skills a Job Description requires from the candidate.

    Args:
        jd_text: raw Job Description text.

    Returns:
        dict matching the schema in JD_REQUIREMENT_EXTRACTION_PROMPT, or an
        error dict if parsing/calling the LLM fails.
    """
    if not jd_text or not jd_text.strip():
        return {"error": "Empty jd_text provided."}

    user_prompt = f"Text:\n{normalize_text(jd_text)}"
    return _run_extraction(JD_REQUIREMENT_EXTRACTION_PROMPT, user_prompt)


# ==========================================
# 6. Phase 2 helper: CV vs Job Description skill-gap
# ==========================================
_SEMANTIC_MODEL = None
_GAP_SIMILARITY_THRESHOLD = 0.6

# Languages are extracted with their proficiency wording attached, because the
# candidate should see "Fluent in written and spoken English" rather than a
# bare "English". But that wording is noise when MATCHING: a JD asking for
# "Fluent in written and spoken English" scores only 0.54 against a CV that
# lists "English" — below the 0.6 gate — so the system would report that the
# candidate does not speak English at all. Stripping the proficiency words on
# both sides before comparing fixes the match without touching what is shown.
#
# This is the same principle as normalize_text(): the conservative form is
# stored, and aggressive folding happens in the matching layer where it belongs.
_PROFICIENCY_WORDS = re.compile(
    r"\b(?:fluent(?:ly)?|fluency|native|bilingual|mother\s+tongue|proficien\w*|"
    r"advanced|intermediate|beginner|basic|elementary|conversational|working|"
    r"professional|excellent|very|good|level|command|skills?|knowledge|"
    r"spoken|written|writing|reading|speaking|understanding|in|and|of|a|an|the)\b",
    re.IGNORECASE)
# Bounded on both sides by non-Arabic, so a short term cannot be cut out of
# the middle of a longer word the way a bare alternation would.
_AR_PROFICIENCY_TERMS = (
    "بطلاقة", "طلاقة", "لغة الأم", "اللغة الأم", "الأم",
    "متقدم", "متقدّم", "متوسط", "متوسّط", "مبتدئ", "ممتاز",
    "جيد", "جيّد", "ضعيف", "محادثة", "كتابة", "قراءة",
    "تحدث", "تحدّث", "مستوى", "إلمام", "أساسي", "لغة",
)
_PROFICIENCY_WORDS_AR = re.compile(
    "|".join(f"(?<![{_ARABIC_RANGE.pattern[1:-1]}]){re.escape(t)}"
             f"(?![{_ARABIC_RANGE.pattern[1:-1]}])"
             for t in _AR_PROFICIENCY_TERMS))


def _language_core(text: str) -> str:
    """The language name with its proficiency wording removed.

    Falls back to the original string when stripping leaves nothing, so an
    entry that is only a qualifier is compared as written rather than as "".
    """
    stripped = _PROFICIENCY_WORDS.sub(" ", text)
    stripped = _PROFICIENCY_WORDS_AR.sub(" ", stripped)
    stripped = re.sub(r"[^\w؀-ۿ]+", " ", stripped).strip()
    return stripped or text


def _get_semantic_model():
    """Lazy-load the sentence-transformers model (only needed for skill-gap matching)."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL


def compute_skill_gap(candidate_result: dict, jd_result: dict) -> dict:
    """
    Compare a candidate's extracted skills (from extract_skills() on a CV)
    against a job description's required skills (from extract_skills() on a
    Job Description), and return the skills the JD asks for that the
    candidate doesn't appear to have.

    Matching is semantic (via all-MiniLM-L6-v2 embeddings), not exact-string,
    since a JD saying "Cloud experience" should count as satisfied by a CV
    listing "AWS", and "Python" should match "python" regardless of wording —
    unlike the exact-match methodology used for Phase 1 F1 benchmarking.

    Returns:
        {
          "missing_technical_skills": [...],
          "missing_soft_skills": [...],
          "missing_languages": [...],
        }
    """
    model = _get_semantic_model()
    gap = {}

    for category, gap_key in [
        ("technical_skills", "missing_technical_skills"),
        ("soft_skills", "missing_soft_skills"),
        ("languages", "missing_languages"),
    ]:
        jd_skills = jd_result.get(category, [])
        candidate_skills = candidate_result.get(category, [])

        if not jd_skills:
            gap[gap_key] = []
            continue
        if not candidate_skills:
            gap[gap_key] = list(jd_skills)
            continue

        # Languages do not go through the embedding model at all. Language
        # names are a closed vocabulary of proper nouns, and the model places
        # them close together precisely BECAUSE they are all languages:
        # measured, French/English = 0.64 and German/Spanish = 0.60, both above
        # the 0.6 gate. Semantic similarity would therefore report that a
        # candidate who speaks only English satisfies a requirement for French.
        # Comparing the language name as a string is both correct and exact.
        if category == "languages":
            have = {_language_core(x).casefold() for x in candidate_skills}
            gap[gap_key] = [
                x for x in jd_skills
                if not any(core == h or core in h or h in core
                           for h in have
                           for core in (_language_core(x).casefold(),))
            ]
            continue

        jd_terms, candidate_terms = jd_skills, candidate_skills

        jd_embeddings = model.encode(jd_terms, convert_to_tensor=True)
        candidate_embeddings = model.encode(candidate_terms, convert_to_tensor=True)

        from sentence_transformers import util
        similarity_matrix = util.cos_sim(jd_embeddings, candidate_embeddings)

        missing = []
        for i, jd_skill in enumerate(jd_skills):
            best_match_score = similarity_matrix[i].max().item()
            if best_match_score < _GAP_SIMILARITY_THRESHOLD:
                missing.append(jd_skill)
        gap[gap_key] = missing

    return gap


# ==========================================
# 6a-2. Output language
# ==========================================
# One instruction block, appended to every generating and judging prompt, so
# the rule is stated once instead of drifting between eight prompts.
#
# The technology-name carve-out is load-bearing, not stylistic. Half the
# Answerability Score is `content_entities`, which counts how many known skills
# a question names by matching against the extracted skill vocabulary — and
# that vocabulary holds "Kubernetes", not "كوبرنيتس". A translated technology
# name scores zero, so the question fails the gate and is regenerated forever.
# Transliterating terms would silently break the scoring layer, not just read
# oddly.
_LANGUAGE_DIRECTIVE = {
    "ar": """
LANGUAGE: Write your output in ARABIC. The candidate's CV is in Arabic, so the
interview is conducted in Arabic.

- Write questions, feedback and any free text in clear Modern Standard Arabic.
- Keep technology, tool, framework and language names in their ORIGINAL LATIN
  script: write Kubernetes, Python, ROS 2, FreeRTOS, SPI — never transliterate
  them into Arabic letters. This is how engineers actually write and speak, and
  the rest of the system matches on those exact names.
- JSON keys stay in English exactly as the schema specifies. Only the VALUES
  are in Arabic.
""",
    "en": "",
}


def _with_language(prompt: str, language: str) -> str:
    """Append the output-language rule to a prompt. English adds nothing, so
    English behaviour is byte-identical to before this existed."""
    return prompt + _LANGUAGE_DIRECTIVE.get(language, "")


# ==========================================
# 6b. System Prompt — Phase 2 (Strategic Question Generation)
# ==========================================
QUESTION_GENERATION_PROMPT = """You are an expert technical interviewer designing
questions for a mock interview.

You are given three things:
1. The skills the job (JD) requires.
2. The skills the candidate's CV shows they already have.
3. The skill gap — JD-required skills the candidate does NOT appear to have.

Your task: generate exactly {num_questions} interview questions, split the way
a real interviewer splits them — part on what the candidate is missing, part
on what they already claim:

- **Exactly {gap_count} questions on GAP skills** (skills the role requires
  that the candidate does not appear to have). Pick the gap skills that are
  most central to the role — the ones a candidate genuinely could not do the
  job without — not peripheral "nice to have" items. These questions test
  transferable experience, awareness of the topic, and how the candidate
  reasons about something they have not used in production.
- **Exactly {existing_count} questions on EXISTING skills** (skills the CV
  already claims). These matter just as much: they are where the candidate's
  real depth is measured. An interview that only probes weaknesses reveals
  nothing about what the person is actually good at.

If one of the two lists holds fewer distinct skills than its quota, take the
shortfall from the other list rather than repeating a skill.

HOW MANY OF EACH KIND:

- **Exactly {soft_count} of the {num_questions} questions must target a SOFT /
  behavioural skill.** The rest are technical. This quota is fixed: without it
  the number of behavioural questions drifts with whatever the two skill lists
  happen to contain, and one interview gets one while another gets three — so
  the interviews are not comparable to each other.
- Draw the soft skills from the JD and CV lists, gap ones first. If together
  they hold fewer than {soft_count} distinct soft skills, ask about the
  soft skills that exist and give the remainder to technical questions rather
  than inventing a soft skill nobody mentioned.
- Where the two splits disagree — the gap/existing split above and this one —
  **this quota wins**, and the gap/existing balance is met as closely as the
  remaining questions allow.

TWO KINDS OF SKILL NEED TWO KINDS OF QUESTION:

- **Technical skills** — ask about a concrete task, mechanism, trade-off or
  scenario involving that technology.

- **Soft / behavioural skills** (communication, teamwork, leadership, time
  management, adaptability, working under pressure) — these CANNOT be tested
  with a knowledge question. "What makes communication effective?" measures
  vocabulary, not behaviour; any candidate can recite the right words.
  Use the **STAR framework** that real HR interviewers use, and ask, inside a
  single spoken question, for all four parts:
    S — a SPECIFIC past situation ("a time when...", "a situation where...")
    T — the candidate's own task or role in it
    A — the actions THEY personally took
    R — the result or outcome
  Anchor it in the past and in one concrete episode. Never hypothetical
  ("what would you do if..."), never general ("how do you usually handle
  conflict?") — a hypothetical answer is an opinion, and only a real episode
  is evidence.
  Example shape: "Describe a specific situation where you had to resolve a
  disagreement inside your team — what was your role, what did you personally
  do, and how did it turn out?"

Rules:

1. Each question must be self-contained and answerable in a spoken interview
   — no "see attached", no multi-part essay prompts.
2. Each question MUST explicitly name, inside its own text, the skill given
   in its "targets_skill" field. A question about RTOS that never writes
   "RTOS" is invalid.
3. Stay focused: one question probes one skill. Do not pad a question with
   unrelated skill names just to make it look technical.
4. Phrase each question with a clear interrogative ("What/How/Why/Which...")
   or a directive verb ("Describe/Explain/Compare/Walk me through"), and ask
   about a concrete task or scenario rather than a vague invitation to talk
   (avoid "Tell me about X").
5. Do not repeat the same targets_skill across questions unless there are
   fewer distinct skills available than {num_questions}.
6. Return only JSON — no preamble, no explanation, no markdown code fences.

Output format:

{{
  "questions": [
    {{
      "question": "string",
      "targets_skill": "string (must also appear inside the question text)"
    }}
  ]
}}
"""


def _validate_questions_schema(parsed: dict) -> str | None:
    if not isinstance(parsed, dict):
        return "Top-level JSON must be an object."
    questions = parsed.get("questions")
    if not isinstance(questions, list) or not questions:
        return "Missing or empty 'questions' list."
    for i, q in enumerate(questions):
        if not isinstance(q, dict) or not {"question", "targets_skill"}.issubset(q):
            return f"Question at index {i} is missing 'question' or 'targets_skill'."
        if not isinstance(q["question"], str) or not isinstance(q["targets_skill"], str):
            return f"Question at index {i} has non-string 'question'/'targets_skill'."
    return None


# ==========================================
# 6c. Answerability Score (AS) — computed programmatically
# ==========================================
# The three sub-criteria and the 0.5/0.3/0.2 weighting are defined by the
# project author in the evaluation framework. Following the methodology of
# Nema & Khapra (EMNLP 2018), "Towards a Better Metric for Evaluating Question
# Generation Systems" — where answerability is derived by *counting* elements
# of the question (named entities, question types, content words) rather than
# by asking a language model to judge it — each sub-score below is computed
# deterministically from the question text. Nothing here relies on the
# generating model rating its own output, which avoids self-preference bias
# and makes every score independently reproducible.
_AS_WEIGHTS = {"content_entities": 0.5, "context_clarity": 0.3, "task_specificity": 0.2}
_ANSWERABILITY_GATE = 0.7

_INTERROGATIVES = ("what", "how", "why", "which", "when", "where", "who")
_DIRECTIVE_VERBS = ("describe", "explain", "compare", "walk me through",
                    "walk us through", "outline", "discuss")
_TASK_VERBS = ("implement", "design", "debug", "optimize", "optimise", "deploy",
               "configure", "build", "troubleshoot", "integrate", "test",
               "handle", "resolve", "identify", "measure", "profile",
               "refactor", "migrate", "scale", "secure", "validate",
               "diagnose", "write", "structure", "ensure", "choose", "select",
               "manage", "process", "approach", "apply", "maintain", "monitor",
               "automate", "containerize", "containerise", "analyze", "analyse",
               "evaluate", "reduce", "improve", "extend", "connect", "trace",
               "verify", "benchmark", "tune", "prevent", "set up")

# A question can ask for something concrete without using an imperative
# technical verb — by asking for a comparison, a worked scenario, or a
# specific past example. These phrasings count as specific too, so that
# conceptual and behavioural questions are not unfairly penalised.
_SPECIFICITY_PHRASES = (
    "specific example", "concrete example", "for example", "a time when",
    "a time you", "a situation where", "scenario", "difference between",
    "differences between", "compare", "versus", " vs ", "trade-off",
    "tradeoff", "what steps", "which steps", "how would you", "how do you",
    "how have you", "walk me through", "walk us through", "step by step",
)

# Filler words that appear inside skill labels ("SQL databases", "Strong
# communication skills") and would otherwise match almost any question text.
_TARGET_STOPWORDS = {
    "and", "or", "the", "of", "in", "with", "for", "to", "on", "at", "by",
    "using", "skills", "skill", "experience", "knowledge", "understanding",
    "familiarity", "strong", "excellent", "proven", "ability", "proficiency",
    "proficient", "basic", "advanced", "solid", "good", "fluent", "native",
}

_MAX_CLEAR_QUESTION_WORDS = 60

# A floor that catches degenerate output, not a judgement on good questions:
# measured across 20 generated questions the shortest was 20 words, so neither
# language's real output comes near it.
#
# The Arabic figure is scaled from the English one by the measured ratio of
# median question length (26 / 31 = 0.85), because Arabic writes the same
# content in fewer tokens — it attaches the article, prepositions and pronouns
# to the word instead of separating them. Keeping one number for both would
# hold Arabic to a stricter standard for saying the same thing.
_MIN_DETAILED_QUESTION_WORDS = 12
_MIN_DETAILED_QUESTION_WORDS_AR = 10

# ── Arabic equivalents ────────────────────────────────────────────────────
# Measured need: an Arabic interview generated correctly-formed questions that
# every criterion scored near zero, because each one matched English strings.
# The gate rejected 10 of 10. Only the vocabulary is language-specific — the
# criteria, the 0.5/0.3/0.2 weights and the 0.7 gate are unchanged, so the two
# languages stay on one scale and the report keeps one equation.
_INTERROGATIVES_AR = ("ما", "ماذا", "كيف", "لماذا", "أي", "أية", "متى", "أين",
                      "من", "هل", "كم", "بماذا", "لماذا")
_DIRECTIVE_VERBS_AR = ("اشرح", "صف", "وضح", "وضّح", "قارن", "اذكر", "بيّن",
                       "بين", "حلل", "حلّل", "عدد", "عدّد", "استعرض", "ناقش",
                       "افترض", "تخيل", "تخيّل")
# Three forms are listed for each action, because Arabic marks person and tense
# with prefixes as well as suffixes and an English stem+suffix search finds
# none of them. Measured: every question in a real Arabic run scored
# task_specificity = 0.0 while asking a perfectly concrete task.
#   - imperative / present : نفّذ، تنفذ، تتعامل   (س is handled as a prefix)
#   - first-person past    : قمت، عملت، استخدمت   ("how did you...")
#   - verbal noun (masdar) : إعداد، تحليل، تنظيم  — in Arabic the masdar is how
#     a task is normally named at all ("إعداد بيئة تطوير" = "setting up a dev
#     environment"), so omitting it misses the most common phrasing.
_TASK_VERBS_AR = (
    # imperative / present
    "نفذ", "نفّذ", "صمم", "صمّم", "برمج", "اكتب", "طور", "طوّر", "حسّن", "حسن",
    "عالج", "شخص", "شخّص", "اختبر", "ادمج", "اضبط", "هيئ", "هيّئ", "أنشئ",
    "انشئ", "ابن", "تعامل", "حدد", "حدّد", "راقب", "تحقق", "تحقّق", "اختر",
    "أدر", "ادر", "طبق", "طبّق", "استخدم", "تستخدم", "تقوم", "تعالج",
    "تتعامل", "تحل", "تصمم", "تكتب", "تنفذ", "تختار", "تضبط", "تدير",
    "تشخص", "تختبر", "تراقب", "تدمج", "تبدأ", "تتطلب", "تكتشف", "تصلح",
    # first-person / second-person past
    "قمت", "عملت", "استخدمت", "صممت", "نفذت", "طورت", "حللت", "اخترت",
    "بنيت", "كتبت", "عالجت", "تعاملت", "واجهت", "أدرت", "ادرت", "اختبرت",
    "ضبطت", "حددت", "أنشأت", "انشأت", "دمجت", "حسنت", "شخصت",
    # verbal nouns
    "إعداد", "اعداد", "تحليل", "تنظيم", "تشخيص", "تصميم", "تنفيذ", "معالجة",
    "اختبار", "ضبط", "بناء", "تطوير", "دمج", "مراقبة", "تحسين", "كتابة",
    "اختيار", "استخدام", "تصحيح", "استرجاع", "تخزين", "برمجة", "أتمتة",
    "اتمتة", "نشر", "توثيق", "تكامل", "قياس",
)

# Behavioural questions in Arabic ask for the same four STAR components; only
# the wording changes. The Situation anchor stays mandatory for the same
# reason: "ماذا لو" invites a hypothetical, and a hypothetical is an opinion.
_STAR_CUES_AR = {
    "situation": ("موقف", "موقفا", "موقفاً", "حادثة", "مرة", "عندما",
                  "حين", "في إحدى", "مثال محدد", "مثالا محددا", "مثالاً محدداً",
                  "تجربة مررت", "سبق أن", "سبق لك"),
    "task": ("دورك", "مسؤوليتك", "مهمتك", "كنت مسؤول", "كنت مسؤولا",
             "كنت مسؤولاً", "المطلوب منك", "ما كان دورك"),
    "action": ("ماذا فعلت", "ما الذي فعلت", "ما الخطوات", "كيف تصرفت",
               "كيف تعاملت", "كيف عالجت", "ما الإجراءات", "الإجراءات التي",
               "ماذا قمت", "كيف قمت", "ما الذي قمت", "ماذا عملت",
               "كيف نفذت", "كيف حللت", "الخطوات التي اتخذت"),
    "result": ("النتيجة", "ماذا حدث", "كيف انتهى", "ما الذي نتج", "الأثر",
               "في النهاية", "ماذا تعلمت"),
}

_SPECIFICITY_PHRASES_AR = (
    "مثال محدد", "مثالا محددا", "مثالاً محدداً", "على سبيل المثال", "مثال عملي",
    "الفرق بين", "الاختلاف بين", "مقارنة بين", "قارن بين", "سيناريو",
    "ما الخطوات", "ما هي الخطوات", "كيف تقوم", "كيف تتعامل", "كيف تعالج",
    "خطوة بخطوة", "متى تختار", "أيهما تختار", "في أي حالة",
)

# Hesitation markers in spoken Arabic. "يعني" is included because it is the
# dominant filler in Libyan and Levantine speech, exactly parallel to "you
# know" in the English list.
_FILLER_PATTERNS_AR = ("يعني", "امم", "اممم", "آه", "اه", "ااه", "مممم",
                       "شن نقول", "كيف نقول", "بصراحة يعني")

_TARGET_STOPWORDS_AR = {
    "و", "أو", "او", "في", "من", "على", "إلى", "الى", "مع", "عن", "ال",
    "مهارات", "مهارة", "خبرة", "معرفة", "إلمام", "المام", "قوية", "ممتازة",
    "جيدة", "أساسية", "اساسية", "متقدمة", "القدرة", "قدرة", "إتقان", "اتقان",
    "استخدام", "نظام", "أنظمة", "انظمة", "لغة", "إدارة", "ادارة",
}

# One lookup keyed by language, so a scorer reads its vocabulary instead of
# closing over module-level English tuples.
#
# Built on first use rather than at import: the English STAR cues and filler
# patterns are defined further down the file, next to the code that motivates
# them, and moving them up here purely to satisfy definition order would put
# them far from their explanation.
_LEXICON = None


def _build_lexicon() -> dict:
    return {
        "en": {
            "interrogatives": _INTERROGATIVES,
            "directive_verbs": _DIRECTIVE_VERBS,
            "task_verbs": _TASK_VERBS,
            "specificity_phrases": _SPECIFICITY_PHRASES,
            "star_cues": _STAR_CUES,
            "fillers": _FILLER_PATTERNS,
            "target_stopwords": _TARGET_STOPWORDS,
            "question_marks": ("?",),
            "min_detailed_words": _MIN_DETAILED_QUESTION_WORDS,
        },
        "ar": {
            "interrogatives": _INTERROGATIVES_AR,
            "directive_verbs": _DIRECTIVE_VERBS_AR,
            "task_verbs": _TASK_VERBS_AR,
            "specificity_phrases": _SPECIFICITY_PHRASES_AR,
            "star_cues": _STAR_CUES_AR,
            "fillers": _FILLER_PATTERNS_AR,
            "target_stopwords": _TARGET_STOPWORDS_AR,
            # Arabic uses U+061F. A question ending in the wrong glyph scored
            # as if it were not a question at all.
            "question_marks": ("؟", "?"),
            "min_detailed_words": _MIN_DETAILED_QUESTION_WORDS_AR,
        },
    }


def _lex(language: str, key: str):
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = _build_lexicon()
    return _LEXICON.get(language, _LEXICON["en"])[key]


# Arabic attaches its conjunctions, prepositions, the future marker and the
# definite article directly to the word (وبالبرمجة = و + بـ + ال + برمجة،
# ستتعامل = سـ + تتعامل), so a strict boundary match finds nothing while a
# plain substring match finds far too much. These are the single-letter
# clitics that may precede a word.
_AR_PREFIXES = "وفبكلس"
_ARABIC_CHAR = r"؀-ۿ"


def _word_boundary_search(needle: str, haystack_lower: str):
    """Search for `needle` as a standalone token, so that e.g. "SQL" does not
    match inside "MySQL" and "C" does not match inside "Cortex".

    For Arabic needles the left boundary also allows the attached clitics
    (و، ف، ب، ك، ل) and the definite article, since Arabic writes them joined
    to the following word rather than separated by a space.
    """
    escaped = re.escape(needle)
    if re.search(f"[{_ARABIC_CHAR}]", needle):
        pattern = (f"(?<![{_ARABIC_CHAR}])[{_AR_PREFIXES}]?(?:ال)?"
                   + escaped + f"(?![{_ARABIC_CHAR}])")
    else:
        pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
    return re.search(pattern, haystack_lower)


def _verb_search(verb: str, haystack_lower: str):
    """Match a task verb including its common inflections, so that "deploy"
    also matches "deployed"/"deploying" and "optimize" also matches
    "optimizing" (a trailing silent -e is dropped before the suffix)."""
    stem = verb[:-1] if verb.endswith("e") else verb
    pattern = r"(?<![a-z0-9])" + re.escape(stem) + r"(?:e|es|ed|ing|s)?(?![a-z0-9])"
    return re.search(pattern, haystack_lower)


def _build_skill_vocabulary(jd_result: dict, cv_result: dict) -> set:
    vocab = set()
    for source in (jd_result, cv_result):
        for key in ("technical_skills", "soft_skills", "languages"):
            for skill in source.get(key, []):
                if isinstance(skill, str) and skill.strip():
                    vocab.add(skill.strip())
    return vocab


def _count_named_entities(question_text: str, vocabulary: set) -> int:
    """Count DISTINCT known skills named in the question. Matching runs
    longest-first and masks each match, so a short skill nested inside a
    longer one ("SQL" inside "SQL databases") is not counted twice."""
    text = question_text.lower()
    count = 0
    for skill in sorted(vocabulary, key=len, reverse=True):
        match = _word_boundary_search(skill.lower(), text)
        if match:
            count += 1
            text = text[:match.start()] + " " * (match.end() - match.start()) + text[match.end():]
    return count


def _target_is_named(question_text: str, targets_skill: str,
                     language: str = "en") -> bool:
    """Does the question actually name the skill it claims to target?

    Checked in two passes, because a question normally names one natural
    variant of a label rather than the label verbatim:
      1. the full label and its parts around separators — "ROS / ROS 2",
         "C/C++", "FPGA (VHDL)";
      2. the individual significant words — a question saying "SQL query"
         does name the skill "SQL databases". Filler words are excluded so
         that a label like "Modeling and Simulation" can't match on "and".
    """
    text = question_text.lower()

    # The dot is a separator here as well as the slash and comma. Without it a
    # target of "React.js" is searched for as one literal token and never
    # matches an answer that says "React", which cost four substantive answers
    # (126-476 words each) a score of zero on a real interview.
    candidates = [targets_skill] + re.split(r"[/,().]", targets_skill)
    for candidate in candidates:
        candidate = candidate.strip().lower()
        if candidate and _word_boundary_search(candidate, text):
            return True

    # An Arabic label carries Arabic filler ("مهارات", "خبرة في") as well as
    # English, and a mixed label like "إدارة قواعد بيانات SQL" contains both.
    stopwords = set(_TARGET_STOPWORDS) | set(_lex(language, "target_stopwords"))
    for word in re.split(r"[\s/,().\-]+", targets_skill.lower()):
        word = word.strip()
        if len(word) >= 3 and word not in stopwords and _word_boundary_search(word, text):
            return True
    return False


def _score_content_entities(question_text: str, targets_skill: str, vocabulary: set,
                            language: str = "en") -> float:
    # A question that never names its own target skill is genuinely broken.
    if not _target_is_named(question_text, targets_skill, language):
        return 0.0
    # Naming the target earns the base score; additional distinct entities add
    # a small bonus. The base is deliberately generous so that a well-focused
    # single-skill question is not punished for staying focused.
    extra_entities = max(0, _count_named_entities(question_text, vocabulary) - 1)
    return min(1.0, 0.6 + 0.2 * extra_entities)


def _score_context_clarity(question_text: str, language: str = "en") -> float:
    text = question_text.lower()
    score = 0.4
    if any(_word_boundary_search(w, text) for w in _lex(language, "interrogatives")) or \
       any(v in text for v in _lex(language, "directive_verbs")):
        score += 0.3
    if question_text.strip().endswith(_lex(language, "question_marks")):
        score += 0.3
    if len(question_text.split()) > _MAX_CLEAR_QUESTION_WORDS:
        score -= 0.3  # run-on questions are harder to parse in a spoken interview
    return max(0.0, min(1.0, score))


def _score_task_specificity(question_text: str, language: str = "en") -> float:
    text = question_text.lower()
    if language == "ar":
        # Arabic verbs inflect by prefix, not by suffix, so the English
        # stem+suffix search would miss every one of them. The lexicon lists
        # the inflected forms an interviewer actually writes instead.
        has_task_verb = any(_word_boundary_search(v, text)
                            for v in _lex(language, "task_verbs"))
    else:
        has_task_verb = any(_verb_search(v, text) for v in _lex(language, "task_verbs"))
    has_specific_ask = any(p in text for p in _lex(language, "specificity_phrases"))
    if not has_task_verb and not has_specific_ask:
        return 0.0  # no concrete task asked — "Tell me about X"
    return 1.0 if len(question_text.split()) >= _lex(language, "min_detailed_words") else 0.5


# STAR (Situation, Task, Action, Result) is the standard behavioural-interview
# framework in HR practice. It is used here as the *grounding* for soft-skill
# questions, exactly as the computed skill gap grounds the technical ones: a
# behavioural question is not invented freely, it is built to elicit the four
# STAR components. Each cue list below is the wording an interviewer uses to
# request that component.
_STAR_CUES = {
    "situation": ("a time when", "a time you", "a situation where",
                  "a situation in which", "an occasion when", "an instance where",
                  "describe a time", "tell me about a time", "a specific example",
                  "a specific situation", "an example of when", "when you had to",
                  "when you were"),
    "task": ("your role", "your responsibility", "your task", "you were responsible",
             "what were you", "you had to", "expected of you", "your part"),
    "action": ("what did you do", "what you did", "what steps", "how did you handle",
               "how did you approach", "how did you respond", "how did you deal",
               "what actions", "actions you took", "you personally do"),
    # Matched on the verb phrase rather than the whole sentence, because the
    # pronoun changes with context: an interviewer opening a topic says "how
    # did IT turn out", following one up says "how did THAT turn out". A cue
    # list tied to one pronoun silently rejects the other — which it did, on
    # a correctly-formed follow-up, during testing.
    "result": ("outcome", "result", "what happened", "turn out", "turned out",
               "end up", "ended up", "how did it end", "how did that end",
               "impact", "in the end", "what came of"),
}


def _score_star_elicitation(question_text: str,
                            situation_established: bool = False,
                            language: str = "en") -> float:
    """For a behavioural question this plays the role entity-counting plays for
    a technical one: it measures whether the question supplies the frame the
    candidate needs in order to give usable evidence.

    An OPENING question is scored as the fraction of the four STAR components
    it asks for, with the Situation anchor MANDATORY — a behavioural question
    not tied to one specific past episode is a hypothetical, and a hypothetical
    answer is an opinion rather than evidence of past behaviour.

    A FOLLOW-UP (`situation_established=True`) is scored differently, because
    the episode already exists in the exchange that preceded it. Requiring it
    to re-establish the Situation would reject the natural interviewer move
    ("and how did that turn out?"), and scoring it out of four would reject any
    follow-up for the crime of asking about one thing at a time. What it must
    do instead is request at least one concrete STAR component.
    """
    text = question_text.lower()
    cue_sets = _lex(language, "star_cues")
    present = {c: any(cue in text for cue in cues) for c, cues in cue_sets.items()}
    if situation_established:
        return 1.0 if any(present.values()) else 0.0
    if not present["situation"]:
        return 0.0
    return round(sum(present.values()) / len(cue_sets), 3)


def _collect_labels(jd_result: dict, cv_result: dict, category: str) -> list:
    return [s for src in (jd_result, cv_result)
            for s in src.get(category, []) if isinstance(s, str) and s.strip()]


def _best_label_overlap(target: str, labels: list) -> tuple:
    """How strongly `target` matches any label in `labels`, as the comparable
    pair (exact_match, longest_matching_label_length). (0, 0) = no match.

    Two properties are deliberate:

    * Whole-word containment, not plain substring — otherwise the extracted
      technical skill "C" would match the soft label "Clear communication".
    * Exactness outranks length, so an exact match on a short label ("Teamwork")
      beats a partial overlap with a longer one. Comparing lengths alone let a
      long technical label swallow an exact soft one.
    """
    exact = 0
    longest = 0
    for label in labels:
        candidate = label.strip().lower()
        if candidate == target:
            exact = 1
            longest = max(longest, len(candidate))
        elif (_word_boundary_search(candidate, target)
                or _word_boundary_search(target, candidate)):
            longest = max(longest, len(candidate))
    return (exact, longest if exact == 0 else len(target))


def _is_soft_skill_target(targets_skill: str, jd_result: dict, cv_result: dict) -> bool:
    """Decide whether a question's target is a behavioural skill.

    The classification is not a fresh judgement: extraction (Phase 1) already
    sorted every skill into technical_skills / soft_skills, so this only has to
    find which list the target belongs to. It matches loosely in both
    directions, because the generating model may write "communication skills"
    where extraction produced "Excellent communication skills".

    Both lists are consulted and the STRONGER match wins, rather than returning
    True on the first soft hit. That ordering fixed a real misclassification:
    with a short extracted label like "Communication", one-directional matching
    made the technical skill "Communication protocols (CAN, SPI, UART)" look
    behavioural — and it would then have been asked about with a STAR question.
    Comparing against the technical list too, the longer, exact technical match
    outweighs the incidental one-word overlap.
    """
    target = targets_skill.strip().lower()
    if not target:
        return False

    soft_match = _best_label_overlap(target, _collect_labels(jd_result, cv_result, "soft_skills"))
    tech_match = _best_label_overlap(target, _collect_labels(jd_result, cv_result, "technical_skills"))
    return soft_match > tech_match


def _derive_is_gap_skill(targets_skill: str, gap_skills: list) -> bool:
    """Derive the gap flag from the *computed* skill gap rather than trusting
    the generating model's own claim, using the same matching threshold that
    produced the gap in the first place."""
    if not gap_skills:
        return False
    target_norm = targets_skill.strip().lower()
    if any(g.strip().lower() == target_norm for g in gap_skills):
        return True
    model = _get_semantic_model()
    from sentence_transformers import util
    target_embedding = model.encode([targets_skill], convert_to_tensor=True)
    gap_embeddings = model.encode(gap_skills, convert_to_tensor=True)
    return util.cos_sim(target_embedding, gap_embeddings).max().item() >= _GAP_SIMILARITY_THRESHOLD


def _compute_answerability(question_text: str, targets_skill: str, vocabulary: set,
                           is_soft_skill: bool,
                           situation_established: bool = False,
                           language: str = "en") -> dict:
    clarity = _score_context_clarity(question_text, language)
    specificity = _score_task_specificity(question_text, language)

    if is_soft_skill:
        # Counting named technical entities is meaningless for a behavioural
        # question, so the heaviest criterion measures the equivalent property
        # for THIS question type: does the question supply what the candidate
        # needs in order to answer it — i.e. does it request a STAR episode.
        # The weights themselves are untouched; only what fills the 0.5 slot
        # changes with the question type.
        entities = None
        star = _score_star_elicitation(question_text, situation_established, language)
        score = (_AS_WEIGHTS["content_entities"] * star
                 + _AS_WEIGHTS["context_clarity"] * clarity
                 + _AS_WEIGHTS["task_specificity"] * specificity)
    else:
        star = None
        entities = _score_content_entities(question_text, targets_skill,
                                           vocabulary, language)
        score = (_AS_WEIGHTS["content_entities"] * entities
                 + _AS_WEIGHTS["context_clarity"] * clarity
                 + _AS_WEIGHTS["task_specificity"] * specificity)

    return {
        "content_entities": entities,
        "star_elicitation": star,
        "context_clarity": round(clarity, 3),
        "task_specificity": round(specificity, 3),
        "answerability_score": round(score, 3),
        "passes_gate": score >= _ANSWERABILITY_GATE,
    }


# A real interview probes both sides: what the candidate is missing (to test
# awareness and transferable reasoning) and what they already claim (which is
# where genuine depth is measured). Asking only about gaps produces an
# interview the candidate fails entirely, yielding no signal about strengths.
_GAP_QUESTION_RATIO = 0.6

# Behavioural questions per interview. Left to the model's discretion this
# drifted with the skill lists — one real interview got a single behavioural
# question and another got three — which makes two candidates' reports
# incomparable on the soft-skill axis. Fixing the count fixes that.
#
# Three of eight keeps the interview technical-first while still giving the
# STAR rubric enough episodes to average over. Raising it without shortening
# the technical half would invert the balance of the interview.
SOFT_QUESTION_COUNT = 3


# Planned questions per interview, before the agent adds any follow-up.
# It was ten. Two real candidates each answered thirteen questions once
# follow-ups were added, and both sessions ran well over an hour; of four
# people who registered, two never finished one. Eight leaves room for the
# agent's probes inside MAX_TOTAL_QUESTIONS without the interview sprawling.
PLANNED_QUESTION_COUNT = 8


def generate_questions(jd_result: dict, cv_result: dict, skill_gap: dict,
                       num_questions: int = PLANNED_QUESTION_COUNT,
                       language: str = "en",
                       soft_count: int = SOFT_QUESTION_COUNT) -> dict:
    """
    Phase 2: Generate strategic interview questions balanced between the
    candidate's skill gap and the skills they already claim. The model only
    writes the questions; every score attached to them (Answerability Score,
    gate result, gap flag) is computed here in code, so the generator never
    grades its own work.

    Args:
        jd_result: output of extract_jd_requirements().
        cv_result: output of extract_skills().
        skill_gap: output of compute_skill_gap().
        num_questions: how many questions to generate in total. They are split
            by _GAP_QUESTION_RATIO between gap skills and existing skills.

    Returns:
        {
          "questions": [
            {"question", "targets_skill", "is_gap_skill", "content_entities",
             "context_clarity", "task_specificity", "answerability_score",
             "passes_gate"}
          ]
        }
        or {"error": ...} on failure.
    """
    gap_count = round(num_questions * _GAP_QUESTION_RATIO)
    existing_count = num_questions - gap_count
    # Never let the behavioural quota swallow the whole interview, however the
    # caller sets it.
    soft_count = max(0, min(soft_count, num_questions - 1))

    user_prompt = (
        f"JD required skills: {json.dumps(jd_result, ensure_ascii=False)}\n\n"
        f"Candidate skills: {json.dumps(cv_result, ensure_ascii=False)}\n\n"
        f"Skill gap (JD requires, candidate lacks): {json.dumps(skill_gap, ensure_ascii=False)}"
    )
    system_prompt = _with_language(
        QUESTION_GENERATION_PROMPT.format(
            num_questions=num_questions,
            gap_count=gap_count,
            existing_count=existing_count,
            soft_count=soft_count,
        ),
        language,
    )

    parsed = _call_llm_json(system_prompt, user_prompt)
    if "error" in parsed:
        return parsed

    schema_error = _validate_questions_schema(parsed)
    if schema_error:
        return {"error": f"Model output failed schema validation: {schema_error}", "raw_output": parsed}

    vocabulary = _build_skill_vocabulary(jd_result, cv_result)
    gap_skills = (skill_gap.get("missing_technical_skills", [])
                  + skill_gap.get("missing_soft_skills", [])
                  + skill_gap.get("missing_languages", []))

    for q in parsed["questions"]:
        is_soft = _is_soft_skill_target(q["targets_skill"], jd_result, cv_result)
        q["is_soft_skill"] = is_soft
        q["is_gap_skill"] = _derive_is_gap_skill(q["targets_skill"], gap_skills)
        q.update(_compute_answerability(q["question"], q["targets_skill"],
                                        vocabulary, is_soft, language=language))

    return parsed


# ==========================================
# 6d. System Prompt — Phase 3 (Answer Evaluation)
# ==========================================
# Deliberately "blind": it does not reveal that this system also wrote the
# question. A model asked to judge an answer to *its own* question tends to
# reward answers phrased the way it would have phrased them. Presenting the
# pair neutrally reduces that conformity effect — the same separation
# principle applied to the Answerability score.
ANSWER_EVALUATION_PROMPT = """You are a strict technical interviewer reviewing a
candidate's answer during a job interview.

You are given an interview question, the skill it is meant to probe, and the
candidate's spoken answer (transcribed).

Judge the three criteria below. They are INDEPENDENT of one another — score
each one on its own terms and do not let a low score on one drag down another.

In particular: a SHORT answer is not automatically a WRONG answer. Brevity is
measured by "depth" alone. If a candidate states something true in one
sentence, "technical_accuracy" is high even though "depth" is low.

For EACH criterion, first state briefly what you observed, then assign a score
between 0.0 and 1.0:

1. "technical_accuracy" — Are the claims the candidate made TRUE?
   Judge only correctness, never how much was said.
   1.0 = everything stated is correct (even if only one sentence was stated)
   0.5 = mostly correct with an imprecision
   0.0 = contains a clear technical error, or is factually wrong
   Example: "Kubernetes manages containers and restarts them if they fail" is
   brief but TRUE → technical_accuracy = 1.0 (and depth would be low).

2. "relevance" — Does the answer address the question that was asked?
   Judge only topic match, never completeness.
   1.0 = it is about what was asked   0.5 = partly, or drifts to a near topic
   0.0 = it is about something else entirely
   Example: a brief answer that engages the right topic is still relevant.

3. "depth" — How far below the surface does it go?
   THIS is where brevity and superficiality are penalised.
   1.0 = explains mechanisms, trade-offs, failure modes, or real experience
   0.5 = correct but textbook-level, no mechanism explained
   0.0 = a bare assertion with nothing behind it

Do not reward confidence or polished phrasing on their own: a fluent answer
that is factually wrong must still score 0.0 on technical_accuracy.

Also write "feedback": two sentences maximum, addressed to the candidate,
naming one concrete thing to improve. Be specific — not "add more detail" but
what detail was missing.

Return only JSON — no preamble, no explanation, no markdown code fences.

Output format:

{{
  "observations": "string (brief, what you noticed)",
  "technical_accuracy": 0.0,
  "relevance": 0.0,
  "depth": 0.0,
  "feedback": "string"
}}
"""


# A behavioural answer cannot be graded on factual correctness — there is no
# right answer to "describe a time you missed a deadline". It is graded on
# whether it supplies EVIDENCE, and STAR is the standard instrument for that.
# So soft-skill answers get their own rubric rather than being forced through a
# technical one, which would otherwise score every honest personal story 0.0 on
# technical_accuracy.
SOFT_ANSWER_EVALUATION_PROMPT = """You are an experienced interviewer reviewing a
candidate's answer to a BEHAVIOURAL interview question.

Behavioural answers are assessed with the STAR framework: a usable answer
describes a specific Situation, the candidate's own Task or role in it, the
Actions they personally took, and the Result.

There is NO factually correct answer here. Do NOT judge technical correctness.
Judge only whether the candidate supplied evidence of the behaviour.

For each STAR component report:
  1.0 = supplied concretely and specifically
  0.5 = alluded to, but vague or generic
  0.0 = absent

Grade the four independently:

1. "situation" — is there ONE specific, real past episode? A general habit or
   a hypothetical is NOT a situation.
   "Last term our team's deployment broke two days before the demo" = 1.0
   "I always make sure to communicate clearly with my team" = 0.0 (a claim
   about themselves, not an episode)

2. "task" — is the candidate's OWN role or responsibility in that episode
   clear? What were they specifically accountable for?

3. "action" — what did THEY personally do? Look for first-person, concrete
   steps. Be careful with answers written entirely in "we": if the candidate's
   own contribution never becomes visible, this is at most 0.5.

4. "result" — what was the outcome? Any stated consequence counts, and a
   negative or mixed outcome counts fully — a quantified figure is NOT
   required.

Then two further criteria, judged independently of the four above:

"relevance" — does the episode actually demonstrate the skill being probed, or
is it a story about something else? 1.0 = it demonstrates it directly.

"depth" — is there reflection: trade-offs weighed, what they learned, what
they would do differently? 1.0 = genuine reflection. 0.0 = a flat retelling.

Also write "feedback": two sentences maximum, addressed to the candidate,
naming the single weakest STAR component and exactly what was missing from it
— not "give more detail" but which detail.

Return only JSON — no preamble, no explanation, no markdown code fences.

Output format:

{{
  "observations": "string (brief, what you noticed)",
  "situation": 0.0,
  "task": 0.0,
  "action": 0.0,
  "result": 0.0,
  "relevance": 0.0,
  "depth": 0.0,
  "feedback": "string"
}}
"""

# ==========================================
# 6e. Answer evaluation — hybrid scoring
# ==========================================
# Same methodology as the Answerability Score: what can be counted is counted
# in code, and only genuine linguistic/knowledge judgement is left to the
# model. Asking a model for a single 0-100 verdict reproduces exactly the
# self-rating failure documented in Phase 2 — scores cluster high and stop
# discriminating.
_ANSWER_WEIGHTS = {
    "technical_accuracy": 0.35,   # model-judged
    "relevance": 0.25,            # model-judged
    "depth": 0.20,                # model-judged
    "technical_density": 0.10,    # counted
    "substance": 0.10,            # counted
}

# Behavioural answers are scored on a different basis. technical_accuracy is
# dropped (there is nothing to be factually right about) and technical_density
# is dropped (naming technologies is not what a teamwork answer is for); their
# combined weight moves to STAR completeness, which is the actual evidence.
_SOFT_ANSWER_WEIGHTS = {
    "star_completeness": 0.45,    # computed here from the four model-judged parts
    "relevance": 0.25,            # model-judged
    "depth": 0.20,                # model-judged
    "substance": 0.10,            # counted
}

# Situation and Action carry the evidential weight: without a real episode and
# without the candidate's own contribution, there is nothing to assess. Task
# and Result complete the picture but a strong answer can state them briefly.
_STAR_COMPONENT_WEIGHTS = {
    "situation": 0.30,
    "task": 0.20,
    "action": 0.30,
    "result": 0.20,
}

_FILLER_MAX_PENALTY = 0.10        # subtracted, never added

# Only unambiguous hesitation markers. "like" and "actually" are deliberately
# excluded: "this works like a load balancer" is correct English, and counting
# it as hesitation would penalise a candidate for speaking properly.
_FILLER_PATTERNS = (
    "um", "umm", "uh", "uhh", "erm", "hmm", "mmm",
    "you know", "i mean", "sort of", "kind of",
)

# Routing thresholds for the adaptive interview.
ANSWER_WEAK_THRESHOLD = 0.4       # below → move on to a different skill
ANSWER_STRONG_THRESHOLD = 0.8     # above → probe deeper on the same skill

_SUBSTANCE_BANDS = ((10, 0.3), (30, 0.7))   # (< words, score); above → 1.0


def _score_substance(answer_text: str) -> float:
    """Is there actually an answer here? Word count catches the degenerate
    cases ("222", "jgjjjjjgjjg") deterministically and for free."""
    words = len(answer_text.split())
    if words == 0:
        return 0.0
    for limit, score in _SUBSTANCE_BANDS:
        if words < limit:
            return score
    return 1.0


def _count_filler_words(answer_text: str, language: str = "en") -> int:
    text = answer_text.lower()
    total = 0
    for filler in _lex(language, "fillers"):
        if " " in filler:
            total += text.count(filler)          # multi-word marker
        else:
            total += len(_word_boundary_search_all(filler, text))
    return total


def _word_boundary_search_all(needle: str, haystack_lower: str) -> list:
    """Every standalone occurrence, using the same boundary rules as
    _word_boundary_search (which returns only the first)."""
    escaped = re.escape(needle)
    if re.search(f"[{_ARABIC_CHAR}]", needle):
        pattern = (f"(?<![{_ARABIC_CHAR}])[{_AR_PREFIXES}]?(?:ال)?"
                   + escaped + f"(?![{_ARABIC_CHAR}])")
    else:
        pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
    return re.findall(pattern, haystack_lower)


def _score_filler_penalty(answer_text: str, language: str = "en") -> tuple[int, float]:
    """Returns (filler_count, penalty). The penalty scales with the *ratio* of
    fillers to words, not the raw count — three hesitations in a long answer
    is normal speech, three in a one-line answer is not."""
    words = len(answer_text.split())
    count = _count_filler_words(answer_text, language)
    if words == 0 or count == 0:
        return count, 0.0
    ratio = count / words
    # 10% or more of the answer being filler earns the full penalty.
    penalty = min(_FILLER_MAX_PENALTY, (ratio / 0.10) * _FILLER_MAX_PENALTY)
    return count, round(penalty, 3)


def _score_technical_density(answer_text: str, vocabulary: set) -> float:
    """How many known skills/technologies the answer actually names. Reuses the
    same counting used for question scoring, so the two stay consistent."""
    named = _count_named_entities(answer_text, vocabulary)
    if named == 0:
        return 0.0
    if named == 1:
        return 0.5
    return 1.0


def _score_star_completeness(parsed: dict) -> float:
    """Combine the four model-judged STAR components into one figure. The model
    reports what it observed per component; the arithmetic stays in code, for
    the same reason the Answerability Score does — a single holistic 0-1 asked
    of a model clusters high and stops discriminating."""
    return round(sum(_STAR_COMPONENT_WEIGHTS[c] * parsed[c]
                     for c in _STAR_COMPONENT_WEIGHTS), 3)


def _evaluate_soft_answer(question: str, answer: str, targets_skill: str,
                          substance: float, filler_count: int,
                          filler_penalty: float, language: str = "en") -> dict:
    """STAR-based evaluation for a behavioural question."""
    user_prompt = (
        f"Behavioural skill being probed: {targets_skill}\n\n"
        f"Interview question:\n{question}\n\n"
        f"Candidate's answer:\n{answer}"
    )
    parsed = _call_llm_json(
        _with_language(SOFT_ANSWER_EVALUATION_PROMPT, language), user_prompt)
    if "error" in parsed:
        return parsed

    required = ("situation", "task", "action", "result", "relevance", "depth")
    for field in required:
        if not isinstance(parsed.get(field), (int, float)):
            return {"error": f"Soft-skill evaluation returned a non-numeric '{field}'.",
                    "raw_output": parsed}

    star_completeness = _score_star_completeness(parsed)
    score = (
        _SOFT_ANSWER_WEIGHTS["star_completeness"] * star_completeness
        + _SOFT_ANSWER_WEIGHTS["relevance"] * parsed["relevance"]
        + _SOFT_ANSWER_WEIGHTS["depth"] * parsed["depth"]
        + _SOFT_ANSWER_WEIGHTS["substance"] * substance
    ) - filler_penalty

    return {
        "final_score": round(max(0.0, min(1.0, score)), 3),
        "is_soft_skill": True,
        "substance": substance,
        "skill_addressed": None,        # not applicable — see evaluate_answer
        "technical_density": None,      # not applicable to a behavioural answer
        "filler_count": filler_count,
        "filler_penalty": filler_penalty,
        "star": {c: parsed[c] for c in _STAR_COMPONENT_WEIGHTS},
        "star_completeness": star_completeness,
        "technical_accuracy": None,     # nothing to be factually right about
        "relevance": parsed["relevance"],
        "depth": parsed["depth"],
        "feedback": parsed.get("feedback", ""),
        "llm_called": True,
    }


def evaluate_answer(question: str, answer: str, targets_skill: str,
                    vocabulary: set, is_soft_skill: bool = False,
                    language: str = "en") -> dict:
    """
    Phase 3: Evaluate a candidate's answer to one interview question.

    Countable properties (substance, whether the skill was addressed, technical
    density, filler words) are computed here in code. Only technical accuracy,
    relevance and depth are judged by the model, and it is shown the pair
    blind — it is not told the question came from this same system.

    An answer with no substance, or one that never engages the skill at all,
    is scored zero WITHOUT calling the model: it is deterministically
    unanswerable, the check is free, and it cannot be flattered.

    Returns:
        {"final_score", "substance", "skill_addressed", "technical_density",
         "filler_count", "filler_penalty", "technical_accuracy", "relevance",
         "depth", "feedback", "llm_called"}
        or {"error": ...} on failure.
    """
    answer = (answer or "").strip()

    substance = _score_substance(answer)
    filler_count, filler_penalty = _score_filler_penalty(answer, language)

    # Reported, but no longer decisive.
    #
    # This used to be a kill switch: an answer that did not literally contain
    # its target skill scored 0 without the evaluator ever being called. That
    # is a bad proxy for "engaged with the topic", and it failed on real
    # interviews — four answers of 126 to 476 words were zeroed, including a
    # detailed account of prioritising a mid-sprint request that simply never
    # used the word "Agile", and a React explanation that said "React" where
    # the skill label read "React.js".
    #
    # Whether an answer is on topic is a semantic judgement, and the rubric
    # already has the right instrument for it: `relevance`, judged by the model.
    # An off-topic answer now scores low through that path (relevance near 0
    # drags the weighted total down) instead of being silently zeroed by a
    # string comparison. The cost is one model call on answers that would
    # previously have been rejected for free — correctness is worth more than
    # that call.
    skill_addressed = (None if is_soft_skill
                       else (_target_is_named(answer, targets_skill, language)
                             if answer else False))

    # --- deterministic short-circuit -------------------------------------
    # Only the genuinely degenerate case: no answer at all. "222" or an empty
    # box needs no model to judge it, and cannot be flattered by one.
    if substance == 0.0:
        return {
            "final_score": 0.0,
            "is_soft_skill": is_soft_skill,
            "substance": substance,
            "skill_addressed": skill_addressed,
            "technical_density": 0.0,
            "filler_count": filler_count,
            "filler_penalty": 0.0,
            "technical_accuracy": None,
            "relevance": None,
            "depth": None,
            "feedback": ("No substantive answer was given for this skill — the "
                         "answer does not engage with the topic asked about."),
            "llm_called": False,
        }

    # --- behavioural questions take the STAR rubric ------------------------
    if is_soft_skill:
        return _evaluate_soft_answer(question, answer, targets_skill,
                                     substance, filler_count, filler_penalty,
                                     language)

    technical_density = _score_technical_density(answer, vocabulary)

    user_prompt = (
        f"Skill being probed: {targets_skill}\n\n"
        f"Interview question:\n{question}\n\n"
        f"Candidate's answer:\n{answer}"
    )
    parsed = _call_llm_json(
        _with_language(ANSWER_EVALUATION_PROMPT, language), user_prompt)
    if "error" in parsed:
        return parsed

    for field in ("technical_accuracy", "relevance", "depth"):
        if not isinstance(parsed.get(field), (int, float)):
            return {"error": f"Answer evaluation returned a non-numeric '{field}'.",
                    "raw_output": parsed}

    # An answer that scores zero on BOTH relevance and depth did not engage
    # with the skill at all — typically it answered a different question well.
    # Three of the five criteria (accuracy, density, substance) never ask what
    # the answer is *about*, so a fluent off-topic reply still collects 0.55 of
    # the weight: measured, an expert PostgreSQL answer to a Kubernetes question
    # scored 0.45 and would have been filed as "adequate at Kubernetes" for a
    # candidate who never mentioned it.
    #
    # The paradox is that the stronger the candidate, the worse the error: a
    # weak off-topic answer stays under the threshold on its own, while a good
    # one is promoted. So the score is zeroed and flagged — and the flag is the
    # point. "Not tested" is a different fact from "does not know", exactly as
    # a NULL final score differs from a stored 0, and the report must not turn
    # the first into the second.
    skill_untested = parsed["relevance"] == 0.0 and parsed["depth"] == 0.0
    if skill_untested:
        score = 0.0
    else:
        score = (
            _ANSWER_WEIGHTS["technical_accuracy"] * parsed["technical_accuracy"]
            + _ANSWER_WEIGHTS["relevance"] * parsed["relevance"]
            + _ANSWER_WEIGHTS["depth"] * parsed["depth"]
            + _ANSWER_WEIGHTS["technical_density"] * technical_density
            + _ANSWER_WEIGHTS["substance"] * substance
        ) - filler_penalty

    return {
        "final_score": round(max(0.0, min(1.0, score)), 3),
        "is_soft_skill": False,
        "substance": substance,
        "skill_addressed": True,
        "skill_untested": skill_untested,
        "technical_density": technical_density,
        "filler_count": filler_count,
        "filler_penalty": filler_penalty,
        "technical_accuracy": parsed["technical_accuracy"],
        "relevance": parsed["relevance"],
        "depth": parsed["depth"],
        "feedback": (
            "This answer did not address the skill being asked about, so it "
            "carries no evidence either way — the skill was not tested."
            if skill_untested else parsed.get("feedback", "")),
        "llm_called": True,
    }


# ==========================================
# 6f. Adaptive interview — dynamically generated questions
# ==========================================
# Questions produced mid-interview must clear the SAME Answerability gate as
# the pre-planned ones. Without this the system would have two tiers: planned
# questions quality-checked, improvised ones not — which would undermine the
# gate's meaning entirely.

FOLLOWUP_QUESTION_PROMPT = """You are a technical interviewer. The candidate just
gave a strong answer about a skill, and you want to find the limit of what they
actually know.

Write ONE follow-up question that goes deeper.

Rules:

1. Build on what they actually said — quote or reference a specific term from
   their answer, so the question could not have been written in advance.
2. Ask about something they did NOT cover: a failure mode, an edge case, a
   trade-off, or the mechanism behind what they described.
3. Do not re-ask what they already answered.
4. The question MUST name the skill being probed in its own text.
5. Phrase it with a clear interrogative ("What/How/Why/Which") or a directive
   verb ("Describe/Explain/Compare"), ask for something concrete, and avoid
   vague openers like "Tell me about".
6. Return only JSON — no preamble, no markdown code fences.

Output format:

{{
  "question": "string (must name the skill, and reference their answer)",
  "targets_skill": "{skill}"
}}
"""

OFF_PLAN_QUESTION_PROMPT = """You are a technical interviewer. While answering a
different question, the candidate mentioned a skill that is relevant to this
role but was not part of the planned question set. You want to follow that
thread, as a real interviewer would.

Write ONE question about that skill.

Rules:

1. The question MUST name the skill "{skill}" in its own text.
2. Ask about a concrete task, scenario, or comparison — not a definition.
3. Phrase it with a clear interrogative ("What/How/Why/Which") or a directive
   verb ("Describe/Explain/Compare"), and avoid vague openers like
   "Tell me about".
4. Keep it self-contained and answerable in a spoken interview.
5. Return only JSON — no preamble, no markdown code fences.

Output format:

{{
  "question": "string (must name {skill})",
  "targets_skill": "{skill}"
}}
"""


# The technical follow-up prompt above is wrong for a behavioural answer: it
# asks for failure modes, edge cases and mechanisms, which produce nonsense
# when the subject is a story about a disagreement in a team. This one instead
# takes the STAR component scores that evaluation already produced and asks
# for whatever the candidate left out — which is exactly the move a real
# interviewer makes ("...and how did that turn out?").
SOFT_FOLLOWUP_QUESTION_PROMPT = """You are an interviewer. The candidate has just
described a real situation from their past, but their account is incomplete.

Here is how complete each part of their STAR answer was
(1.0 = given concretely, 0.5 = vague, 0.0 = absent):

  Situation = {situation}
  Task      = {task}
  Action    = {action}
  Result    = {result}

Write ONE follow-up question that fills in the WEAKEST part.

Rules:

1. Reference something specific the candidate actually said — a name, a
   decision, a deadline — so the question could not have been written before
   hearing their answer.
2. Ask for the missing part directly, and use its wording:
   - Situation weakest → ask for one concrete occasion, using the words
     "a specific situation where" or "a time when".
   - Task weakest    → ask what THEIR OWN role or responsibility was, using
     the words "your role" or "your responsibility".
   - Action weakest  → ask what they personally did, using the words
     "what did you do". This is the right choice when they told the story
     entirely in "we" and their own contribution never became visible.
   - Result weakest  → ask about the outcome, using the words
     "how did it turn out" or "what happened".
3. Keep those exact cue phrases. They are what makes the question answerable
   rather than an invitation to generalise.
4. Do NOT ask about technical mechanisms, failure modes, edge cases or
   trade-offs. This is a behavioural question, not a technical one.
5. One question, answerable out loud, ending in "?".
6. Return only JSON — no preamble, no markdown code fences.

Output format:

{{
  "question": "string",
  "targets_skill": "{skill}"
}}
"""

# Off-plan behavioural questions open a NEW episode rather than continuing one,
# so unlike the follow-up above they must still establish the Situation
# themselves — the same requirement as a planned behavioural question.
SOFT_OFF_PLAN_QUESTION_PROMPT = """You are an interviewer. While answering a
different question, the candidate revealed something about the behavioural
skill "{skill}", which was not part of the planned question set. You want to
follow that thread, as a real interviewer would.

Write ONE behavioural question about "{skill}", using the STAR framework.

Rules:

1. Anchor it in ONE specific past episode. Use the words "a specific situation
   where" or "a time when" — never "what would you do if" and never "how do
   you usually".
2. In the same question, also ask for their own role, what they personally
   did, and how it turned out.
3. Reference what they just said, so the question is clearly a response to
   them rather than a generic prompt.
4. One question, answerable out loud, ending in "?".
5. Return only JSON — no preamble, no markdown code fences.

Output format:

{{
  "question": "string",
  "targets_skill": "{skill}"
}}
"""

MAX_PROBE_DEPTH = 2            # follow-ups allowed per skill
MAX_OFF_PLAN_QUESTIONS = 2     # off-plan pivots allowed per interview
MAX_TOTAL_QUESTIONS = 12       # hard ceiling on interview length
_MAX_QUESTION_GEN_RETRIES = 2  # attempts to clear the gate before giving up


def _generate_gated_question(system_prompt: str, user_prompt: str, skill: str,
                             vocabulary: set, is_soft_skill: bool,
                             situation_established: bool = False,
                             language: str = "en") -> dict:
    """Generate one question and hold it to the same Answerability gate used
    for the planned set. Retries once on failure; the caller is expected to
    fall back to the planned queue if this still returns None, so a weak
    question is never shown to a candidate."""
    system_prompt = _with_language(system_prompt, language)
    for _ in range(_MAX_QUESTION_GEN_RETRIES):
        parsed = _call_llm_json(system_prompt, user_prompt)
        if "error" in parsed:
            continue
        question_text = parsed.get("question")
        if not isinstance(question_text, str) or not question_text.strip():
            continue

        parsed["targets_skill"] = parsed.get("targets_skill") or skill
        parsed["is_soft_skill"] = is_soft_skill
        parsed.update(_compute_answerability(
            question_text, parsed["targets_skill"], vocabulary, is_soft_skill,
            situation_established, language))
        parsed["is_gap_skill"] = False   # set by the caller, which knows the gap
        if parsed["passes_gate"]:
            return parsed
    return None


def generate_followup(original_question: str, answer: str, targets_skill: str,
                      vocabulary: set, is_soft_skill: bool = False,
                      star: dict | None = None, language: str = "en") -> dict:
    """Generate a deeper follow-up grounded in what the candidate actually
    said. Returns None if no generated question clears the gate.

    A behavioural follow-up takes a different route entirely: it is told which
    STAR components the answer was missing and asks for the weakest one, and
    it is scored with `situation_established=True` because the episode was
    already established by the answer being followed up on.
    """
    user_prompt = (
        f"Skill being probed: {targets_skill}\n\n"
        f"Question already asked:\n{original_question}\n\n"
        f"Candidate's answer:\n{answer}"
    )

    if is_soft_skill:
        star = star or {}
        system_prompt = SOFT_FOLLOWUP_QUESTION_PROMPT.format(
            skill=targets_skill,
            situation=star.get("situation", 0.0),
            task=star.get("task", 0.0),
            action=star.get("action", 0.0),
            result=star.get("result", 0.0),
        )
        return _generate_gated_question(system_prompt, user_prompt, targets_skill,
                                        vocabulary, True,
                                        situation_established=True,
                                        language=language)

    return _generate_gated_question(
        FOLLOWUP_QUESTION_PROMPT.format(skill=targets_skill),
        user_prompt, targets_skill, vocabulary, False, language=language)


def generate_off_plan_question(skill_name: str, answer_context: str,
                               vocabulary: set, is_soft_skill: bool = False,
                               language: str = "en") -> dict:
    """Generate a question about a skill the candidate raised themselves,
    outside the planned set. Returns None if it can't clear the gate.

    Unlike a follow-up, this opens a NEW episode, so a behavioural one must
    still establish its own Situation — the same requirement as a planned
    behavioural question.
    """
    user_prompt = (
        f"Skill to ask about: {skill_name}\n\n"
        f"The candidate mentioned it while saying:\n{answer_context}"
    )
    prompt = (SOFT_OFF_PLAN_QUESTION_PROMPT if is_soft_skill
              else OFF_PLAN_QUESTION_PROMPT)
    return _generate_gated_question(
        prompt.format(skill=skill_name),
        user_prompt, skill_name, vocabulary, is_soft_skill, language=language)


# ==========================================
# 6g. System Prompt — the adaptive interview agent (ReAct)
# ==========================================
AGENT_SYSTEM_PROMPT = """You are a technical interviewer running an adaptive
interview. After each answer you receive a numeric evaluation, and you decide
what to do next — exactly as an experienced interviewer would.

Reason step by step before acting:
  1. What do the evaluation components say? Accuracy, relevance and depth are
     SEPARATE signals — read them individually, not just the final score.
  2. Do you need more information about the interview state before deciding?
  3. Which tool serves the goal?

How to read the components:

• LOW accuracy + LOW depth    → they do not know this topic → move on.
• HIGH accuracy + LOW depth   → they know it but did not explain → probe deeper.
• HIGH accuracy + HIGH depth  → topic is covered → continue to the next skill.
• LOW accuracy + HIGH depth   → they talked at length but were WRONG. This is
  worse than a short correct answer, not better. Move on.

Also: if the candidate mentioned a named skill that is relevant to the role but
was not the subject of the question, you may follow that thread with
ask_about_skill — a real interviewer notices such openings.

Constraints:
- Do not move on if no other skills remain — check the state first.
- Do not probe the same skill more than twice.
- Always give a concrete reason, citing the numbers you based it on.

Call exactly one decision tool (move_to_another_skill, probe_deeper,
ask_about_skill, or continue_as_planned) once you have decided. You may call
get_interview_state first if you need it.
"""


# ==========================================
# 7. Standalone test harness
# ==========================================
if __name__ == "__main__":
    # A synthetic CV, not a real one. It deliberately mixes the cases the
    # extractor has to separate: named tools, a skill buried in a project
    # sentence rather than a skills list, job duties described in prose (which
    # must NOT be extracted), soft skills, and languages with proficiency
    # qualifiers attached.
    sample_cv_text = """
    Final-year Electrical and Electronic Engineering student, Automatic Control track.

    Projects: built a data-logging system on an ESP32 with an MPU6050 sensor,
    storing readings in MySQL and displaying live values on an OLED module.
    Simulated a differential-drive robot in ROS 2 (Humble) using Turtlesim.

    Coursework: Python, MATLAB, C, Arduino.

    Experience: documented laboratory procedures, prepared weekly progress
    reports, and coordinated with the supervising engineer on scheduling.

    Personal skills: strong communication skills, teamwork, time management.

    Languages: Arabic (Native), English (Intermediate).
    """

    print("Running Phase 1 skill extraction test...\n")
    result = extract_skills(sample_cv_text, document_type="CV")
    print(json.dumps(result, indent=2, ensure_ascii=False))
