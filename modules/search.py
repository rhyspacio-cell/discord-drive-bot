"""
Ollama-assisted search planning for Google Drive retrieval.

This module converts a user's natural-language Drive question into a
structured search plan.

The search plan separates:

    1. required_terms
       Terms that strongly identify the requested documents.

    2. phrases
       Multi-word concepts that identify the requested documents.

    3. optional_terms
       Additional retrieval terms that may help but are not essential.

    4. context_terms
       Terms describing WHAT information the user wants from the matching
       documents. These are intentionally NOT used to broaden Drive
       candidate retrieval.

    5. exclude_terms
       Terms representing explicit exclusions from the user's request.

The local Ollama model performs the initial interpretation.

A deterministic cleanup layer then validates and corrects the model output.
This is important because the model may occasionally place generic words
such as "documents", "where", or "working" into retrieval categories.

The deterministic layer is intentionally conservative:

    - It does not invent search terms.
    - It does not infer facts about the user.
    - It does not broaden the user's query.
    - It only removes or relocates terms that are known to be unsuitable
      for Drive candidate retrieval.
    - It keeps the final search plan small and predictable.

The resulting plan is consumed by the Drive search layer.
"""


# ============================================================================
# IMPORTS
# ============================================================================

import json
import re
from datetime import date, datetime, timedelta

from modules.llm import LocalLLMError, generate_local_response
from modules.query_constraints import extract_query_constraints


# ============================================================================
# SEARCH PLAN CONSTANTS
# ============================================================================

# These are the five categories that contain search-related values.
#
# The order matters during the final duplicate-removal pass:
# the first category containing a value keeps it.
SEARCH_PLAN_KEYS = (
    "required_terms",
    "phrases",
    "optional_terms",
    "context_terms",
    "exclude_terms",
)


# Generic words that should not normally participate in Drive candidate
# retrieval.
#
# These words describe the user's request rather than identifying the
# document that contains the answer.
#
# IMPORTANT:
# These values are also removed from INSIDE phrases.
#
# Example:
#
#     "research documents"
#
# becomes:
#
#     "research"
#
# This prevents the phrase from eventually causing drive.py to search for
# both "research" and the generic word "documents".
GENERIC_RETRIEVAL_TERMS = {
    "file",
    "files",
    "document",
    "documents",
    "information",
    "content",
    "question",
    "answer",
    "paragraph",
}


# Generic question/formatting words.
#
# These are removed from every search-plan category because they do not
# identify a useful Drive candidate.
GENERIC_QUESTION_TERMS = {
    "where",
    "who",
    "what",
    "when",
    "which",
    "why",
    "how",
    "paragraph",
    "question",
    "answer",
}

GENERIC_CONTROL_TERMS = {
    "search",
    "search for",
    "find",
    "show",
    "show me",
    "look",
    "look for",
    "from",
    "to",
    "present",
    "until",
    "since",
    "between",
    "through",
    "that",
    "entered",
    "date",
    "dates",
    "list",
    "lists",
}


# Employment-related terms describe the information being requested rather
# than normally identifying the user's documents.
#
# Example:
#
#     "Where does Rhys Pacio currently work?"
#
# Retrieval identity:
#
#     "rhys pacio"
#
# Context:
#
#     "currently"
#     "working"
#
# We deliberately do NOT use "working" as a Drive candidate search term
# because many unrelated documents may contain that word.
EMPLOYMENT_CONTEXT_TERMS = {
    "currently",
    "currently working",
    "working",
    "working now",
    "works",
    "works now",
    "current employer",
    "employer",
    "employment",
    "job",
    "where they work",
    "where is the person working",
}


# ============================================================================
# SEARCH PLAN JSON SCHEMA
# ============================================================================

# Ollama can use this schema for structured JSON output.
#
# Keeping the schema explicit makes the model output easier to validate and
# prevents the search planner from returning arbitrary JSON structures.
SEARCH_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
        },
        "required_terms": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "phrases": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "optional_terms": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "context_terms": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "exclude_terms": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "time_range": {
            "type": "object",
            "properties": {
                "from": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
                "to": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
            },
            "additionalProperties": False,
        },
        "answer_type": {
            "type": "string",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    "required": [
        "intent",
        "required_terms",
        "phrases",
        "optional_terms",
        "context_terms",
        "exclude_terms",
        "time_range",
        "answer_type",
        "confidence",
    ],
}


# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================

def _normalize_value(value: str) -> str:
    """
    Normalize one search value returned by the local model.

    Normalization rules:

        - Remove leading/trailing whitespace.
        - Convert to lowercase.
        - Replace underscores with spaces.
        - Collapse repeated whitespace.

    Example:

        "  Computer_Science  "
        ->
        "computer science"
    """

    value = value.strip().lower()

    # Drive search should use normal spaces rather than underscores.
    value = value.replace("_", " ")

    # Collapse repeated whitespace.
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def _clean_values(plan, key):
    """
    Validate and normalize one search-plan category.

    The model is required to return a list of strings for every search
    category.

    Invalid values are rejected rather than silently converted because
    malformed model output should not become a Drive query.
    """

    values = plan.get(key, [])

    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise LocalLLMError(
            "The local AI model returned an invalid search plan."
        )

    cleaned = []
    seen = set()

    for value in values:
        value = _normalize_value(value)

        if not value:
            continue

        # Avoid excessively large search values.
        if len(value) > 100:
            continue

        # Remove duplicates within the same category.
        if value in seen:
            continue

        seen.add(value)
        cleaned.append(value)

    return cleaned


def _remove_generic_words_from_phrase(phrase: str) -> str:
    """
    Remove generic retrieval words from inside a phrase.

    This is an important deterministic correction.

    Example:

        "research documents"
        ->
        "research"

    Another example:

        "academic transcript document"
        ->
        "academic transcript"

    Only exact whitespace-separated words are removed.

    This function does NOT remove words such as "working" because employment
    handling is performed separately.
    """

    words = phrase.split()

    filtered_words = [
        word
        for word in words
        if word not in GENERIC_RETRIEVAL_TERMS
        and word not in GENERIC_CONTROL_TERMS
    ]

    return " ".join(filtered_words).strip()


def _extract_explicit_search_anchor(question: str):
    """Extract an explicit SEARCH: anchor when the user provides one."""
    if not isinstance(question, str):
        return None

    match = re.search(
        r"\bSEARCH\s*:\s*(.+?)(?=\s+\b(?:from|to|until|since|between|which|what|who|when|why|how|where)\b|$)",
        question,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    anchor = re.sub(r"\s+", " ", match.group(1)).strip()
    return anchor or None


def _strip_generic_control_terms(value: str) -> str:
    """Remove user-request and date-structure words that should not affect relevance."""
    if not isinstance(value, str):
        return value

    cleaned = value.strip().lower()

    for phrase in sorted(
        GENERIC_CONTROL_TERMS,
        key=len,
        reverse=True,
    ):
        if not phrase:
            continue
        cleaned = re.sub(
            rf"\b{re.escape(phrase)}\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

    words = re.findall(
        r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",
        cleaned,
    )

    return " ".join(words).strip()


def _parse_iso_date(value):
    """Normalize a date-like value to YYYY-MM-DD, or None for an open end."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError("Time range values must be ISO dates or null.")

    value = value.strip()

    if not value:
        return None

    if value.lower() == "present":
        return None

    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        pass

    for fmt in (
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %Y",
        "%b %Y",
    ):
        try:
            parsed = datetime.strptime(value, fmt)
            if fmt in {"%B %Y", "%b %Y"}:
                return date(parsed.year, parsed.month, 1).isoformat()
            return parsed.date().isoformat()
        except ValueError:
            continue

    month_names = (
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    )

    text = value.lower()

    for month_name in month_names:
        if text.startswith(month_name):
            if re.search(r"\b\d{4}\b", text):
                match = re.search(r"\b(\d{4})\b", text)
                year = int(match.group(1))
                month_index = month_names.index(month_name) + 1
                return date(year, month_index, 1).isoformat()

    raise ValueError(f"Unsupported date value: {value!r}")


def _normalize_time_range(time_range):
    """Validate and normalize a time_range object from the planner or question."""
    if time_range is None:
        return {"from": None, "to": None}

    if not isinstance(time_range, dict):
        raise LocalLLMError(
            "The local AI model returned an invalid time range."
        )

    raw_from = time_range.get("from")
    raw_to = time_range.get("to")

    if raw_from is not None and not isinstance(raw_from, str):
        raise LocalLLMError(
            "The local AI model returned an invalid time range."
        )

    if raw_to is not None and not isinstance(raw_to, str):
        raise LocalLLMError(
            "The local AI model returned an invalid time range."
        )

    from_value = None
    to_value = None

    if raw_from is not None and raw_from.strip():
        from_value = _parse_iso_date(raw_from)

    if raw_to is not None and raw_to.strip() and raw_to.lower() != "present":
        to_value = _parse_iso_date(raw_to)

    if from_value and to_value and date.fromisoformat(from_value) > date.fromisoformat(to_value):
        raise LocalLLMError(
            "The local AI model returned an invalid time range."
        )

    return {
        "from": from_value,
        "to": to_value,
    }


def _normalize_time_range_for_query(time_range):
    """Return a query-safe time range with month-end semantics for inclusive end dates."""
    normalized = _normalize_time_range(time_range)

    start = normalized.get("from")
    end = normalized.get("to")

    if start is not None and end is not None and date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("Start date cannot be after the end date.")

    return normalized


def _extract_time_range_from_question(question: str):
    """Interpret a small set of explicit date range expressions into a normalized range."""
    if not isinstance(question, str):
        return None

    if not re.search(r"\b(?:from|between|to|until|present)\b", question, flags=re.IGNORECASE):
        return None

    months = "(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    date_text = rf"(?:{months}\s+(?:\d{{1,2}}(?:\s*(?:,|\s+)\s*\d{{4}}|\s+\d{{4}})|\d{{4}}))"

    patterns = [
        rf"\b(?:from|since)\s+(?P<start>{date_text})(?:\s*(?:to|until|through|-|and)\s*(?P<end>present|{date_text}))?\b",
        rf"\bbetween\s+(?P<start>{date_text})\s*(?:and|to)\s*(?P<end>present|{date_text})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if not match:
            continue

        start_raw = match.group("start").strip()
        end_raw = match.group("end")

        if end_raw is None:
            end_value = None
        else:
            end_raw = end_raw.strip()
            if end_raw.lower() == "present":
                end_value = None
            else:
                end_value = _parse_iso_date(end_raw)

        start_value = _parse_iso_date(start_raw)

        if end_value is not None and date.fromisoformat(start_value) > date.fromisoformat(end_value):
            raise LocalLLMError("The local AI model returned an invalid time range.")

        return {
            "from": start_value,
            "to": end_value,
        }

    return None


def _extract_explicit_date_constraints(question: str):
    """Return explicit document and activity dates mentioned in the question."""
    if not isinstance(question, str):
        return {
            "document_dates": [],
            "activity_dates": [],
        }

    text = question.strip()
    if not text:
        return {
            "document_dates": [],
            "activity_dates": [],
        }

    months = "(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    date_text = rf"(?:{months}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*,?\s*\d{{4}})?|\d{{4}}-\d{{1,2}}-\d{{1,2}})"

    document_dates = []
    activity_dates = []

    for match in re.finditer(date_text, text, flags=re.IGNORECASE):
        raw_date = match.group(0).strip()
        try:
            normalized = _parse_iso_date(raw_date)
        except ValueError:
            continue

        if normalized is None:
            continue

        lower = text.lower()
        date_context = lower[max(0, match.start() - 30):match.end() + 60]
        if re.search(r"\b(?:submitted|document|letter|dated|filed)\b", date_context, flags=re.IGNORECASE):
            document_dates.append(normalized)
        elif re.search(r"\b(?:planned\s+activity|sampling|scheduled|activity|site\s+visit|conducting)\b", date_context, flags=re.IGNORECASE):
            activity_dates.append(normalized)
        elif re.search(r"\b(?:which\s+company|company\s+is\s+associated|for\s+each\s+company|list\s+all\s+applicable|all\s+applicable|what\s+planned\s+activity|what\s+activity)\b", lower, flags=re.IGNORECASE):
            activity_dates.append(normalized)

    def dedupe(values):
        seen = set()
        out = []
        for value in values:
            if value not in seen:
                seen.add(value)
                out.append(value)
        return out

    return {
        "document_dates": dedupe(document_dates),
        "activity_dates": dedupe(activity_dates),
    }


def _looks_like_date_value(value: str) -> bool:
    """Detect explicit date strings that should not leak into retrieval term lists."""
    if not isinstance(value, str):
        return False

    text = value.strip().lower()
    if not text:
        return False

    if text == "present":
        return True

    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", text):
        return True

    if re.search(r"\d{4}", text):
        return True

    if re.search(r"\d{1,2}\s*(?:,|\s+)\s*\d{4}", text):
        return True

    return False


def _clean_generic_retrieval_terms(validated):
    """
    Remove generic retrieval words from retrieval-oriented categories.

    Retrieval-oriented categories are:

        - required_terms
        - phrases
        - optional_terms

    Context terms are intentionally excluded because they describe the
    information the user wants rather than the Drive candidate query.

    The cleanup handles both:

        "documents"

    and:

        "research documents"

    so generic words cannot leak into the Drive query through a phrase.
    """

    # ------------------------------------------------------------------
    # Clean required_terms and optional_terms.
    # ------------------------------------------------------------------

    for key in (
        "required_terms",
        "optional_terms",
    ):
        validated[key] = [
            value
            for value in validated[key]
            if value not in GENERIC_RETRIEVAL_TERMS
        ]

    # ------------------------------------------------------------------
    # Clean phrases.
    #
    # Generic words are removed individually from the phrase rather than
    # only checking whether the entire phrase equals a generic word.
    # ------------------------------------------------------------------

    cleaned_phrases = []

    for phrase in validated["phrases"]:
        cleaned_phrase = _remove_generic_words_from_phrase(phrase)

        if not cleaned_phrase:
            continue

        if cleaned_phrase in GENERIC_RETRIEVAL_TERMS:
            continue

        cleaned_phrases.append(cleaned_phrase)

    validated["phrases"] = cleaned_phrases

    return validated


# ============================================================================
# MODEL PROMPT
# ============================================================================

def _build_search_planner_prompt(question: str) -> str:
    """
    Build the prompt used by Ollama to interpret the user's Drive question.

    The prompt deliberately distinguishes:

        document identity
            from
        answer context.

    This prevents words such as "working", "degree", "grade", or "where"
    from unnecessarily becoming Drive retrieval terms.
    """

    return f"""
You are a search-query planner for a personal Google Drive.

The user provides ONE question describing the information they want from
their Google Drive.

Your job is to identify the important words and concepts that can help
locate the relevant files.

Return JSON only.

The JSON must contain exactly these keys:

{{
  "intent": "",
  "required_terms": [],
  "phrases": [],
  "optional_terms": [],
  "context_terms": [],
  "exclude_terms": [],
  "time_range": {{"from": null, "to": null}},
  "answer_type": "",
  "confidence": 0.0
}}

MEANING OF EACH CATEGORY:

- required_terms:
  Terms that are genuinely useful for identifying the relevant documents.
  Use as few as possible.

  Required terms must help identify the document itself. Prefer terms
  describing the document type or subject of the requested information.

  For example, for:
  "What degree was awarded in my Computer Science diploma?"

  Good required terms:
  ["diploma"]

  Good phrase:
  ["computer science"]

  Do not make answer concepts such as "degree" or "awarded" required
  terms unless they are themselves likely to identify the document.

- phrases:
  Important multi-word concepts that should stay together conceptually.

  Examples:
  "computer science"
  "bachelor of science"
  "academic transcript"

  Do NOT create phrases from generic words such as:
  "documents"
  "files"
  "information"
  "content"
  "question"
  "answer"

  For example:

  BAD:
  ["research documents"]

  BETTER:
  ["research"]

- optional_terms:
  Additional terms that may help find the right documents but are not
  necessary.

  Optional terms should help identify relevant documents.

  Do NOT put generic question words or answer-context words such as:
  "where"
  "who"
  "what"
  "currently"
  "working"
  "works"
  "working now"

  into optional_terms when they describe the information the user wants.

- context_terms:
  Terms describing the specific information the user wants to retrieve
  from the matching documents.

  Examples:
  "Where does John Smith currently work?" -> ["currently", "working"]
  "What degree did John Smith receive?" -> ["degree", "received"]
  "What was my final grade in thermodynamics?" -> ["final", "grade"]

  Context terms describe the requested information, not the identity
  of the person or document.

  For employment questions, terms such as:
  "currently", "working", "works", "employer", "employment", "job"
  should normally be context_terms when they describe what the user
  wants to know.

  IMPORTANT:

  Phrases such as:
  "working now"
  "currently working"
  "works now"
  "current employer"
  "where they work"
  "where is the person working"

  describe the requested employment information. They should normally
  be context_terms, NOT optional_terms.

  Do not treat "working now" or similar employment phrases as a phrase
  identifying the person.

  Do not use generic formatting or question words such as:
  "paragraph", "question", "answer", "who", "what", "where".

- exclude_terms:
  Terms that should be excluded ONLY when the user's question clearly
  excludes something.

- time_range:
  Optional explicit date interval describing the user-requested Google Drive
  modified-time range.

  When present, it must be separate from required_terms, phrases,
  optional_terms, and context_terms. It must use this exact shape:

  {{"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}}

  or:

  {{"from": "YYYY-MM-DD", "to": null}}

  Use null for an open-ended range such as "to present". Do not put date
  strings into the keyword arrays. Keep the date range explicit and
  deterministic.

  Only create a time_range when the user explicitly asks for a date range,
  such as:
  - "from August 2026 to present"
  - "between August 23, 2026 and September 1, 2026"
  - "from January 2026 to June 2026"

  Do not infer a date filter from ordinary years mentioned as subject matter.
  Do not invent dates. If the date range is unclear or impossible to
  normalize, set time_range to {{"from": null, "to": null}} instead of
  guessing.

IMPORTANT RULES:

1. Use normal words and spaces.

2. NEVER use underscores.

   WRONG:
   "computer_science"
   "academic_transcript"

   CORRECT:
   "computer science"
   "academic transcript"

3. Do not use Google Drive query syntax.

4. Do not generate operators such as:
   "and", "or", "contains", "not", "="

5. Do not generate URLs, code, commands, credentials, secrets, tokens,
   or instructions.

6. Do not invent specific facts that are not present in the user's question.

7. Do not guess a person's school, employer, degree, company, location,
   dates, names, or other specific information.

8. Do not repeat the same value across different categories.

9. A concept should have ONE purpose.

10. Do not put a phrase into both "phrases" and "optional_terms".

11. Do not put the same word into both "required_terms" and
    "context_terms" unless there is a very strong reason.

12. Prefer fewer meaningful search terms over many weak terms.

13. Do not include generic words such as:
    "file"
    "files"
    "document"
    "documents"
    "information"
    "content"
    "question"
    "answer"
    "paragraph"

14. Generic retrieval words must NOT be hidden inside phrases.

    BAD:
    ["research documents"]

    GOOD:
    ["research"]

15. The total number of values across all five search-term lists must be
    12 or fewer.

16. Keep search values lowercase.

17. confidence must be a number between 0.0 and 1.0.

18. The search plan must be based only on the user's question.

19. The question's subject is more important than generic question words.

20. When the user asks what a person currently does, where they work,
    or who employs them, treat employment-related words as context_terms,
    not optional_terms, unless the word is specifically needed to identify
    the relevant document.

21. When an employment question contains "working now", "works now",
    "currently working", "current employer", "where they work", or similar
    wording, put those concepts in context_terms.

22. Do NOT put employment-context words in optional_terms unless there is
    an unusual and specific reason that the phrase itself identifies a
    document.

23. For a question about a person's current employment, the person's name
    should normally be the primary phrase used to identify relevant files.

24. Employment words describe what information should be extracted from
    those files; they are not normally the primary identity/search phrase.

25. Generic question words such as "where", "who", and "what" should
    normally NOT be used for Drive candidate retrieval.

26. Employment-context words such as "currently", "working", "works",
    "working now", "employer", "employment", and "job" should normally
    NOT be used for Drive candidate retrieval when they only describe
    what information the user wants from an already-identified document.

27. Generic retrieval words such as "file", "files", "document",
    "documents", "information", and "content" should normally NOT be used
    for Drive candidate retrieval.

EXAMPLE:

Question:
What degree was awarded in my Computer Science diploma?

Good JSON:

{{
  "intent": "degree information",
  "required_terms": ["diploma"],
  "phrases": ["computer science"],
  "optional_terms": [],
  "context_terms": ["degree", "awarded"],
  "exclude_terms": [],
  "answer_type": "degree",
  "confidence": 0.9
}}

EXAMPLE 2:

Question:
What was my final grade in thermodynamics?

Good JSON:

{{
  "intent": "final grade",
  "required_terms": ["thermodynamics"],
  "phrases": [],
  "optional_terms": ["grade"],
  "context_terms": ["final"],
  "exclude_terms": [],
  "answer_type": "grade",
  "confidence": 0.9
}}

EXAMPLE 3:

Question:
Find my internship report from 2024.

Good JSON:

{{
  "intent": "internship report",
  "required_terms": ["internship"],
  "phrases": ["internship report"],
  "optional_terms": ["2024"],
  "context_terms": [],
  "exclude_terms": [],
  "answer_type": "document",
  "confidence": 0.9
}}

EXAMPLE 4:

Question:
What are the main topics covered in my research documents?

Good JSON:

{{
  "intent": "research topics",
  "required_terms": ["research"],
  "phrases": [],
  "optional_terms": [],
  "context_terms": ["main topics", "covered"],
  "exclude_terms": [],
  "answer_type": "summary",
  "confidence": 0.9
}}

IMPORTANT:

Do NOT return:
"research documents"

The word "documents" is generic and should not be a Drive retrieval term.

EXAMPLE 5:

Question:
Who is authorized to operate the laboratory vehicle? SEARCH: AUTHORIZATION LETTERS

Good JSON:

{{
    "intent": "laboratory vehicle authorization",
    "required_terms": ["authorization letter"],
    "phrases": [],
    "optional_terms": ["laboratory vehicle"],
    "context_terms": ["authorized", "operate"],
    "exclude_terms": [],
    "answer_type": "person",
    "confidence": 0.95
}}

Now create the search plan for this question:

Question:
{question}
"""


# ============================================================================
# SEARCH PLAN INTERPRETATION
# ============================================================================

def interpret_search_request(question: str):
    """
    Turn a user question into a validated Google Drive retrieval plan.

    Processing pipeline:

        1. Ask Ollama for a structured search plan.
        2. Remove Markdown JSON fences if the model adds them.
        3. Parse the JSON.
        4. Validate every search-plan category.
        5. Normalize all values.
        6. Remove generic question words.
        7. Move employment-context terms out of retrieval categories.
        8. Remove generic retrieval words, including inside phrases.
        9. Remove duplicates across categories.
        10. Validate metadata.
        11. Enforce the maximum search-term count.
        12. Return the final deterministic plan.

    The function raises LocalLLMError when the model returns unusable data.
    """

    # ------------------------------------------------------------------
    # STEP 1: Build the planner prompt.
    # ------------------------------------------------------------------

    prompt = _build_search_planner_prompt(question)

    # ------------------------------------------------------------------
    # STEP 2: Ask Ollama for structured JSON.
    # ------------------------------------------------------------------

    raw_plan = generate_local_response(
        prompt,
        response_format=SEARCH_PLAN_SCHEMA,
    ).strip()

    # ------------------------------------------------------------------
    # STEP 3: Remove Markdown code fences if the local model added them.
    # ------------------------------------------------------------------

    if raw_plan.startswith("```"):
        raw_plan = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw_plan,
            flags=re.IGNORECASE,
        ).strip()

    # ------------------------------------------------------------------
    # STEP 4: Parse JSON.
    # ------------------------------------------------------------------

    try:
        plan = json.loads(raw_plan)

    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        print(
            "[LOCAL LLM ERROR]",
            "interpret_search_request",
            type(exc).__name__,
            repr(exc),
        )

        print("[RAW SEARCH PLAN]", raw_plan)

        raise LocalLLMError(
            "The local AI model returned an invalid search plan."
        ) from exc

    if not isinstance(plan, dict):
        raise LocalLLMError(
            "The local AI model returned an invalid search plan."
        )

    # ------------------------------------------------------------------
    # STEP 5: Validate and normalize all search categories.
    # ------------------------------------------------------------------

    validated = {
        key: _clean_values(plan, key)
        for key in SEARCH_PLAN_KEYS
    }

    # ------------------------------------------------------------------
    # STEP 5a: Validate and normalize the optional explicit time range.
    # ------------------------------------------------------------------

    if "time_range" in plan:
        validated["time_range"] = _normalize_time_range(plan.get("time_range"))
    else:
        parsed_range = _extract_time_range_from_question(question)
        validated["time_range"] = parsed_range if parsed_range else {"from": None, "to": None}

    if validated["time_range"] is None:
        validated["time_range"] = {"from": None, "to": None}

    if validated["time_range"].get("from") is not None and validated["time_range"].get("to") is not None:
        from_date = date.fromisoformat(validated["time_range"]["from"])
        to_date = date.fromisoformat(validated["time_range"]["to"])
        if from_date > to_date:
            raise LocalLLMError(
                "The local AI model returned an invalid time range."
            )

    explicit_date_constraints = _extract_explicit_date_constraints(question)
    validated["document_dates"] = explicit_date_constraints["document_dates"]
    validated["activity_dates"] = explicit_date_constraints["activity_dates"]

    for key in SEARCH_PLAN_KEYS:
        validated[key] = [
            value
            for value in validated[key]
            if not (validated["time_range"].get("from") is not None or validated["time_range"].get("to") is not None)
            or not _looks_like_date_value(value)
        ]

    # =========================================================================
    # DETERMINISTIC CORRECTION LAYER
    # =========================================================================
    #
    # The model is allowed to make an initial interpretation, but several
    # categories must be enforced deterministically before the search plan
    # reaches the Drive search layer.
    #
    # This prevents model variability from changing the fundamental meaning
    # of retrieval.
    # =========================================================================

    # ------------------------------------------------------------------
    # STEP 6: Remove generic question/formatting words.
    #
    # These should never affect Drive retrieval or relevance scoring.
    # ------------------------------------------------------------------

    for key in SEARCH_PLAN_KEYS:
        validated[key] = [
            value
            for value in validated[key]
            if value not in GENERIC_QUESTION_TERMS
        ]

    # ------------------------------------------------------------------
    # STEP 6b: Remove generic request/date-control words that should not
    # affect relevance scores.
    # ------------------------------------------------------------------

    for key in SEARCH_PLAN_KEYS:
        cleaned_values = []

        for value in validated[key]:
            stripped_value = _strip_generic_control_terms(value)

            if not stripped_value:
                continue

            cleaned_values.append(stripped_value)

        validated[key] = cleaned_values

    # ------------------------------------------------------------------
    # STEP 7: Move employment-context terms out of retrieval categories.
    #
    # Example:
    #
    #     optional_terms = ["working"]
    #
    # becomes:
    #
    #     optional_terms = []
    #     context_terms = ["working"]
    # ------------------------------------------------------------------

    moved_context_terms = []

    # ..................................................................

    # optional_terms
    remaining_optional_terms = []

    for value in validated["optional_terms"]:
        if value in EMPLOYMENT_CONTEXT_TERMS:
            moved_context_terms.append(value)
            continue

        remaining_optional_terms.append(value)

    validated["optional_terms"] = remaining_optional_terms

    # ..................................................................

    # required_terms and phrases
    for key in (
        "required_terms",
        "phrases",
    ):
        remaining_values = []

        for value in validated[key]:
            if value in EMPLOYMENT_CONTEXT_TERMS:
                moved_context_terms.append(value)
                continue

            remaining_values.append(value)

        validated[key] = remaining_values

    # ..................................................................

    # Keep any employment terms that were already correctly placed in
    # context_terms and append terms moved from the retrieval categories.
    validated["context_terms"].extend(
        moved_context_terms
    )

    # ------------------------------------------------------------------
    # STEP 8: Remove generic retrieval words.
    #
    # This is the important fix for cases such as:
    #
    #     "research documents"
    #
    # which must become:
    #
    #     "research"
    #
    # before the plan reaches drive.py.
    # ------------------------------------------------------------------

    validated = _clean_generic_retrieval_terms(validated)

    # ------------------------------------------------------------------
    # STEP 8b: Preserve any explicit SEARCH: anchor supplied by the user.
    # ------------------------------------------------------------------

    search_anchor = _extract_explicit_search_anchor(question)
    if search_anchor:
        normalized_anchor = _normalize_value(search_anchor)
        anchor_values = [
            value
            for value in (
                normalized_anchor,
                normalized_anchor.replace("letters", "letter"),
            )
            if value and value.strip()
        ]

        for key in ("required_terms", "phrases"):
            for value in anchor_values:
                if value not in validated[key]:
                    validated[key].insert(0, value)

        for key in SEARCH_PLAN_KEYS:
            validated[key] = [
                value
                for value in validated[key]
                if not (
                    value.lower() == "coordinates"
                    and normalized_anchor.lower().startswith("coordination")
                )
            ]

    # ------------------------------------------------------------------
    # STEP 9: Remove duplicates across categories.
    #
    # The first category containing a value keeps it.
    # ------------------------------------------------------------------

    seen_values = set()

    for key in SEARCH_PLAN_KEYS:
        unique_values = []

        for value in validated[key]:
            if value in seen_values:
                continue

            seen_values.add(value)
            unique_values.append(value)

        validated[key] = unique_values

    # ------------------------------------------------------------------
    # STEP 10: Ensure the model produced at least one usable search value.
    # ------------------------------------------------------------------

    if not any(
        validated[key]
        for key in SEARCH_PLAN_KEYS
    ):
        raise LocalLLMError(
            "The local AI model returned an empty search plan."
        )

    # =========================================================================
    # METADATA VALIDATION
    # =========================================================================

    # ------------------------------------------------------------------
    # STEP 11: Validate intent.
    # ------------------------------------------------------------------

    intent = plan.get("intent", "")

    if not isinstance(intent, str):
        raise LocalLLMError(
            "The local AI model returned an invalid search plan."
        )

    # ------------------------------------------------------------------
    # STEP 12: Validate answer_type.
    # ------------------------------------------------------------------

    answer_type = plan.get("answer_type", "")

    if not isinstance(answer_type, str):
        raise LocalLLMError(
            "The local AI model returned an invalid search plan."
        )

    # ------------------------------------------------------------------
    # STEP 13: Validate confidence.
    # ------------------------------------------------------------------

    confidence = plan.get("confidence", 0.0)

    if not isinstance(confidence, (int, float)):
        raise LocalLLMError(
            "The local AI model returned an invalid search plan."
        )

    if not 0 <= confidence <= 1:
        raise LocalLLMError(
            "The local AI model returned an invalid search plan."
        )

    # ------------------------------------------------------------------
    # STEP 14: Normalize metadata.
    # ------------------------------------------------------------------

    validated["intent"] = intent.strip()[:300]
    validated["answer_type"] = answer_type.strip()[:100]
    validated["confidence"] = float(confidence)
    validated["query_constraints"] = extract_query_constraints(question)

    # =========================================================================
    # FINAL SAFETY VALIDATION
    # =========================================================================

    # ------------------------------------------------------------------
    # STEP 15: Enforce the maximum total number of search terms.
    #
    # This keeps the generated Drive query bounded and prevents the local
    # model from flooding the retrieval layer with weak terms.
    # ------------------------------------------------------------------

    total_terms = sum(
        len(validated[key])
        for key in SEARCH_PLAN_KEYS
    )

    if total_terms > 12:
        raise LocalLLMError(
            "The local AI model returned too many search terms."
        )

    # ------------------------------------------------------------------
    # STEP 16: Log the final search plan.
    #
    # This is intentionally logged AFTER all deterministic corrections so
    # debugging output shows the exact plan that reaches the next layer.
    # ------------------------------------------------------------------

    print(
        "[SEARCH PLAN]",
        json.dumps(
            validated,
            indent=2,
        ),
    )

    if validated.get("time_range"):
        print(
            "[TIME RANGE]",
            json.dumps(
                validated["time_range"],
                sort_keys=True,
            ),
        )

    return validated