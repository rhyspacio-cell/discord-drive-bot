"""Ollama-assisted search planning for Google Drive retrieval."""

import json
import re

from modules.llm import LocalLLMError, generate_local_response


SEARCH_PLAN_KEYS = (
    "required_terms",
    "phrases",
    "optional_terms",
    "context_terms",
    "exclude_terms",
)


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
        "answer_type",
        "confidence",
    ],
}


def _normalize_value(value: str) -> str:
    """Normalize one search value returned by the local model."""
    value = value.strip().lower()

    # Drive search should use normal spaces rather than underscores.
    value = value.replace("_", " ")

    # Collapse repeated whitespace.
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def _clean_values(plan, key):
    """Validate and clean one search-plan category."""
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

        if len(value) > 100:
            continue

        if value in seen:
            continue

        seen.add(value)
        cleaned.append(value)

    return cleaned


def interpret_search_request(question: str):
    """Turn a user question into a validated Google Drive retrieval plan."""

    prompt = f"""
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

14. The total number of values across all five search-term lists must be
    12 or fewer.

15. Keep search values lowercase.

16. confidence must be a number between 0.0 and 1.0.

17. The search plan must be based only on the user's question.

18. The question's subject is more important than generic question words.

19. When the user asks what a person currently does, where they work,
    or who employs them, treat employment-related words as context_terms,
    not optional_terms, unless the word is specifically needed to identify
    the relevant document.

20. When an employment question contains "working now", "works now",
    "currently working", "current employer", "where they work", or similar
    wording, put those concepts in context_terms. Do NOT put them in
    optional_terms unless there is an unusual and specific reason that the
    phrase itself identifies a document.

21. For a question about a person's current employment, the person's name
    should normally be the primary phrase used to identify relevant files.
    Employment words describe what information should be extracted from
    those files; they are not normally the primary identity/search phrase.

22. Generic question words such as "where", "who", and "what" should
    normally NOT be used for Drive candidate retrieval.

23. Employment-context words such as "currently", "working", "works",
    "working now", "employer", "employment", and "job" should normally
    NOT be used for Drive candidate retrieval when they only describe
    what information the user wants from an already-identified document.

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
Give a small paragraph on who Rhys Pacio is and where he is currently working.

Good JSON:

{{
  "intent": "person employment information",
  "required_terms": [],
  "phrases": ["rhys pacio"],
  "optional_terms": [],
  "context_terms": ["currently", "working"],
  "exclude_terms": [],
  "answer_type": "biography",
  "confidence": 0.95
}}

EXAMPLE 5:

Question:
Give a small paragraph on who Rhys Pacio is and where he is currently working now.

Good JSON:

{{
  "intent": "person employment information",
  "required_terms": [],
  "phrases": ["rhys pacio"],
  "optional_terms": [],
  "context_terms": ["currently", "working", "working now"],
  "exclude_terms": [],
  "answer_type": "biography",
  "confidence": 0.95
}}

Now create the search plan for this question:

Question:
{question}
"""

    raw_plan = generate_local_response(
        prompt,
        response_format=SEARCH_PLAN_SCHEMA,
    ).strip()

    # Some local models wrap JSON in Markdown code fences.
    if raw_plan.startswith("```"):
        raw_plan = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw_plan,
            flags=re.IGNORECASE,
        ).strip()

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

    # Validate and normalize all search categories.
    validated = {
        key: _clean_values(plan, key)
        for key in SEARCH_PLAN_KEYS
    }

    # ------------------------------------------------------------------
    # Deterministic correction layer.
    #
    # The local model may occasionally place employment/question-context
    # terms into the wrong category.
    #
    # Generic question words must NEVER participate in relevance
    # scoring. Employment-context words describe WHAT information the
    # user wants from a matching document and should not broaden Drive
    # candidate retrieval.
    # ------------------------------------------------------------------

    employment_context_terms = {
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

    generic_question_terms = {
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

    moved_context_terms = []

    # First, clean generic question/formatting words out of EVERY
    # category. They must never affect Drive retrieval or relevance
    # scoring.
    for key in SEARCH_PLAN_KEYS:
        validated[key] = [
            value
            for value in validated[key]
            if value not in generic_question_terms
        ]

    # Move employment-context terms out of optional_terms.
    remaining_optional_terms = []

    for value in validated["optional_terms"]:
        if value in employment_context_terms:
            moved_context_terms.append(value)
            continue

        remaining_optional_terms.append(value)

    validated["optional_terms"] = remaining_optional_terms

    # Also move employment-context terms from other retrieval-oriented
    # categories into context_terms when the model put them there.
    for key in (
        "required_terms",
        "phrases",
    ):
        remaining_values = []

        for value in validated[key]:
            if value in employment_context_terms:
                moved_context_terms.append(value)
                continue

            remaining_values.append(value)

        validated[key] = remaining_values

    # If the model already placed employment terms in context_terms,
    # keep them there. Add any terms that were moved from the other
    # categories.
    validated["context_terms"].extend(
        moved_context_terms
    )

    # ------------------------------------------------------------------
    # Remove duplicates across categories.
    #
    # The first category that contains a value keeps it.
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

    if not any(validated[key] for key in SEARCH_PLAN_KEYS):
        raise LocalLLMError(
            "The local AI model returned an empty search plan."
        )

    # Validate metadata fields.
    intent = plan.get("intent", "")
    answer_type = plan.get("answer_type", "")
    confidence = plan.get("confidence", 0.0)

    if not isinstance(intent, str):
        raise LocalLLMError(
            "The local AI model returned an invalid search plan."
        )

    if not isinstance(answer_type, str):
        raise LocalLLMError(
            "The local AI model returned an invalid search plan."
        )

    if not isinstance(confidence, (int, float)):
        raise LocalLLMError(
            "The local AI model returned an invalid search plan."
        )

    if not 0 <= confidence <= 1:
        raise LocalLLMError(
            "The local AI model returned an invalid search plan."
        )

    validated["intent"] = intent.strip()[:300]
    validated["answer_type"] = answer_type.strip()[:100]
    validated["confidence"] = float(confidence)

    # Final safety check.
    total_terms = sum(
        len(validated[key])
        for key in SEARCH_PLAN_KEYS
    )

    if total_terms > 12:
        raise LocalLLMError(
            "The local AI model returned too many search terms."
        )

    print(
        "[SEARCH PLAN]",
        json.dumps(validated, indent=2),
    )

    return validated