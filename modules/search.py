"""Ollama-assisted search planning for Google Drive retrieval."""

import json
import re

from modules.llm import LocalLLMError, generate_local_summary


SEARCH_PLAN_KEYS = (
    "required_terms",
    "phrases",
    "optional_terms",
    "context_terms",
    "exclude_terms",
)


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
    ["computer science diploma"]

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

- context_terms:
  Words describing what the user wants to find or answer.
  These are mainly useful for ranking documents after retrieval.

    For example, "Who is John Smith and where does he currently work?"
    should produce terms such as ["currently", "work", "employer"].
    Do not put generic formatting words such as "paragraph" here.

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

EXAMPLE:

Question:
What degree was awarded in my Computer Science diploma?

Good JSON:

{{
  "intent": "degree information",
    "required_terms": ["diploma"],
    "phrases": ["computer science diploma"],
    "optional_terms": ["computer science"],
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
    "required_terms": ["rhys pacio"],
    "phrases": ["rhys pacio"],
    "optional_terms": ["employer"],
    "context_terms": ["currently", "working"],
    "exclude_terms": [],
    "answer_type": "biography",
    "confidence": 0.95
}}

Now create the search plan for this question:

Question:
{question}
"""

    raw_plan = generate_local_summary(prompt).strip()

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

    # Remove duplicates across categories.
    # The first category that contains a value keeps it.
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
