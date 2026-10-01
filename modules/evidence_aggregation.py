"""Generic validated-evidence aggregation helpers.

Coordination Letter extraction remains specialized, but the aggregation decision is
made from the canonical query constraints and validated evidence rather than from
hardcoded document-type triggers.
"""

import re
from datetime import date
from dataclasses import dataclass

from modules.drive import is_list_aggregation_query
from modules.evidence import _coordination_letter_subject_supported
from modules.query_constraints import extract_query_constraints


MONTH_PATTERN = (
    r"January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|"
    r"Sept|Oct|Nov|Dec"
)
DATE_PATTERN = (
    rf"(?:{MONTH_PATTERN})\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s+\d{{4}}"
    rf"|\d{{4}}-\d{{1,2}}-\d{{1,2}}"
)
PLANNED_ACTIVITY_PATTERN = re.compile(
    rf"\bthat\s+(?P<company>.+?)\s+will\s+be\s+"
    rf"(?P<activity>[^.\n]*?)\bon\s+(?P<date>{DATE_PATTERN})\b",
    re.IGNORECASE,
)
ALTERNATE_PLANNED_ACTIVITY_PATTERN = re.compile(
    rf"\b(?P<company>[A-Z][A-Za-z0-9&.'-]*(?:[ \t]+[A-Z][A-Za-z0-9&.'-]*){{0,5}})[ \t]+"
    rf"(?i:will\s+be\s+|will\s+|plans?\s+to\s+|is\s+scheduled\s+to\s+)"
    rf"(?P<activity>[^.\n]*?)\s+(?i:on|for)\s+(?P<date>(?i:{DATE_PATTERN}))\b",
)
DATE_VALUE_PATTERN = re.compile(
    rf"(?P<month>{MONTH_PATTERN})\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s*,?\s+"
    r"(?P<year>\d{4})",
    re.IGNORECASE,
)
EXPLICIT_COMPANY_SCOPE_PATTERN = re.compile(
    r"^\s*for\s+(?P<company>[^,?]+?)\s*,",
    re.IGNORECASE,
)


def normalize_company_name(value):
    """Normalize punctuation and spacing for conservative company matching."""
    return " ".join(
        re.findall(
            r"[a-z0-9]+",
            value.casefold(),
        )
    )


def _explicit_company_scope(question):
    match = EXPLICIT_COMPANY_SCOPE_PATTERN.search(str(question))
    if not match:
        return None
    company = re.sub(r"\s+", " ", match.group("company")).strip()
    normalized = company.casefold()
    company_markers = (
        " inc",
        " corporation",
        " industries",
        " engineering",
        " services",
        " llc",
        " limited",
        " group",
    )
    if any(marker in normalized for marker in company_markers):
        return company
    is_organization_acronym = (
        company.isupper()
        and len(company) > 1
        and len(company.split()) == 1
    )
    is_company_list_scope = bool(
        re.search(r"\b(?:companies|list\s+all|all\s+planned|dates?)\b", str(question), re.IGNORECASE)
    )
    return company if is_organization_acronym and is_company_list_scope else None


def _coordination_letter_prefix(text):
    """Return the leading document window used for coordination-letter schema checks."""
    if not isinstance(text, str):
        return ""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    prefix = "\n".join(paragraphs[:3])
    return prefix[:2000]


def _planned_activity_matches(text):
    matches = list(PLANNED_ACTIVITY_PATTERN.finditer(text))
    primary_spans = [(match.start(), match.end()) for match in matches]
    matches.extend(
        match
        for match in ALTERNATE_PLANNED_ACTIVITY_PATTERN.finditer(text)
        if not any(
            match.start() < end and start < match.end()
            for start, end in primary_spans
        )
    )
    unique_matches = {}
    for match in matches:
        key = (
            normalize_company_name(match.group("company")),
            re.sub(r"\s+", " ", match.group("activity").casefold()).strip(),
            match.group("date").casefold(),
        )
        unique_matches.setdefault(key, match)
    return sorted(unique_matches.values(), key=lambda match: (match.start(), match.end()))


def supporting_document_names(aggregation):
    """Return unique source names that support the displayed records."""
    return list(
        dict.fromkeys(
            record["source"]
            for record in aggregation["records"]
        )
    )


@dataclass(frozen=True)
class EligibilityResult:
    """Structured proof that a specialized handler is compatible with current evidence."""

    eligible: bool
    reason: str = ""

    @property
    def proven(self):
        return self.eligible


class SpecializedHandler:
    """Small contract for evidence-qualified specialized document processing."""

    name = "generic"
    required_schema = {}

    def determine_eligibility(self, question, search_plan, documents):
        raise NotImplementedError

    def extract_records(self, question, documents, search_plan):
        raise NotImplementedError


def _coordination_letter_query_hint(question, search_plan):
    """A lightweight candidate-discovery signal only; not proof of eligibility."""
    if not isinstance(search_plan, dict):
        return False

    answer_context = " ".join(
        [
            str(question),
            str(search_plan.get("intent", "")),
            str(search_plan.get("answer_type", "")),
            *[
                str(value)
                for key in (
                    "required_terms",
                    "phrases",
                    "optional_terms",
                    "context_terms",
                )
                for value in (search_plan.get(key, []) or [])
            ],
        ]
    )
    return re.search(
        r"\bcoordination\s+letters?\b",
        answer_context,
        re.IGNORECASE,
    ) is not None


class CoordinationLetterHandler(SpecializedHandler):
    """Prove the current evidence actually contains a Coordination Letter schema."""

    name = "coordination_letter"
    required_schema = {
        "document_type": "Coordination Letter",
        "required_fields": [
            "company",
            "planned_activity_date",
            "relationship",
        ],
        "semantics": "company is engaged in a planned activity on a specific date",
    }

    def determine_eligibility(self, question, search_plan, documents):
        if not isinstance(documents, list) or not documents:
            return EligibilityResult(False, "no evidence documents supplied")

        if re.search(
            r"\bactual(?:ly)?\b.*\b(?:entry|enter(?:ed)?)\b",
            str(question),
            re.IGNORECASE,
        ):
            return EligibilityResult(False, "question asks for actual entry dates, not planned activity dates")

        candidate_hint = _coordination_letter_query_hint(question, search_plan)
        constraints = (
            search_plan.get("query_constraints")
            if isinstance(search_plan, dict)
            else None
        ) or extract_query_constraints(question)
        company_scope = _explicit_company_scope(question)
        if company_scope:
            constraints = dict(constraints)
            constraints["subject_entity"] = company_scope
            constraints["subject_type"] = "company"

        proven_document = False
        for document in documents:
            text = document.get("text", "")
            if not isinstance(text, str):
                text = ""

            if not re.search(
                r"\bcoordination\s+letters?\b",
                _coordination_letter_prefix(text),
                re.IGNORECASE,
            ):
                continue

            matches = _planned_activity_matches(text)
            if not matches:
                continue

            for match in matches:
                company = re.sub(r"\s+", " ", match.group("company")).strip().rstrip(".").strip()
                planned_date = parse_planned_date(match.group("date"))
                if company and planned_date and _coordination_event_matches_constraints(
                    match,
                    text,
                    constraints,
                ):
                    proven_document = True
                    break

            if proven_document:
                break

        if not proven_document and not candidate_hint:
            return EligibilityResult(False, "query and current evidence do not prove Coordination Letter schema")

        if not proven_document:
            return EligibilityResult(False, "current evidence does not prove Coordination Letter company/date schema")

        return EligibilityResult(True, "current evidence proves the Coordination Letter schema")

    def extract_records(self, question, documents, search_plan):
        return extract_coordination_letter_records(question, documents, search_plan)


def get_specialized_handlers():
    """Return the currently registered specialized handlers."""
    return [CoordinationLetterHandler()]


def evaluate_specialized_handlers(question, documents, search_plan):
    """Select a specialized handler only when validated query semantics require it."""
    constraints = (
        search_plan.get("query_constraints")
        if isinstance(search_plan, dict)
        else None
    ) or extract_query_constraints(question)
    aggregation_requirements = determine_aggregation_requirements(
        constraints,
        documents,
        question,
    )
    if (
        not aggregation_requirements.get("requires_aggregation")
        and constraints.get("document_type") != "Coordination Letter"
    ):
        return None

    for handler in get_specialized_handlers():
        eligibility = handler.determine_eligibility(question, search_plan, documents)
        if eligibility.proven:
            return handler
    return None


def _company_subject_matches(subject, company):
    """Match explicit company scopes conservatively without accepting unrelated subsidiaries."""
    if not subject or not company:
        return False
    subject_key = normalize_company_name(subject)
    company_key = normalize_company_name(company)
    if company_key == subject_key:
        return True
    if not company_key.startswith(f"{subject_key} "):
        return False
    subject_tokens = subject_key.split()
    company_tokens = company_key.split()
    if len(subject_tokens) <= 2:
        return True
    return False


def _coordination_event_matches_constraints(match, text, constraints):
    """Verify requested entity/activity against the same current record evidence."""
    subject = constraints.get("subject_entity")
    subject_type = constraints.get("subject_type")
    if subject and subject_type == "company":
        company = re.sub(r"\s+", " ", match.group("company")).strip().rstrip(".").strip()
        if not _company_subject_matches(subject, company):
            return False
    elif subject:
        start = max(0, match.start() - 300)
        end = min(len(text), match.end() + 300)
        if not _coordination_letter_subject_supported(
            text[start:end],
            subject,
            subject_type,
        ):
            return False

    activity = constraints.get("activity")
    if activity:
        expected = re.findall(r"[a-z0-9]+", activity.casefold())
        actual = re.findall(r"[a-z0-9]+", match.group("activity").casefold())
        if not expected or any(term not in actual for term in expected):
            return False

    return True


def determine_aggregation_requirements(query_constraints, validated_evidence=None, question=None):
    """Return the generic aggregation requirement derived from semantic constraints.

    The decision is based on the canonical query structure and validated evidence,
    not on a document-type keyword or a hardcoded Coordination Letter trigger.
    """
    if not isinstance(query_constraints, dict):
        return {
            "requires_aggregation": False,
            "mode": "SINGLE_SOURCE",
            "reason": "no query constraints",
        }

    documents = validated_evidence or []
    evidence_count = len(documents) if isinstance(documents, (list, tuple)) else 0
    if evidence_count <= 1:
        return {
            "requires_aggregation": False,
            "mode": "SINGLE_SOURCE",
            "reason": "one validated source is sufficient",
        }

    requested_field = query_constraints.get("requested_field")
    subject_entity = query_constraints.get("subject_entity")
    question_text = str(question or "")
    list_markers = re.search(
        r"\b(?:all|list\s+all|for\s+each|across|multiple|every|which\s+companies|each\s+company)\b",
        question_text,
        re.IGNORECASE,
    )
    if list_markers or requested_field in {"risks", "discussion"}:
        return {
            "requires_aggregation": True,
            "mode": "MULTI_RECORD",
            "reason": "question requires combined validated claims",
        }

    if subject_entity and evidence_count > 1 and requested_field in {
        "planned_activity_date",
        "entry_date",
        "date",
    }:
        return {
            "requires_aggregation": False,
            "mode": "SINGLE_SOURCE",
            "reason": "the same subject should resolve from one validated claim unless the question explicitly asks for a list",
        }

    return {
        "requires_aggregation": False,
        "mode": "SINGLE_SOURCE",
        "reason": "no multi-source aggregation is required",
    }


def parse_planned_date(value):
    """Parse an explicit month-name or ISO planned-activity date."""
    date_match = DATE_VALUE_PATTERN.fullmatch(value.strip())

    if date_match:
        month = date_match.group("month")
        day = int(date_match.group("day"))
        year = int(date_match.group("year"))
        month_names = {
            "jan": 1,
            "january": 1,
            "feb": 2,
            "february": 2,
            "mar": 3,
            "march": 3,
            "apr": 4,
            "april": 4,
            "may": 5,
            "jun": 6,
            "june": 6,
            "jul": 7,
            "july": 7,
            "aug": 8,
            "august": 8,
            "sep": 9,
            "sept": 9,
            "september": 9,
            "oct": 10,
            "october": 10,
            "nov": 11,
            "november": 11,
            "dec": 12,
            "december": 12,
        }
        try:
            return date(year, month_names[month.casefold()], day)
        except ValueError:
            return None

    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _date_bound(search_plan, key, question):
    value = (search_plan.get("time_range") or {}).get(key)
    if value is None:
        if key == "to" and re.search(
            r"\bto\s+present\b",
            str(question),
            re.IGNORECASE,
        ):
            return date.today()
        return None

    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _extract_document_dates(question):
    """Extract exact document/submission dates from the question."""
    if not isinstance(question, str):
        return []

    dates = []
    for match in re.finditer(
        rf"(?:{MONTH_PATTERN})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*,?\s*\d{{4}})?|\d{{4}}-\d{{1,2}}-\d{{1,2}}",
        question,
        re.IGNORECASE,
    ):
        date_value = parse_planned_date(match.group(0))
        if date_value is None:
            continue
        context = question[max(0, match.start() - 25):match.end() + 40].lower()
        if re.search(r"\b(?:submitted|document|letter|dated|filed)\b", context, re.IGNORECASE):
            dates.append(date_value)

    return list(dict.fromkeys(dates))


def _extract_question_date(question):
    """Return a single explicit planned/sampling date from the question when present."""
    if not isinstance(question, str):
        return None

    text = question.strip()
    if not text:
        return None

    if re.search(r"\b(?:submitted|document|letter|dated|filed)\b", text, re.IGNORECASE):
        return None

    matches = []
    for match in re.finditer(
        rf"(?:{MONTH_PATTERN})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*,?\s*\d{{4}})?|\d{{4}}-\d{{1,2}}-\d{{1,2}}",
        text,
        re.IGNORECASE,
    ):
        value = parse_planned_date(match.group(0))
        if value is not None:
            matches.append(value)

    if not matches:
        return None

    return matches[0]


def _extract_submission_dates(text):
    """Parse submission/document dates in a Coordination Letter document."""
    if not isinstance(text, str):
        return []

    dates = []
    for match in re.finditer(
        rf"(?:DATE\s+SUBMITTED|DATE\s+FILED|DATE\s+ISSUED|DATED|SUBMITTED\s+ON)\s*[:\-]?\s*(?:{MONTH_PATTERN})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*,?\s*\d{{4}})?|\d{{4}}-\d{{1,2}}-\d{{1,2}}",
        text,
        re.IGNORECASE,
    ):
        parsed = parse_planned_date(match.group(0).split(":")[-1].strip() if ":" in match.group(0) else match.group(0))
        if parsed is not None:
            dates.append(parsed)

    return list(dict.fromkeys(dates))


def extract_coordination_letter_records(question, documents, search_plan):
    """Extract evidence-backed company/planned-date records for this query type."""
    if re.search(r"\bactual(?:ly)?\b.*\b(?:entry|enter(?:ed)?)\b", str(question), re.IGNORECASE):
        return None

    requested_date = _extract_question_date(question)
    document_dates = _extract_document_dates(question)

    lower_bound = _date_bound(search_plan, "from", question)
    upper_bound = _date_bound(search_plan, "to", question)
    requested_company = _explicit_company_scope(question)
    constraints = (
        search_plan.get("query_constraints")
        if isinstance(search_plan, dict)
        else None
    ) or extract_query_constraints(question)
    if not requested_company and constraints.get("subject_type") == "company":
        requested_company = constraints.get("subject_entity")
    event_constraints = constraints
    if requested_company:
        event_constraints = dict(constraints)
        event_constraints["subject_entity"] = requested_company
        event_constraints["subject_type"] = "company"
    records = []
    unresolved = []

    for document in documents:
        name = str(document.get("name", "Unnamed file"))
        text = document.get("text", "")
        if not isinstance(text, str):
            text = ""

        if not re.search(
            r"\bcoordination\s+letters?\b",
            _coordination_letter_prefix(text),
            re.IGNORECASE,
        ):
            continue

        if document_dates:
            submission_dates = _extract_submission_dates(text)
            if not any(doc_date in submission_dates for doc_date in document_dates):
                continue

        matches = _planned_activity_matches(text)
        if not matches:
            unresolved.append(name)
            continue

        found_supported_event = False
        for match in matches:
            company = re.sub(
                r"\s+",
                " ",
                match.group("company"),
            ).strip().rstrip(".").strip()
            planned_date = parse_planned_date(match.group("date"))
            if not company or planned_date is None:
                unresolved.append(name)
                continue

            found_supported_event = True
            if not _coordination_event_matches_constraints(match, text, event_constraints):
                continue
            if requested_date is not None and planned_date != requested_date:
                continue
            if lower_bound and planned_date < lower_bound:
                continue
            if upper_bound and planned_date > upper_bound:
                continue

            record = {
                "company": company,
                "date": planned_date,
                "source": name,
                "evidence": match.group(0),
            }
            source_id = document.get("file_id") or document.get("candidate_id")
            if source_id:
                record["source_id"] = source_id
            records.append(record)

        if not found_supported_event and not any(
            unresolved_name == name
            for unresolved_name in unresolved
        ):
            unresolved.append(name)

    scope_unresolved = None
    if requested_company:
        requested_key = normalize_company_name(requested_company)
        matching_records = [
            record
            for record in records
            if _company_subject_matches(requested_company, record["company"])
        ]
        matched_company_names = {
            normalize_company_name(record["company"])
            for record in matching_records
        }
        if len(matched_company_names) > 1:
            records = []
            scope_unresolved = requested_company
        else:
            records = matching_records
            if not records:
                scope_unresolved = requested_company

    aggregation = {
        "records": records,
        "unresolved": unresolved,
    }
    if requested_company:
        aggregation["requested_company"] = requested_company
        aggregation["scope_unresolved"] = scope_unresolved

    return aggregation


def format_coordination_letter_records(aggregation):
    """Render the authoritative aggregation without asking the LLM to select records."""
    grouped_records = {}
    display_names = {}

    for record in aggregation["records"]:
        company_key = record["company"].casefold()
        display_names.setdefault(company_key, record["company"])
        grouped_records.setdefault(company_key, []).append(record)

    lines = ["Companies and planned activity dates from the retrieved Coordination Letters:"]

    if grouped_records:
        for company_key, company_records in grouped_records.items():
            company_records.sort(key=lambda record: record["date"])
            date_values = [
                f"{record['date'].strftime('%B')} {record['date'].day}, "
                f"{record['date'].year} ({record['source']})"
                for record in company_records
            ]
            lines.append(
                f"- {display_names[company_key]}: "
                + "; ".join(date_values)
            )
    else:
        if aggregation.get("scope_unresolved"):
            lines.append(
                f"- No qualifying planned activity date could be uniquely "
                f"matched to {aggregation['scope_unresolved']} in the supplied letters."
            )
        else:
            lines.append("- No company/planned activity date could be established from the supplied letters.")

    if aggregation["unresolved"]:
        lines.append("")
        lines.append("Unresolved Coordination Letters:")
        lines.extend(
            f"- {name}: company or planned activity date not established from the extracted text."
            for name in aggregation["unresolved"]
        )

    return "\n".join(lines)