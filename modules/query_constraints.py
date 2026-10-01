"""Lightweight semantic constraints extracted from the user's original query."""

import re


PROPER_NAME = r"[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,3}"
ENTITY_EXCLUSIONS = {
    "what",
    "when",
    "where",
    "which",
    "who",
    "how",
    "the",
    "coordination",
    "letter",
    "letters",
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
}
ACTIVITY_EXCLUSIONS = {
    "a",
    "an",
    "and",
    "are",
    "by",
    "date",
    "dates",
    "day",
    "days",
    "did",
    "do",
    "does",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "planned",
    "please",
    "scheduled",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "activity",
    "activities",
    "company",
    "companies",
    "or",
    "stated",
    "specified",
    "submitted",
    "scheduled",
    "will",
    "take",
    "place",
    "involving",
    "involve",
    "during",
    "about",
    "regarding",
    "concerning",
    "associated",
    "related",
    "does",
    "did",
    "say",
    "says",
    "described",
    "describe",
    "is",
    "are",
    "were",
    "be",
    "mentioned",
    "include",
    "including",
    "applicable",
    "multiple",
    "every",
    "present",
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
    "sampling",
    "coordination",
    "letter",
    "letters",
}


def _find_subject(question):
    patterns = (
        rf"(?P<entity>{PROPER_NAME})['’]s\b",
        rf"\b(?i:for|about|regarding|concerning|involving|associated\s+with|related\s+to)\s+(?P<entity>{PROPER_NAME})\b",
        rf"\b(?i:represent|authorized\s+to\s+represent|authorize\w*\s+to\s+represent)\s+(?P<entity>{PROPER_NAME})\b",
        rf"\b(?P<entity>{PROPER_NAME})\s+(?:(?:is|was|has\s+been)\s+)?authorized\s+to\s+(?:operate|use|access|receive|bring|carry|perform|transport)\b",
        rf"\b(?i:who\s+is|where\s+is|what\s+is|when\s+is)\s+(?P<entity>{PROPER_NAME})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, question)
        if not match:
            continue
        entity = re.sub(r"\s+", " ", match.group("entity")).strip()
        if entity.casefold() in ENTITY_EXCLUSIONS:
            continue
        if entity.split()[0].casefold() in ENTITY_EXCLUSIONS:
            continue
        return entity, match.span("entity"), match.group(0).endswith(("'s", "’s"))

    return None, None, False


def _clean_activity(value):
    value = re.sub(r"\bSEARCH\s*:.+$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4}\b|\b\d{4}\b", " ", value)
    value = re.sub(
        r"^\s*(?:(?:what|when|where|which|who|how|is|are|was|were|the|a|an|date|of|for|in|on)\b[\s?.,]*)+",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", value.lower())
    tokens = [token for token in tokens if token not in ACTIVITY_EXCLUSIONS]
    while tokens and tokens[-1] in {"please", "all", "each", "any"}:
        tokens.pop()
    return " ".join(tokens[:8])


def _extract_activity(question, entity_span, possessive):
    if entity_span:
        before = question[:entity_span[0]]
        after = question[entity_span[1]:]
        if possessive:
            after = re.sub(r"^['’]s\b", " ", after)
            activity = _clean_activity(after)
            if activity:
                return activity
        return _clean_activity(before)

    without_anchor = re.sub(r"\bSEARCH\s*:.+$", "", question, flags=re.IGNORECASE)
    without_anchor = re.sub(
        r"^\s*(?:(?:what|when|where|which|who|how|is|are|was|were|the|a|an|date|of|for|in|on)\b[\s?.,]*)+",
        " ",
        without_anchor,
        flags=re.IGNORECASE,
    )
    return _clean_activity(without_anchor)


def extract_query_constraints(question):
    """Preserve explicit subject, activity, and requested-field constraints."""
    question = question if isinstance(question, str) else ""
    subject, subject_span, possessive = _find_subject(question)
    normalized = question.casefold()

    if re.search(r"\b(?:authorized|authorization|authorize)\b", normalized):
        requested_field = "authorization_role"
    elif re.search(r"\brisks?\b", normalized):
        requested_field = "risks"
    elif re.search(r"\b(?:discussed|discussion|what\s+was\s+covered)\b", normalized):
        requested_field = "discussion"
    elif re.search(r"\bwhat\s+(?:activity|task|event)\b", normalized):
        requested_field = "activity"
    elif re.search(r"\b(?:salary|compensation|remuneration|annual\s+pay|base\s+pay)\b", normalized):
        requested_field = "salary"
    elif re.search(
        r"\b(?:when|what\s+(?:day|dates?)|which\s+(?:day|dates?)|dates?|day)\b",
        normalized,
    ) or re.search(r"\b(?:planned|scheduled|specified)\s+(?:activity\s+)?(?:date|day)\b", normalized):
        requested_field = (
            "entry_date"
            if re.search(r"\bentered\b", normalized)
            else "planned_activity_date"
        )
    else:
        requested_field = None

    project_context = re.search(r"\bproject\b", normalized) and requested_field in {
        "risks",
        "discussion",
    }
    if project_context:
        subject = "project"
        subject_span = None
        possessive = False

    aggregate_query = re.search(
        r"\b(?:which\s+companies|all|list\s+all|all\s+applicable|for\s+each\s+company|entered\s+ipi)\b",
        normalized,
    ) is not None
    scheduling_context = re.search(
        r"\b(?:planned|scheduled|inspection|maintenance|training|activity|event|visit|assessment|meeting|risk|risks|discussed|discussion)\b",
        normalized,
    ) is not None
    if project_context and requested_field == "risks":
        activity = "project"
    elif project_context and requested_field == "discussion":
        activity = "project meeting"
    else:
        activity = (
            _extract_activity(question, subject_span, possessive)
            if scheduling_context and not (aggregate_query and not possessive)
            else None
        )
    if "coordination letter" in normalized and not subject and not possessive:
        activity = None

    document_type = (
        "Coordination Letter"
        if re.search(r"\bcoordination\s+letters?\b", normalized)
        else None
    )

    company_markers = (
        " services",
        " corporation",
        " industries",
        " engineering",
        " inc",
        " llc",
        " limited",
        " group",
    )
    subject_type = None
    if subject:
        if subject.casefold() == "project":
            subject_type = "project"
        elif any(marker in subject.casefold() for marker in company_markers):
            subject_type = "company"
        else:
            subject_type = "person"

    return {
        "subject_entity": subject,
        "requested_subject": subject,
        "subject_type": subject_type,
        "organization": subject if subject_type == "company" else None,
        "activity": activity or None,
        "requested_activity": activity or None,
        "requested_field": requested_field,
        "requested_relationship": (
            "authorized_representative"
            if requested_field == "authorization_role"
            and re.search(r"\brepresent\w*\b", normalized)
            else "authorized_operator"
            if requested_field == "authorization_role"
            and re.search(r"\boperate\w*\b", normalized)
            else None
        ),
        "document_type": document_type,
        "requested_document_type": document_type,
    }