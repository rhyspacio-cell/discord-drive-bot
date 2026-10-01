"""Candidate evidence validation and the final answer-context firewall."""

import logging
import re
def _coordination_letter_subject_supported(text, subject, subject_type):
    role_labels = PERSON_SUBJECT_ROLE_LABELS | {"contact"}
    if subject_type == "company":
        role_labels |= ORGANIZATION_SUBJECT_ROLE_LABELS
    for line in text.splitlines():
        parts = re.split(r"\s*(?:\||:|\- )\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        label = re.sub(r"[_\s]+", " ", parts[0].casefold().strip())
        if label in role_labels and _contains_phrase(parts[1], subject):
            return True
    subject_pattern = re.escape(subject)
    return bool(
        re.search(
            rf"\b(?:planned\s+)?(?:activity|event|inspection|sampling)\s+for\s+{subject_pattern}\b",
            text,
            re.IGNORECASE,
        )
        or re.search(
            rf"\b{subject_pattern}['’]s\s+(?:planned\s+)?(?:activity|event|inspection|sampling)\b",
            text,
            re.IGNORECASE,
        )
    )

from dataclasses import asdict, dataclass
from enum import Enum

from modules.query_constraints import extract_query_constraints


logger = logging.getLogger(__name__)
_VALIDATION_TOKEN = object()
PERSON_SUBJECT_ROLE_LABELS = {
    "person",
    "requested subject",
    "subject",
    "employee",
    "individual",
    "client",
    "participant",
    "project manager",
    "responsible person",
    "activity owner",
    "assigned to",
    "activity for",
    "performed by",
    "authorized person",
    "authorized operator",
}
ORGANIZATION_SUBJECT_ROLE_LABELS = {"company", "organization"}
NON_SUBJECT_ROLE_LABELS = {
    "representative",
    "company representative",
    "authorized representative",
    "prepared by",
    "document author",
    "author",
    "approver",
    "supervisor",
    "witness",
    "signatory",
    "reviewer",
    "authorized recipient",
    "document recipient",
    "recipient",
    "employer",
    "related entity",
    "mentioned entity",
    "contact",
}


class EvidenceRejectionReason(str, Enum):
    SUBJECT_MISMATCH = "subject_mismatch"
    ACTIVITY_MISMATCH = "activity_mismatch"
    REQUESTED_FIELD_MISSING = "requested_field_missing"
    ORGANIZATION_MISMATCH = "organization_mismatch"
    DOCUMENT_TYPE_MISMATCH = "document_type_mismatch"
    RELATIONSHIP_UNSUPPORTED = "relationship_unsupported"
    FILENAME_ONLY = "filename_only"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    EXTRACTION_FAILURE = "extraction_failure"
    EXTRACTION_EMPTY = "extraction_empty"
    EXTRACTION_NOT_ATTEMPTED = "extraction_not_attempted"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS_EVIDENCE = "ambiguous_evidence"


class EvidenceState(str, Enum):
    NO_CANDIDATES = "no_candidates"
    CANDIDATES_FOUND_BUT_NO_VALID_EVIDENCE = "candidates_found_but_no_valid_evidence"
    VALID_EVIDENCE_FOUND = "valid_evidence_found"
    AMBIGUOUS_EVIDENCE = "ambiguous_evidence"
    EXTRACTION_FAILURE = "extraction_failure"


class ExtractionStatus(str, Enum):
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    SUCCESS = "SUCCESS"
    EMPTY = "EMPTY"
    FAILURE = "FAILURE"
    UNSUPPORTED = "UNSUPPORTED"


class EvidenceValidationError(RuntimeError):
    """Raised when unvalidated evidence reaches answer generation."""


@dataclass(frozen=True)
class EvidenceAssessment:
    file_name: str
    subject_match: bool | None
    activity_match: bool | None
    requested_field_match: bool | None
    organization_match: bool | None
    document_type_match: bool | None
    relationship_supported: bool
    filename_only_match: bool
    evidence_score: float
    eligible: bool
    rejection_reasons: tuple[EvidenceRejectionReason, ...]
    candidate_id: str | None = None
    file_id: str | None = None
    mime_type: str | None = None
    extraction_method: str | None = None
    extraction_status: ExtractionStatus = ExtractionStatus.SUCCESS
    extraction_error: str | None = None
    extracted_char_count: int = 0
    extracted_text_available: bool = False
    retrieved: bool = True
    evidence_span: str | None = None
    validation_reason: str | None = None

    @property
    def rejection_codes(self):
        return tuple(reason.value for reason in self.rejection_reasons)

    def to_dict(self):
        result = asdict(self)
        codes = [reason.value for reason in self.rejection_reasons]
        result["rejection_reasons"] = codes
        result["rejection_codes"] = codes
        result["extraction_status"] = self.extraction_status.value
        return result


class ValidatedEvidence:
    """Opaque evidence payload created only by the validator."""

    __slots__ = ("_documents", "_assessments")

    def __init__(self, documents, assessments, token=None):
        if token is not _VALIDATION_TOKEN:
            raise EvidenceValidationError(
                "ValidatedEvidence must be created by validate_candidates."
            )
        self._documents = tuple(documents)
        self._assessments = dict(assessments)

    @property
    def documents(self):
        return self._documents

    @property
    def assessments(self):
        return dict(self._assessments)


@dataclass(frozen=True)
class CandidateEvidenceResult:
    assessments: dict[str, EvidenceAssessment]
    validated_evidence: ValidatedEvidence
    state: EvidenceState


def _words(value):
    return re.findall(r"[a-z0-9]+", str(value).casefold())


def _contains_phrase(text, value):
    expected = _words(value)
    actual = _words(text)
    if not expected:
        return True
    return any(
        actual[index:index + len(expected)] == expected
        for index in range(max(0, len(actual) - len(expected) + 1))
    )


def _contains_concept(text, value):
    def normalize(token):
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if len(token) > 6 and token.endswith("tion"):
            token = token[:-3]
        return token

    expected = [normalize(token) for token in _words(value)]
    actual = [normalize(token) for token in _words(text)]
    if not expected:
        return True
    window_size = max(6, len(expected) + 2)
    expected_words = set(expected)
    return any(
        expected_words.issubset(actual[index:index + window_size])
        for index in range(len(actual))
    )


def _field_supported(text, requested_field):
    if not requested_field:
        return None
    lowered = text.casefold()
    if requested_field in {"date", "planned_activity_date", "entry_date"}:
        has_date = bool(
            re.search(
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                text,
                re.IGNORECASE,
            )
        )
        context_pattern = (
            r"\b(?:planned|scheduled|plans?\s+to|activity\s+date|date\s*[:\-]|will\s+be|will\s+conduct|will\s+perform|will\s+take\s+place|conduct(?:ing)?|to\s+take\s+place)\b"
            if requested_field == "planned_activity_date"
            else r"\b(?:planned|scheduled|activity\s+date|date\s*[:\-]|will\s+be|will\s+conduct|will\s+perform|will\s+take\s+place|conduct(?:ing)?|to\s+take\s+place|entered|on)\b"
        )
        return has_date and bool(re.search(context_pattern, lowered))
    if requested_field == "risks":
        return bool(
            re.search(
                r"\b(?:identified\s+)?(?:project\s+)?risks?\s*(?:identified\s+)?(?:include|includes|are|is|were|consist\s+of|identified\s+as|[:\-])",
                lowered,
            )
        )
    if requested_field == "salary":
        return bool(
            re.search(
                r"\b(?:salary|compensation|remuneration|annual\s+pay|base\s+pay)\b",
                lowered,
            )
        )
    if requested_field == "activity":
        return bool(
            re.search(
                r"\b(?:will\s+be\s+conducting|conduct(?:ing)?|sampling|inspection|maintenance|training|testing|assessment|survey|activity\s*[:\-])\b",
                lowered,
            )
        )
    if requested_field == "discussion":
        return bool(
            re.search(
                r"\b(?:discuss(?:ed|ion)?|meeting|agenda|minutes|topics?|covered)\b",
                lowered,
            )
        )
    if requested_field == "authorization_role":
        return bool(
            re.search(r"\b(?:authorized|authorization|authorizes?|authorize)\b", lowered)
            and re.search(
                r"\b(?:represent|operate|receive|use|access|perform|bring|carry|enter|take|possess|transport)\b",
                lowered,
            )
        )
    return _contains_phrase(text, requested_field)


def supported_activity_dates(text, constraints):
    """Return date strings locally supported by the requested activity context."""
    activity = constraints.get("activity")
    requested_field = constraints.get("requested_field")
    date_pattern = re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        re.IGNORECASE,
    )
    supported = []
    for match in date_pattern.finditer(text):
        start = text.rfind("\n", 0, match.start()) + 1
        end = text.find("\n", match.end())
        line = text[start:end] if end >= 0 else text[start:]
        if activity and not _contains_concept(line, activity):
            continue
        if requested_field and _field_supported(line, requested_field) is False:
            continue
        supported.append(match.group(0))
    return supported


def _document_type_supported(text, document_type):
    if not document_type:
        return None
    if document_type.casefold() == "coordination letter":
        first_line = next(
            (line.strip() for line in text.splitlines() if line.strip()),
            "",
        )
        if re.search(r"\b(?:meeting\s+minutes|project\s+status\s+report|risk\s+assessment)\b", first_line, re.IGNORECASE):
            return False
        prefix = "\n".join(
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text)[:3]
            if paragraph.strip()
        )[:2000]
        return bool(
            re.search(r"\bcoordination\s+letters?\b", prefix, re.IGNORECASE)
        )
    return _contains_phrase(text, document_type)


def _subject_and_field_are_bound(paragraph, subject, requested_field):
    """Require that the subject and the requested field are tied to the same local claim window."""
    if not subject or not requested_field or requested_field not in {
        "date",
        "planned_activity_date",
        "entry_date",
    }:
        return True

    sentence_segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+|\n+", paragraph)
        if segment.strip()
    ]
    field_pattern = re.compile(
        r"\b(?:activity\s+date|planned\s+date|date|scheduled\s+date|entry\s+date)\b",
        re.IGNORECASE,
    )
    date_pattern = re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        re.IGNORECASE,
    )

    for index, sentence in enumerate(sentence_segments):
        if subject.casefold() not in sentence.casefold():
            continue
        for lookahead in range(index, min(index + 3, len(sentence_segments))):
            candidate = sentence_segments[lookahead]
            if not date_pattern.search(candidate):
                continue
            if field_pattern.search(candidate) is None and "date" not in candidate.casefold():
                continue
            window = " ".join(sentence_segments[index:lookahead + 1])
            other_names = set()
            for value in re.findall(
                r"\b[A-Z][A-Za-z0-9&'-]*(?:\s+[A-Z][A-Za-z0-9&'-]*){0,3}\b",
                window,
            ):
                cleaned = re.sub(r"[.:;]", "", value).strip()
                if not cleaned:
                    continue
                normalized = cleaned.casefold()
                if normalized == subject.casefold():
                    continue
                if normalized in {
                    "person",
                    "company",
                    "project",
                    "activity",
                    "date",
                    "representative",
                    "manager",
                    "letter",
                    "report",
                    "contact",
                }:
                    continue
                if any(month in normalized for month in (
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
                )):
                    continue
                if "date" in normalized or "activity" in normalized:
                    continue
                other_names.add(normalized)
            if not other_names:
                return True
    return False


def _relationship_supported(text, constraints, field_match):
    subject = constraints.get("subject_entity")
    activity = constraints.get("activity")
    requested_field = constraints.get("requested_field")
    if not any((subject, activity, requested_field, constraints.get("document_type"))):
        return True

    paragraphs = [
        paragraph
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    if not paragraphs:
        paragraphs = [text]

    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for sentence in sentences:
            subject_ok = not subject or _subject_supported(
                sentence,
                subject,
                requested_field,
                constraints.get("requested_relationship"),
                constraints.get("subject_type"),
                constraints.get("document_type"),
            )
            activity_ok = not activity or _contains_concept(sentence, activity)
            field_ok = (
                field_match is not False
                and _field_supported(sentence, requested_field) is not False
            )
            if subject_ok and activity_ok and field_ok and len(sentence) <= 1200:
                return True

        structured_subject_labels = PERSON_SUBJECT_ROLE_LABELS | (
            ORGANIZATION_SUBJECT_ROLE_LABELS
            if constraints.get("subject_type") == "company"
            else set()
        )
        if constraints.get("document_type") == "Coordination Letter":
            structured_subject_labels = structured_subject_labels | {"contact"}
        structured_subject_pattern = "|".join(
            re.escape(label)
            for label in sorted(structured_subject_labels, key=len, reverse=True)
        )
        structured_subject = (
            not subject
            or bool(
                re.search(
                    rf"\b(?:{structured_subject_pattern})\s*[:\-]",
                    paragraph,
                    re.IGNORECASE,
                )
                    and _subject_supported(
                        paragraph,
                        subject,
                        requested_field,
                        constraints.get("requested_relationship"),
                        constraints.get("subject_type"),
                        constraints.get("document_type"),
                    )
            )
        )
        structured_activity = (
            not activity
            or (
                bool(re.search(r"\b(?:activity|event|task)\s*[:\-]", paragraph, re.IGNORECASE))
                and _contains_concept(paragraph, activity)
            )
        )
        structured_field = (
            not requested_field
            or (
                requested_field in {"date", "planned_activity_date", "entry_date"}
                and bool(re.search(r"\b(?:activity\s+date|planned\s+date|date)\s*[:\-]", paragraph, re.IGNORECASE))
                and _field_supported(paragraph, requested_field) is True
            )
            or (
                requested_field not in {"date", "planned_activity_date", "entry_date"}
                and _field_supported(paragraph, requested_field) is True
            )
        )
        if subject and requested_field in {"date", "planned_activity_date", "entry_date"}:
            structured_field = structured_field and _subject_and_field_are_bound(
                paragraph,
                subject,
                requested_field,
            )
        if structured_subject and structured_activity and structured_field:
            return True

        if (
            subject
            and subject.casefold() == "project"
            and requested_field == "risks"
            and _subject_supported(paragraph, subject, requested_field)
            and _field_supported(paragraph, requested_field) is True
        ):
            return True

    if (
        constraints.get("document_type") == "Coordination Letter"
        and subject
        and _contains_phrase(text, constraints["document_type"])
        and _coordination_letter_subject_supported(
            text,
            subject,
            constraints.get("subject_type"),
        )
    ):
        return any(
            (not activity or _contains_concept(sentence, activity))
            and _field_supported(sentence, requested_field) is not False
            for paragraph in paragraphs
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
        )

    return False


def _authorization_relationship_supported(text, subject, requested_relationship):
    if not subject:
        return True
    subject_pattern = re.escape(subject)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]
    if requested_relationship == "authorized_representative":
        patterns = (
            rf"\b{subject_pattern}\s+(?:hereby\s+)?authoriz\w*\s+[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){{0,3}}\s+to\s+represent\s+(?:him|her|them)\b",
            rf"\b[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){{0,3}}\s+(?:is|was|has\s+been)\s+authorized\s+to\s+represent\s+{subject_pattern}\b",
            rf"\b(?:person\s+represented|represented\s+person|principal)\s*[:|\-]\s*{subject_pattern}\b",
        )
        for sentence in sentences:
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in patterns):
                return True
        structured_roles = {
            re.sub(r"\s+", " ", parts[0].strip().casefold()): parts[1].strip()
            for line in text.splitlines()
            if len(parts := re.split(r"\s*(?:\||:|\- )\s*", line, maxsplit=1)) == 2
        }
        represented = next(
            (
                value
                for label, value in structured_roles.items()
                if label in {"person represented", "represented person", "principal"}
            ),
            "",
        )
        has_representative = any(
            label in {"authorized representative", "representative"}
            for label in structured_roles
        )
        return bool(_contains_phrase(represented, subject) and has_representative)

    if requested_relationship == "authorized_operator":
        patterns = (
            rf"\b{subject_pattern}\s+(?:(?:is|was|has\s+been)\s+)?(?:hereby\s+|expressly\s+)?authorized\s+to\s+operate\b",
            rf"\bauthoriz\w*\s+{subject_pattern}\s+to\s+operate\b",
            rf"\b(?:authorized\s+operator|authorized\s+person|operator)\s*[:|\-]\s*{subject_pattern}\b",
        )
        for sentence in sentences:
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in patterns):
                return True
        return False

    return any(
        _contains_phrase(sentence, subject)
        and re.search(r"\bauthoriz\w*\b", sentence, re.IGNORECASE)
        for sentence in sentences
    )


def _subject_supported(
    text,
    subject,
    requested_field,
    requested_relationship=None,
    subject_type=None,
    document_type=None,
):
    subject_present = _contains_phrase(text, subject)
    if subject.casefold() == "project":
        subject_present = subject_present or bool(
            re.search(r"\bprojects\b", text, re.IGNORECASE)
        )
    if not subject_present:
        return False
    if subject.casefold() == "project":
        if requested_field == "risks":
            return bool(re.search(r"\brisks?\b", text, re.IGNORECASE))
        if requested_field == "discussion":
            return bool(re.search(r"\b(?:meeting|discuss|agenda|minutes|topics?)\b", text, re.IGNORECASE))
        return True
    if requested_field == "authorization_role":
        return _authorization_relationship_supported(
            text,
            subject,
            requested_relationship,
        )
    coordination_contact = document_type == "Coordination Letter"
    subject_roles = PERSON_SUBJECT_ROLE_LABELS | (
        ORGANIZATION_SUBJECT_ROLE_LABELS if subject_type == "company" else set()
    )
    if coordination_contact:
        subject_roles = subject_roles | {"contact"}
    other_roles = NON_SUBJECT_ROLE_LABELS | (
        ORGANIZATION_SUBJECT_ROLE_LABELS if subject_type != "company" else set()
    )
    if coordination_contact:
        other_roles = other_roles - {"contact"}
    role_values = {"subject": [], "other": []}
    for line in text.splitlines():
        parts = re.split(r"\s*(?:\||:)\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        label = re.sub(r"[_\s]+", " ", parts[0].strip().casefold())
        if label in subject_roles:
            role_values["subject"].append(parts[1].strip())
        elif label in other_roles:
            role_values["other"].append(parts[1].strip())

    subject_is_bound = any(
        _contains_phrase(value, subject)
        for value in role_values["subject"]
    )
    subject_is_other_role = any(
        _contains_phrase(value, subject)
        for value in role_values["other"]
    )
    subject_role_labels = subject_roles
    other_role_labels = other_roles
    subject_field_match = any(
        re.search(
            rf"\b{re.escape(label)}\s*[:\-]\s*[^.;\n]{{0,100}}\b{re.escape(subject)}\b",
            text,
            re.IGNORECASE,
        )
        for label in subject_role_labels
    )
    other_field_match = any(
        re.search(
            rf"\b{re.escape(label)}\s*[:\-]\s*[^.;\n]{{0,100}}\b{re.escape(subject)}\b",
            text,
            re.IGNORECASE,
        )
        for label in other_role_labels
    )
    subject_is_bound = subject_is_bound or subject_field_match
    subject_is_other_role = subject_is_other_role or other_field_match
    if subject_is_other_role and role_values["subject"] and not subject_is_bound:
        return False
    if subject_is_other_role and not subject_is_bound:
        return False
    if role_values["subject"] and not subject_is_bound:
        return False
    return True


def _generic_context_match(question, text, search_plan):
    values = []
    if isinstance(search_plan, dict):
        for key in ("required_terms", "phrases", "optional_terms", "context_terms"):
            items = search_plan.get(key, [])
            if isinstance(items, list):
                values.extend(str(item) for item in items)
    if not values:
        stop_words = {
            "a", "an", "and", "are", "for", "from", "how", "in", "is", "it",
            "me", "of", "on", "the", "to", "was", "what", "when", "where",
            "which", "who", "with", "my", "our", "their", "this", "that",
        }
        values = [word for word in _words(question) if word not in stop_words]
    return any(_contains_concept(text, value) for value in values if _words(value))


def _validated_evidence_span(question, text, constraints, search_plan):
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ] or [text.strip()]
    has_structured_constraints = any(
        constraints.get(key)
        for key in ("subject_entity", "activity", "requested_field", "document_type")
    )

    for paragraph in paragraphs:
        if has_structured_constraints:
            field_match = _field_supported(paragraph, constraints.get("requested_field"))
            if _relationship_supported(paragraph, constraints, field_match):
                return paragraph
            continue

        for sentence in re.split(r"(?<=[.!?])\s+|\n+", paragraph):
            if sentence.strip() and _generic_context_match(question, sentence, search_plan):
                return sentence.strip()
    return None


def assess_candidate(question, candidate, search_plan=None):
    """Assess one extracted candidate against the current query constraints."""
    constraints = (
        search_plan.get("query_constraints")
        if isinstance(search_plan, dict)
        else None
    ) or extract_query_constraints(question)
    file_name = str(candidate.get("name", "Unnamed file"))
    text = candidate.get("text", "")
    text = text if isinstance(text, str) else ""
    extraction_status = candidate.get("extraction_status")
    if not isinstance(extraction_status, ExtractionStatus):
        try:
            extraction_status = ExtractionStatus(str(extraction_status).upper())
        except ValueError:
            extraction_status = (
                ExtractionStatus.SUCCESS
                if text.strip()
                else ExtractionStatus.EMPTY
            )
    filename = file_name.casefold()

    subject = constraints.get("subject_entity")
    activity = constraints.get("activity")
    requested_field = constraints.get("requested_field")
    document_type = constraints.get("document_type")
    subject_match = (
        None
        if not subject
        else _subject_supported(
            text,
            subject,
            requested_field,
            constraints.get("requested_relationship"),
            constraints.get("subject_type"),
            constraints.get("document_type"),
        )
    )
    activity_match = None if not activity else _contains_concept(text, activity)
    organization_match = (
        subject_match
        if constraints.get("subject_type") == "company"
        else None
    )
    requested_field_match = _field_supported(text, requested_field)
    document_type_match = (
        None
        if not document_type
        else _document_type_supported(text, document_type)
    )

    relationship_supported = _relationship_supported(
        text,
        constraints,
        requested_field_match,
    )
    constraint_terms = [subject, activity, document_type]
    filename_match = any(
        value and all(word in filename for word in _words(value))
        for value in constraint_terms
    )
    content_match = any(
        value and _contains_concept(text, value)
        for value in constraint_terms
    )
    filename_only_match = bool(
        filename_match
        and (
            (not content_match and document_type_match is not True)
            or not relationship_supported
            or document_type_match is False
        )
    )

    reasons = []
    if extraction_status is ExtractionStatus.FAILURE:
        reasons.append(EvidenceRejectionReason.EXTRACTION_FAILURE)
    elif extraction_status is ExtractionStatus.EMPTY:
        reasons.append(EvidenceRejectionReason.EXTRACTION_EMPTY)
    elif extraction_status is ExtractionStatus.UNSUPPORTED:
        reasons.append(EvidenceRejectionReason.UNSUPPORTED)
    elif extraction_status is ExtractionStatus.NOT_ATTEMPTED:
        reasons.append(EvidenceRejectionReason.EXTRACTION_NOT_ATTEMPTED)
    if subject_match is False:
        reasons.append(EvidenceRejectionReason.SUBJECT_MISMATCH)
    if activity_match is False:
        reasons.append(EvidenceRejectionReason.ACTIVITY_MISMATCH)
    if requested_field_match is False:
        reasons.append(EvidenceRejectionReason.REQUESTED_FIELD_MISSING)
    if organization_match is False:
        reasons.append(EvidenceRejectionReason.ORGANIZATION_MISMATCH)
    if document_type_match is False:
        reasons.append(EvidenceRejectionReason.DOCUMENT_TYPE_MISMATCH)
    if filename_only_match:
        reasons.append(EvidenceRejectionReason.FILENAME_ONLY)
    if not relationship_supported:
        reasons.append(EvidenceRejectionReason.RELATIONSHIP_UNSUPPORTED)

    has_structured_constraints = any(
        (subject, activity, requested_field, document_type)
    )
    generic_match = _generic_context_match(question, text, search_plan)
    if not has_structured_constraints and not generic_match:
        reasons.append(EvidenceRejectionReason.INSUFFICIENT_CONTEXT)

    explicit_negative_evidence = bool(
        requested_field
        and re.search(
            r"\b(?:no|not|never|without)\b.{0,80}\b(?:planned|scheduled|activity\s+date|date|record|company)\b",
            text,
            re.IGNORECASE,
        )
        and document_type_match is True
    )
    if explicit_negative_evidence:
        reasons = [
            reason
            for reason in reasons
            if reason not in {
                EvidenceRejectionReason.REQUESTED_FIELD_MISSING,
                EvidenceRejectionReason.RELATIONSHIP_UNSUPPORTED,
                EvidenceRejectionReason.FILENAME_ONLY,
            }
        ]
        relationship_supported = True
        requested_field_match = True
        if document_type_match is True:
            filename_only_match = False

    evidence_span = (
        _validated_evidence_span(question, text, constraints, search_plan)
        if not reasons and text.strip()
        else None
    )
    if not evidence_span and not reasons:
        reasons.append(EvidenceRejectionReason.INSUFFICIENT_CONTEXT)
    eligible = not reasons and bool(evidence_span)
    if not text.strip() and EvidenceRejectionReason.EXTRACTION_EMPTY not in reasons:
        reasons.append(EvidenceRejectionReason.EXTRACTION_FAILURE)

    checks = [
        value
        for value in (
            subject_match,
            activity_match,
            requested_field_match,
            organization_match,
            document_type_match,
        )
        if value is not None
    ]
    evidence_score = (
        round((sum(checks) / len(checks)) * 0.6 + (0.4 if relationship_supported else 0), 2)
        if checks
        else (0.75 if generic_match else 0.0)
    )
    if filename_only_match:
        evidence_score = min(evidence_score, 0.1)

    return EvidenceAssessment(
        file_name=file_name,
        subject_match=subject_match,
        activity_match=activity_match,
        requested_field_match=requested_field_match,
        organization_match=organization_match,
        document_type_match=document_type_match,
        relationship_supported=relationship_supported,
        filename_only_match=filename_only_match,
        evidence_score=evidence_score,
        eligible=eligible,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        candidate_id=candidate.get("candidate_id"),
        file_id=candidate.get("file_id"),
        mime_type=candidate.get("mime_type"),
        extraction_method=candidate.get("extraction_method"),
        extraction_status=extraction_status,
        extraction_error=candidate.get("extraction_error"),
        extracted_char_count=len(text),
        extracted_text_available=bool(text.strip()),
        evidence_span=evidence_span,
        validation_reason=(
            "validated claim-supporting span" if eligible else "; ".join(
                reason.value for reason in dict.fromkeys(reasons)
            )
        ),
    )


def validate_candidates(question, candidates, search_plan=None, search_audit=None):
    """Assess every extracted candidate and mint the final validated payload."""
    candidates = list(candidates or [])
    assessments = {}
    validated_documents = []

    name_counts = {}
    for candidate in candidates:
        assessment = assess_candidate(question, candidate, search_plan)
        base_key = assessment.file_id or assessment.candidate_id or assessment.file_name
        key = base_key
        if key in assessments:
            name_counts[assessment.file_name] = name_counts.get(assessment.file_name, 1) + 1
            key = f"{assessment.file_name}#{name_counts[assessment.file_name]}"
        assessments[key] = assessment
        if assessment.eligible:
            validated_document = dict(candidate)
            validated_document["_evidence_assessment"] = assessment
            validated_documents.append(validated_document)

        logger.debug(
            "Candidate evidence assessment: %s",
            assessment.to_dict(),
        )

    if search_audit is not None:
        existing_ids = {
            assessment.file_id or assessment.candidate_id
            for assessment in assessments.values()
        }
        for diagnostic in search_audit.get("candidate_diagnostics", []):
            candidate_id = diagnostic.get("candidate_id")
            if candidate_id in existing_ids:
                continue
            try:
                extraction_status = ExtractionStatus(
                    diagnostic.get(
                        "extraction_status",
                        ExtractionStatus.NOT_ATTEMPTED.value,
                    )
                )
            except ValueError:
                extraction_status = ExtractionStatus.FAILURE
            if extraction_status is ExtractionStatus.SUCCESS:
                continue
            reason_code = {
                ExtractionStatus.EMPTY: EvidenceRejectionReason.EXTRACTION_EMPTY,
                ExtractionStatus.NOT_ATTEMPTED: EvidenceRejectionReason.EXTRACTION_NOT_ATTEMPTED,
                ExtractionStatus.UNSUPPORTED: EvidenceRejectionReason.UNSUPPORTED,
            }.get(extraction_status, EvidenceRejectionReason.EXTRACTION_FAILURE)
            assessment = EvidenceAssessment(
                file_name=diagnostic.get("file_name", "Unnamed file"),
                subject_match=None,
                activity_match=None,
                requested_field_match=None,
                organization_match=None,
                document_type_match=None,
                relationship_supported=False,
                filename_only_match=False,
                evidence_score=0.0,
                eligible=False,
                rejection_reasons=(reason_code,),
                candidate_id=candidate_id,
                file_id=diagnostic.get("file_id"),
                mime_type=diagnostic.get("mime_type"),
                extraction_method=diagnostic.get("extraction_method"),
                extraction_status=extraction_status,
                extraction_error=diagnostic.get("extraction_error"),
                extracted_char_count=diagnostic.get("extracted_char_count", 0),
                extracted_text_available=diagnostic.get("extracted_text_available", False),
            )
            assessments[candidate_id or assessment.file_name] = assessment
            logger.debug("Candidate evidence assessment: %s", assessment.to_dict())

    evidence = ValidatedEvidence(
        validated_documents,
        assessments,
        _VALIDATION_TOKEN,
    )
    candidate_diagnostics = (
        search_audit.get("candidate_diagnostics", [])
        if search_audit is not None
        else []
    )
    has_retrieved_candidates = bool(candidates or candidate_diagnostics)
    has_extracted_text = any(
        diagnostic.get("extraction_status") == ExtractionStatus.SUCCESS.value
        and diagnostic.get("extracted_text_available")
        for diagnostic in candidate_diagnostics
    )
    has_extraction_errors = bool(
        search_audit is not None
        and any(search_audit.get(key) for key in ("failed", "skipped", "empty"))
    )
    state = (
        EvidenceState.VALID_EVIDENCE_FOUND
        if validated_documents
        else EvidenceState.CANDIDATES_FOUND_BUT_NO_VALID_EVIDENCE
        if candidates or has_extracted_text
        else EvidenceState.EXTRACTION_FAILURE
        if has_extraction_errors
        else EvidenceState.CANDIDATES_FOUND_BUT_NO_VALID_EVIDENCE
        if has_retrieved_candidates
        else EvidenceState.NO_CANDIDATES
    )

    if search_audit is not None:
        assessment_map = search_audit.setdefault("assessments", {})
        assessment_records = search_audit.setdefault("assessment_records", [])
        assessment_map.clear()
        assessment_by_id = {
            assessment.file_id or assessment.candidate_id: assessment
            for assessment in assessments.values()
        }
        assessment_records.clear()
        diagnostic_ids = {
            diagnostic.get("candidate_id")
            for diagnostic in search_audit.get("candidate_diagnostics", [])
        }
        for diagnostic in search_audit.get("candidate_diagnostics", []):
            candidate_id = diagnostic.get("candidate_id")
            assessment = assessment_by_id.get(candidate_id)
            if assessment is not None:
                assessment_data = assessment.to_dict()
                diagnostic["evidence_assessment"] = assessment_data
                diagnostic["eligible"] = assessment.eligible
                diagnostic["rejection_reasons"] = [
                    reason.value for reason in assessment.rejection_reasons
                ]
                diagnostic["reference"] = False
                diagnostic["reference_status"] = (
                    "CLAIM_SUPPORTING" if assessment.eligible else "REJECTED"
                )
                diagnostic["reference_reason"] = (
                    "NO_FINAL_CLAIM_PROVENANCE"
                    if assessment.eligible
                    else ",".join(assessment.rejection_codes)
                )
            else:
                assessment_data = diagnostic.get("evidence_assessment")
                diagnostic["reference"] = False
                diagnostic["reference_status"] = (
                    "ANALYZED_ONLY"
                    if diagnostic.get("extraction_status") == ExtractionStatus.NOT_ATTEMPTED.value
                    else "EXTRACTED_ONLY"
                    if diagnostic.get("extraction_status") == ExtractionStatus.SUCCESS.value
                    else "REJECTED"
                )
                diagnostic["reference_reason"] = "NO_VALIDATED_CLAIM_SUPPORT"

            if assessment_data is not None:
                assessment_records.append(assessment_data)
                output_key = diagnostic.get("file_name", "Unnamed file")
                if output_key in assessment_map:
                    output_key = candidate_id or output_key
                assessment_map[output_key] = assessment_data

        for key, assessment in assessments.items():
            candidate_id = assessment.file_id or assessment.candidate_id
            if candidate_id and candidate_id in diagnostic_ids:
                continue
            assessment_data = assessment.to_dict()
            assessment_records.append(assessment_data)
            output_key = assessment.file_name
            if output_key in assessment_map:
                output_key = key
            assessment_map[output_key] = assessment_data
        search_audit["state"] = state.value

    return CandidateEvidenceResult(assessments, evidence, state)


def validate_answer_evidence(evidence):
    """Fail closed if anything except validator-minted eligible evidence arrives."""
    if not isinstance(evidence, ValidatedEvidence):
        raise EvidenceValidationError(
            "Answer generation requires a ValidatedEvidence payload."
        )
    for document in evidence.documents:
        assessment = document.get("_evidence_assessment")
        if not isinstance(assessment, EvidenceAssessment) or not assessment.eligible:
            raise EvidenceValidationError(
                "Unvalidated evidence reached answer generation."
            )
        span = assessment.evidence_span
        text = document.get("text")
        if not isinstance(span, str) or not span.strip() or not isinstance(text, str) or span not in text:
            raise EvidenceValidationError(
                "Validated evidence is missing an authentic source span."
            )
    return evidence.documents