import io
import logging
import random
import re
import time
from datetime import date, timedelta

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader
from docx import Document

from modules.config import (
    MAX_CHARS_PER_FILE,
    MAX_DOWNLOAD_BYTES,
    MAX_FILES,
    MAX_TOTAL_CHARS,
)
from modules.storage import load_credentials, save_credentials
from modules.query_constraints import extract_query_constraints
from modules.evidence import (
    EvidenceState,
    ExtractionStatus,
    validate_candidates,
)


logger = logging.getLogger(__name__)


class DriveError(RuntimeError):
    """Raised when Google Drive access or file extraction fails."""
    pass


class DriveFileTooLargeError(DriveError):
    """Raised when a file exceeds the configured download limit."""
    pass


class DriveFileUnavailableError(DriveError):
    """Raised when a Drive file cannot currently be accessed."""
    pass


def execute_drive_request(
    request,
    operation_name,
    max_attempts=3,
):
    """
    Execute a Google Drive API request with retries for transient errors.

    Permanent errors such as 403/404 are not retried.
    Transient rate-limit and server errors are retried with
    truncated exponential backoff.
    """

    for attempt in range(max_attempts):
        try:
            return request.execute()

        except HttpError as exc:
            status_code = getattr(
                exc.resp,
                "status",
                None,
            )

            retryable = status_code in {
                429,
                500,
                502,
                503,
                504,
            }

            if not retryable:
                raise

            if attempt == max_attempts - 1:
                raise

            delay = min(
                (2 ** attempt) + random.uniform(0, 1),
                8,
            )

            print(
                "[DRIVE RETRY]",
                {
                    "operation": operation_name,
                    "status": status_code,
                    "attempt": attempt + 1,
                    "next_delay": round(delay, 2),
                },
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Drive request failed: {operation_name}"
    )


def get_drive_service(discord_user_id: int):
    """Create a Google Drive API client for one Discord user."""
    credentials = load_credentials(discord_user_id)

    if not credentials:
        raise DriveError(
            "Google Drive could not be accessed. "
            "Make sure your Drive is connected and "
            "that the Google authorization is still valid."
        )

    if credentials.expired and credentials.refresh_token:
        from google.auth.transport.requests import Request

        credentials.refresh(Request())
        save_credentials(discord_user_id, credentials)

    return build(
        "drive",
        "v3",
        credentials=credentials,
    )


def download_drive_file(service, file_id: str):
    """Download a normal Drive file with retries for transient errors."""

    request_ = service.files().get_media(
        fileId=file_id
    )

    buffer = io.BytesIO()

    downloader = MediaIoBaseDownload(
        buffer,
        request_,
    )

    finished = False

    while not finished:
        try:
            _, finished = downloader.next_chunk()

        except HttpError as exc:
            status_code = getattr(
                exc.resp,
                "status",
                None,
            )

            if status_code not in {
                429,
                500,
                502,
                503,
                504,
            }:
                raise

            retry_succeeded = False

            for attempt in range(3):
                delay = min(
                    (2 ** attempt) + random.uniform(0, 1),
                    8,
                )

                print(
                    "[DRIVE DOWNLOAD RETRY]",
                    {
                        "status": status_code,
                        "attempt": attempt + 1,
                        "next_delay": round(delay, 2),
                    },
                )

                time.sleep(delay)

                try:
                    _, finished = (
                        downloader.next_chunk()
                    )

                    retry_succeeded = True
                    break

                except HttpError as retry_exc:
                    retry_status = getattr(
                        retry_exc.resp,
                        "status",
                        None,
                    )

                    if retry_status not in {
                        429,
                        500,
                        502,
                        503,
                        504,
                    }:
                        raise

                    if attempt == 2:
                        raise

            if not retry_succeeded:
                raise

        if buffer.tell() > MAX_DOWNLOAD_BYTES:
            raise DriveFileTooLargeError(
                f"File exceeds the {MAX_DOWNLOAD_BYTES} byte download limit."
            )

    return buffer.getvalue()


def resolve_drive_shortcut(service, file_info):
    """Resolve a Google Drive shortcut to its target file."""

    if file_info.get(
        "mimeType"
    ) != "application/vnd.google-apps.shortcut":
        return file_info

    shortcut_details = file_info.get(
        "shortcutDetails",
        {},
    )

    target_id = shortcut_details.get(
        "targetId"
    )

    if not target_id:
        raise DriveFileUnavailableError(
            f"Drive shortcut '{file_info.get('name', 'Unnamed file')}' "
            "does not contain a target file."
        )

    try:
        target = execute_drive_request(
            service.files()
            .get(
                fileId=target_id,
                fields=(
                    "id,"
                    "name,"
                    "mimeType,"
                    "size,"
                    "modifiedTime,"
                    "shortcutDetails"
                ),
            ),
            f"resolve shortcut {file_info.get('name', 'Unnamed file')}",
        )

    except HttpError as exc:
        status_code = getattr(
            exc.resp,
            "status",
            None,
        )

        if status_code == 404:
            raise DriveFileUnavailableError(
                f"Shortcut target for "
                f"'{file_info.get('name', 'Unnamed file')}' "
                "is unavailable or the current account no longer "
                "has access to it."
            ) from exc

        if status_code == 403:
            raise DriveFileUnavailableError(
                f"The current account does not have access to the "
                f"target of shortcut "
                f"'{file_info.get('name', 'Unnamed file')}'."
            ) from exc

        raise

    print(
        "[SHORTCUT RESOLVED]",
        {
            "shortcut": file_info.get("name"),
            "target": target.get("name"),
            "targetMimeType": target.get("mimeType"),
        },
    )

    return target


def extract_file_text(service, file_info):
    """Extract text from a supported Drive file."""

    original_filename = file_info.get(
        "name",
        "Unnamed file",
    )

    file_info = resolve_drive_shortcut(
        service,
        file_info,
    )

    file_id = file_info["id"]

    mime_type = file_info.get(
        "mimeType",
        "",
    )

    filename = original_filename

    print(
        "[FILE METADATA]",
        {
            "name": filename,
            "id": file_id,
            "mimeType": mime_type,
            "shortcutDetails": file_info.get(
                "shortcutDetails"
            ),
        },
    )

    # Google Docs
    if mime_type == "application/vnd.google-apps.document":
        data = execute_drive_request(
            service.files().export(
                fileId=file_id,
                mimeType="text/plain",
            ),
            f"export {filename}",
        )

        return filename, data.decode(
            "utf-8",
            errors="replace",
        )

    # Google Slides
    if mime_type == "application/vnd.google-apps.presentation":
        data = execute_drive_request(
            service.files().export(
                fileId=file_id,
                mimeType="text/plain",
            ),
            f"export {filename}",
        )

        return filename, data.decode(
            "utf-8",
            errors="replace",
        )

    # Google Sheets
    if mime_type == "application/vnd.google-apps.spreadsheet":
        data = execute_drive_request(
            service.files().export(
                fileId=file_id,
                mimeType="text/csv",
            ),
            f"export {filename}",
        )

        return filename, data.decode(
            "utf-8",
            errors="replace",
        )

    is_pdf = (
        mime_type == "application/pdf"
        or filename.lower().endswith(".pdf")
    )

    is_docx = (
        mime_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or filename.lower().endswith(".docx")
    )

    text_extensions = (
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".xml",
        ".html",
        ".py",
        ".js",
        ".ts",
        ".css",
        ".sql",
    )

    is_text = (
        mime_type.startswith("text/")
        or filename.lower().endswith(text_extensions)
    )

    if not is_pdf and not is_docx and not is_text:
        return filename, ""

    raw = download_drive_file(
        service,
        file_id,
    )

    # PDF
    if is_pdf:
        reader = PdfReader(
            io.BytesIO(raw)
        )

        extracted = []
        character_count = 0

        for page in reader.pages:
            page_text = page.extract_text() or ""

            remaining = (
                MAX_CHARS_PER_FILE
                - character_count
            )

            if remaining <= 0:
                break

            extracted_text = page_text[:remaining]

            extracted.append(
                extracted_text
            )

            character_count += len(
                extracted_text
            )

        return filename, "\n".join(
            extracted
        )

    # DOCX
    if is_docx:
        document = Document(
            io.BytesIO(raw)
        )

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]
        paragraphs.extend(_extract_docx_table_text(table) for table in document.tables)
        paragraphs.extend(
            paragraph.text
            for section in document.sections
            for paragraph in (
                list(section.header.paragraphs)
                + list(section.footer.paragraphs)
            )
            if paragraph.text.strip()
        )

        return filename, "\n".join(
            paragraphs
        )

    # Normal text files
    if is_text:
        return filename, raw.decode(
            "utf-8",
            errors="replace",
        )

    return filename, ""


def _extract_docx_table_text(table):
    """Extract table rows, including nested tables, without dropping form fields."""
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            parts = [
                paragraph.text.strip()
                for paragraph in cell.paragraphs
                if paragraph.text.strip()
            ]
            parts.extend(
                _extract_docx_table_text(nested_table)
                for nested_table in cell.tables
            )
            cells.append(" ".join(parts))
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extraction_method(file_info):
    mime_type = file_info.get("mimeType", "")
    filename = str(file_info.get("name", "")).lower()
    if mime_type in {
        "application/vnd.google-apps.document",
        "application/vnd.google-apps.presentation",
        "application/vnd.google-apps.spreadsheet",
    }:
        return "google_drive_export"
    if mime_type == "application/pdf" or filename.endswith(".pdf"):
        return "pdf_text"
    if (
        mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or filename.endswith(".docx")
    ):
        return "docx_paragraphs_tables_headers_footers"
    if mime_type.startswith("text/") or filename.endswith(
        (".txt", ".md", ".csv", ".json", ".xml", ".html", ".py", ".js", ".ts", ".css", ".sql")
    ):
        return "text_decode"
    return None


def is_google_export(file_info):
    """Return whether Drive handles the file through a native export."""

    mime_type = file_info.get(
        "mimeType",
        "",
    )

    if mime_type == "application/vnd.google-apps.shortcut":
        target_mime_type = (
            file_info
            .get("shortcutDetails", {})
            .get("targetMimeType", "")
        )

        return target_mime_type.startswith(
            "application/vnd.google-apps."
        )

    return mime_type in {
        "application/vnd.google-apps.document",
        "application/vnd.google-apps.presentation",
        "application/vnd.google-apps.spreadsheet",
    }


def is_extractable_file(file_info):
    """Return whether the file type is supported by the text extractor."""

    mime_type = file_info.get(
        "mimeType",
        "",
    )

    filename = file_info.get(
        "name",
        "",
    ).lower()

    if mime_type in {
        "application/vnd.google-apps.document",
        "application/vnd.google-apps.presentation",
        "application/vnd.google-apps.spreadsheet",
    }:
        return True

    if mime_type == "application/pdf":
        return True

    if (
        mime_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        return True

    text_extensions = (
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".xml",
        ".html",
        ".py",
        ".js",
        ".ts",
        ".css",
        ".sql",
    )

    return (
        mime_type.startswith("text/")
        or filename.endswith(text_extensions)
    )


SEARCH_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "is",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def get_search_terms(search_query: str):
    """Extract meaningful keyword terms from a natural-language search."""

    terms = [
        term
        for term in re.findall(
            r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",
            search_query.lower(),
        )
        if term not in SEARCH_STOP_WORDS
    ]

    if not terms:
        raise DriveError(
            "Enter meaningful search terms to find Drive files."
        )

    return terms


def is_list_aggregation_query(search_plan):
    """Return True only for explicit-date list/aggregation retrieval cases."""
    if not isinstance(search_plan, dict):
        return False

    time_range = search_plan.get("time_range") or {}
    has_explicit_range = bool(
        time_range.get("from") is not None
        or time_range.get("to") is not None
    )

    if not has_explicit_range:
        return False

    answer_type = str(search_plan.get("answer_type", "")).strip().lower()
    intent = str(search_plan.get("intent", "")).strip().lower()

    list_answer_types = {
        "list",
        "records",
        "summary",
        "table",
        "aggregation",
        "overview",
        "timeline",
        "dates",
        "report",
        "coordinate letters",
        "coordination letters",
        "letters",
    }

    if answer_type in list_answer_types:
        return True

    plan_values = []
    for key in (
        "required_terms",
        "phrases",
        "optional_terms",
        "context_terms",
    ):
        values = search_plan.get(key, [])
        if isinstance(values, list):
            plan_values.extend(str(value) for value in values)

    all_words = set(
        re.findall(
            r"[a-z0-9]+",
            " ".join([intent, answer_type, *plan_values]),
        )
    )

    aggregation_indicators = {
        "list",
        "listing",
        "records",
        "entries",
        "dates",
        "timeline",
        "summary",
        "overview",
        "companies",
        "events",
        "all",
    }

    # Explicit date-range aggregation requests are multi-record questions.
    # They are typically identifiable by plural/aggregate terms such as
    # "companies" or "dates" even when the answer type is a narrow document
    # category like "coordinate letters" rather than a literal "summary".
    list_hits = aggregation_indicators & all_words

    if not list_hits:
        return False

    if answer_type in {"document", "biography", "person", "company", "employee"}:
        # Keep focused single-entity questions on the normal path even if the
        # question contains a date range; they are not aggregation requests.
        if list_hits & {"companies", "records", "entries", "dates", "timeline", "summary", "overview", "events", "all", "list", "listing"}:
            return True
        return False

    return True


def get_drive_candidate_limit():
    """Return the existing bound for Drive metadata candidates."""
    return max(
        MAX_FILES * 3,
        MAX_FILES,
    )


def get_document_limit_for_query(search_plan=None):
    """Return the safe document cap for the current question type."""
    if is_list_aggregation_query(search_plan):
        return get_drive_candidate_limit()
    return MAX_FILES


def select_extraction_candidates(metadata_ranked, search_plan=None):
    """Widen candidate selection for explicit-date list queries without removing MAX_FILES limits."""
    if not metadata_ranked:
        return []

    query_constraints = (
        search_plan.get("query_constraints", {})
        if isinstance(search_plan, dict)
        else {}
    )
    has_semantic_constraints = bool(
        query_constraints.get("subject_entity")
        or query_constraints.get("activity")
    )
    limit = (
        get_drive_candidate_limit()
        if has_semantic_constraints and not is_list_aggregation_query(search_plan)
        else get_document_limit_for_query(search_plan)
    )
    candidate_limit = min(len(metadata_ranked), limit)

    return [
        file_info
        for _, file_info in metadata_ranked[:candidate_limit]
    ]


def build_drive_search_query(
    question: str,
    search_plan=None,
):
    """
    Build a broad Google Drive candidate query.

    Google Drive performs broad candidate retrieval using OR.

    Python performs the more precise relevance ranking afterward.
    """

    def validate_time_range(time_range):
        """Validate and return a normalized time range for Drive filtering."""
        if time_range is None:
            return None

        if not isinstance(time_range, dict):
            raise DriveError(
                "The search plan contains an invalid time range."
            )

        start_raw = time_range.get("from")
        end_raw = time_range.get("to")

        if start_raw is not None and not isinstance(start_raw, str):
            raise DriveError(
                "The search plan contains an invalid time range."
            )

        if end_raw is not None and not isinstance(end_raw, str):
            raise DriveError(
                "The search plan contains an invalid time range."
            )

        start = None
        end = None

        if start_raw:
            try:
                start = date.fromisoformat(start_raw)
            except ValueError as exc:
                raise DriveError(
                    "The search plan contains an invalid time range."
                ) from exc

        if end_raw:
            try:
                end = date.fromisoformat(end_raw)
            except ValueError as exc:
                raise DriveError(
                    "The search plan contains an invalid time range."
                ) from exc

        if start is not None and end is not None and start > end:
            raise DriveError(
                "The search plan contains an invalid time range."
            )

        return {
            "from": start,
            "to": end,
        }

    if search_plan is None:
        terms = get_search_terms(question)
        excludes = []
        time_range = None

    else:
        required_terms = search_plan.get(
            "required_terms",
            [],
        )

        phrases = search_plan.get(
            "phrases",
            [],
        )

        optional_terms = search_plan.get(
            "optional_terms",
            [],
        )

        excludes = search_plan.get(
            "exclude_terms",
            [],
        )

        time_range = validate_time_range(
            search_plan.get("time_range")
        )

        document_dates = [
            str(value).strip()
            for value in search_plan.get("document_dates", [])
            if isinstance(value, str) and value.strip()
        ]
        activity_dates = [
            str(value).strip()
            for value in search_plan.get("activity_dates", [])
            if isinstance(value, str) and value.strip()
        ]

        all_positive_values = (
            required_terms
            + phrases
            + optional_terms
        )

        query_constraints = search_plan.get("query_constraints") or {}
        subject_entity = query_constraints.get("subject_entity")
        activity = query_constraints.get("activity")
        if isinstance(subject_entity, str) and subject_entity.strip():
            all_positive_values.append(subject_entity.strip())
        if isinstance(activity, str) and activity.strip():
            all_positive_values.append(activity.strip())

        if not all(
            isinstance(term, str)
            and term.strip()
            for term in all_positive_values
        ):
            raise DriveError(
                "The search plan contains invalid terms."
            )

        if not all(
            isinstance(term, str)
            and term.strip()
            for term in excludes
        ):
            raise DriveError(
                "The search plan contains invalid terms."
            )

        required_terms = [
            term.strip().lower()
            for term in required_terms
        ]

        phrases = [
            phrase.strip().lower()
            for phrase in phrases
        ]

        optional_terms = [
            term.strip().lower()
            for term in optional_terms
        ]

        excludes = [
            term.strip().lower()
            for term in excludes
        ]

        terms = list(
            required_terms
        )

        if isinstance(subject_entity, str) and subject_entity.strip():
            terms.append(subject_entity.strip().lower())
        if isinstance(activity, str) and activity.strip():
            terms.extend(
                re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", activity.lower())
            )

        # Break phrases into searchable words.
        for phrase in phrases:
            phrase_terms = [
                term
                for term in re.findall(
                    r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",
                    phrase,
                )
                if term.lower() not in {"that", "entered", "date", "dates", "list", "lists"}
            ]

            terms.extend(
                phrase_terms
            )

        terms.extend(
            optional_terms
        )

        # Remove duplicates while preserving order.
        for date_value in document_dates + activity_dates:
            date_terms = [
                term
                for term in re.findall(
                    r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",
                    date_value,
                )
                if term
            ]
            terms.extend(date_terms)

        terms = list(
            dict.fromkeys(
                term
                for term in terms
                if term
            )
        )

        if not terms:
            terms = get_search_terms(
                question
            )

    def escape(term):
        """Escape a value for use inside a Google Drive query."""

        return (
            term
            .replace("\\", "\\\\")
            .replace("'", "\\'")
        )

    positive_clauses = [
        (
            f"(name contains '{escape(term)}' "
            f"or fullText contains '{escape(term)}')"
        )
        for term in terms
    ]

    exclude_clauses = [
        (
            f"not name contains '{escape(term)}' "
            f"and not fullText contains '{escape(term)}'"
        )
        for term in excludes
    ]

    if not positive_clauses:
        raise DriveError(
            "Enter meaningful search terms to find Drive files."
        )

    base = (
        "trashed = false "
        "and mimeType != "
        "'application/vnd.google-apps.folder'"
    )

    positive_query = " or ".join(
        positive_clauses
    )

    query_parts = [
        base,
    ]

    if time_range and time_range.get("from"):
        from_date = time_range["from"]
        query_parts.append(
            f"modifiedTime >= '{from_date.isoformat()}T00:00:00'"
        )

    if time_range and time_range.get("to"):
        end_date = time_range["to"]
        upper_bound = end_date + timedelta(days=1)
        query_parts.append(
            f"modifiedTime < '{upper_bound.isoformat()}T00:00:00'"
        )

    query_parts.append(
        f"({positive_query})",
    )

    if exclude_clauses:
        query_parts.extend(
            exclude_clauses
        )

    final_query = " and ".join(
        query_parts
    )

    print(
        "[DRIVE QUERY]",
        final_query,
    )

    return final_query


def score_file_relevance(
    file_info,
    question,
    search_plan=None,
    text="",
):
    """Score a Drive file using filename and extracted content relevance."""

    filename = file_info.get(
        "name",
        "",
    ).lower()

    content = text.lower()

    filename_words = set(
        re.findall(
            r"[a-z0-9]+",
            filename,
        )
    )

    content_words = set(
        re.findall(
            r"[a-z0-9]+",
            content,
        )
    )

    required_terms = []
    optional_terms = []
    context_terms = []
    phrases = []

    if search_plan:
        required_terms = [
            term.lower().strip()
            for term in search_plan.get(
                "required_terms",
                [],
            )
        ]

        optional_terms = [
            term.lower().strip()
            for term in search_plan.get(
                "optional_terms",
                [],
            )
        ]

        context_terms = [
            term.lower().strip()
            for term in search_plan.get(
                "context_terms",
                [],
            )
        ]

        phrases = [
            phrase.lower().strip()
            for phrase in search_plan.get(
                "phrases",
                [],
            )
        ]

        phrase_terms = []

        for phrase in phrases:
            phrase_terms.extend(
                term
                for term in re.findall(
                    r"[a-z0-9]+",
                    phrase,
                )
                if term not in {"that", "entered", "date", "dates", "list", "lists"}
            )

        search_terms = list(
            dict.fromkeys(
                required_terms
                + optional_terms
                + context_terms
                + phrase_terms
            )
        )

    else:
        search_terms = get_search_terms(
            question
        )

        required_terms = search_terms

    constraints = (
        search_plan.get("query_constraints")
        if isinstance(search_plan, dict)
        else None
    ) or extract_query_constraints(question)

    subject_entity = constraints.get("subject_entity")
    activity = constraints.get("activity")

    if content and (subject_entity or activity):
        evidence_sentences = re.split(r"(?<=[.!?])\s+|\n\s*\n", content)
        matched_relationship = False

        for sentence in evidence_sentences:
            sentence_words = re.findall(r"[a-z0-9]+", sentence)
            if subject_entity:
                subject_words = re.findall(r"[a-z0-9]+", subject_entity.lower())
                subject_present = bool(subject_words) and any(
                    sentence_words[index:index + len(subject_words)] == subject_words
                    for index in range(
                        max(0, len(sentence_words) - len(subject_words) + 1)
                    )
                )
            else:
                subject_present = True

            if activity:
                activity_words = re.findall(r"[a-z0-9]+", activity.lower())
                activity_positions = [
                    index
                    for index, word in enumerate(sentence_words)
                    if word in activity_words
                ]
                activity_present = (
                    len(set(activity_words)) == len(set(
                        sentence_words[index]
                        for index in activity_positions
                    ))
                    and bool(activity_positions)
                    and max(activity_positions) - min(activity_positions) <= 6
                )
            else:
                activity_present = True

            if subject_present and activity_present:
                matched_relationship = True
                break

        if not matched_relationship:
            return 0

    score = 0
    reasons = []

    if content and subject_entity:
        score += 120
        reasons.append("subject entity and activity relationship in evidence")
    if content and activity:
        score += 80
        reasons.append("activity terms in coherent evidence")

    # Score individual search terms.
    for term in search_terms:

        if not term:
            continue

        if term in {"2026", "2025", "2024", "2023"}:
            continue

        # Filename substring match.
        if term in filename:

            if term in required_terms:
                score += 35
                reasons.append(
                    f"required filename substring: {term}"
                )

            elif term in optional_terms:
                score += 10
                reasons.append(
                    f"optional filename substring: {term}"
                )

            elif term in context_terms:
                score += 3
                reasons.append(
                    f"context filename substring: {term}"
                )

            else:
                score += 5
                reasons.append(
                    f"filename substring: {term}"
                )

        # Exact filename word match.
        if term in filename_words:

            if term in required_terms:
                score += 30
                reasons.append(
                    f"required filename word: {term}"
                )

            elif term in optional_terms:
                score += 6
                reasons.append(
                    f"optional filename word: {term}"
                )

            elif term in context_terms:
                score += 2
                reasons.append(
                    f"context filename word: {term}"
                )

            else:
                score += 3
                reasons.append(
                    f"filename word: {term}"
                )

        # Exact content word match.
        if term in content_words:

            if term in required_terms:
                score += 15
                reasons.append(
                    f"required content: {term}"
                )

            elif term in optional_terms:
                score += 8
                reasons.append(
                    f"optional content: {term}"
                )

            elif term in context_terms:
                score += 10
                reasons.append(
                    f"context content: {term}"
                )

            else:
                score += 5
                reasons.append(
                    f"content: {term}"
                )

    # Score complete phrases.
    for phrase in phrases:

        if not phrase:
            continue

        # Complete phrase in filename.
        if phrase in filename:
            score += 80
            reasons.append(
                f"phrase filename: {phrase}"
            )

        # Complete phrase in content.
        if phrase in content:
            score += 60
            reasons.append(
                f"phrase content: {phrase}"
            )

    print(
        "[SCORE DEBUG]",
        file_info.get(
            "name",
            "Unnamed file",
        ),
        "score=",
        score,
        "reasons=",
        reasons,
    )

    return score


def _collect_documents_for_query(
    service,
    query,
    question,
    search_plan=None,
    search_audit=None,
):
    """
    Retrieve bounded metadata candidates, then extract and rank the best.

    MAX_FILES limits extraction and download work as well as final output.

    Drive shortcuts are resolved before checking whether their target
    file type is supported by the text extractor.
    """

    if search_audit is None:
        search_audit = {
            "candidates": [],
            "analyzed": [],
            "used": [],
            "assessments": {},
            "state": EvidenceState.NO_CANDIDATES.value,
            "skipped": [],
            "failed": [],
            "empty": [],
        }
    search_audit.setdefault("assessments", {})
    search_audit.setdefault("assessment_records", [])
    search_audit.setdefault("candidate_diagnostics", [])

    candidates = []
    skipped_files = []

    page_token = None

    candidate_limit = get_drive_candidate_limit()

    while len(candidates) < candidate_limit:

        remaining = (
            candidate_limit
            - len(candidates)
        )

        response = (
            service.files()
            .list(
                q=query,
                pageSize=min(
                    100,
                    remaining,
                ),
                pageToken=page_token,
                fields=(
                    "nextPageToken,"
                    "files("
                    "id,name,mimeType,size,"
                    "modifiedTime,shortcutDetails"
                    ")"
                ),
            )
            .execute()
        )

        page_files = response.get(
            "files",
            [],
        )

        if not page_files:
            break

        candidates.extend(
            page_files
        )

        # Record every file returned by Google Drive.
        search_audit["candidates"].extend(
            file_info.get(
                "name",
                "Unnamed file",
            )
            for file_info in page_files
        )

        page_token = response.get(
            "nextPageToken"
        )

        if not page_token:
            break

    print(
        "[DRIVE CANDIDATES]",
        len(candidates),
    )

    candidate_diagnostics = []
    diagnostics_by_id = {}
    for index, file_info in enumerate(candidates):
        filename = file_info.get("name", "Unnamed file")
        file_id = file_info.get("id")
        candidate_id = str(file_id or f"{filename}::{index}")
        diagnostic = {
            "candidate_id": candidate_id,
            "file_id": file_id,
            "file_name": filename,
            "mime_type": file_info.get("mimeType"),
            "extraction_method": _extraction_method(file_info),
            "retrieved": True,
            "extraction_status": ExtractionStatus.NOT_ATTEMPTED.value,
            "extraction_error": None,
            "extracted_char_count": 0,
            "extracted_text_available": False,
        }
        candidate_diagnostics.append(diagnostic)
        diagnostics_by_id[candidate_id] = diagnostic
    search_audit["candidate_diagnostics"] = candidate_diagnostics
    logger.debug(
        "Drive query constraints: %s",
        (search_plan or {}).get("query_constraints", {}),
    )

    metadata_ranked = [
        (
            score_file_relevance(
                file_info,
                question,
                search_plan,
                text="",
            ),
            file_info,
        )
        for file_info in candidates
    ]

    metadata_ranked.sort(
        key=lambda item: (
            -item[0],
            str(item[1].get("name", "")).casefold(),
            str(item[1].get("id", "")),
        ),
    )

    extraction_candidates = select_extraction_candidates(
        metadata_ranked,
        search_plan,
    )

    print(
        "[EXTRACTION CANDIDATES]",
        [
            file_info.get(
                "name",
                "Unnamed file",
            )
            for file_info in extraction_candidates
        ],
    )

    extracted_documents = []

    candidate_indexes = {id(info): index for index, info in enumerate(candidates)}
    for file_info in extraction_candidates:

        filename = file_info.get(
            "name",
            "Unnamed file",
        )
        candidate_index = candidate_indexes.get(id(file_info), 0)
        file_id = file_info.get("id")
        candidate_id = str(file_id or f"{filename}::{candidate_index}")
        candidate_diagnostic = diagnostics_by_id[candidate_id]

        def mark_extraction(status, error=None, text=None, resolved_info=None):
            candidate_diagnostic["extraction_status"] = status.value
            candidate_diagnostic["extraction_error"] = error
            if text is not None:
                candidate_diagnostic["extracted_char_count"] = len(text)
                candidate_diagnostic["extracted_text_available"] = bool(text.strip())
            if resolved_info is not None:
                candidate_diagnostic["mime_type"] = resolved_info.get("mimeType")
                candidate_diagnostic["extraction_method"] = _extraction_method(resolved_info)
            logger.debug("Drive extraction lifecycle: %s", candidate_diagnostic)

        # ---------------------------------------------------------
        # Resolve Drive shortcuts BEFORE checking extractability.
        #
        # A shortcut itself has MIME type
        # application/vnd.google-apps.shortcut, but its target may
        # be a supported PDF, DOCX, Google Doc, Sheet, etc.
        # ---------------------------------------------------------

        resolved_file_info = file_info

        try:
            resolved_file_info = resolve_drive_shortcut(
                service,
                file_info,
            )

        except DriveFileUnavailableError as exc:

            reason = str(exc)
            mark_extraction(ExtractionStatus.FAILURE, reason)

            skipped_files.append(
                filename
            )

            search_audit[
                "skipped"
            ].append({
                "name": filename,
                "reason": reason,
            })

            print(
                "[FILE SKIPPED]",
                filename,
                reason,
            )

            continue

        except HttpError as exc:

            status_code = getattr(
                exc.resp,
                "status",
                None,
            )

            if status_code == 404:
                reason = (
                    "File is unavailable or "
                    "the current account no longer "
                    "has access to it."
                )

            elif status_code == 403:
                reason = (
                    "The current account does not "
                    "have permission to read this file."
                )

            elif status_code in {
                500,
                502,
                503,
                504,
            }:
                reason = (
                    f"Google Drive temporarily "
                    f"returned HTTP {status_code}."
                )

            else:
                reason = (
                    f"Google Drive returned "
                    f"HTTP {status_code}."
                )

            mark_extraction(ExtractionStatus.FAILURE, reason)

            skipped_files.append(
                filename
            )

            search_audit[
                "skipped"
            ].append({
                "name": filename,
                "reason": reason,
            })

            print(
                "[FILE SKIPPED]",
                filename,
                reason,
            )

            continue

        # ---------------------------------------------------------
        # Check the RESOLVED target, not the shortcut itself.
        # ---------------------------------------------------------

        if not is_extractable_file(
            resolved_file_info
        ):
            reason = (
                "File type is not supported by the text extractor."
            )
            mark_extraction(
                ExtractionStatus.UNSUPPORTED,
                reason,
                resolved_info=resolved_file_info,
            )

            skipped_files.append(
                filename
            )

            search_audit[
                "skipped"
            ].append({
                "name": filename,
                "reason": reason,
            })

            print(
                "[FILE SKIPPED]",
                filename,
                reason,
            )

            continue

        # Record that this file entered the extraction stage.
        search_audit["analyzed"].append(
            filename
        )

        # ---------------------------------------------------------
        # Check the resolved target's size.
        #
        # Google-native files are exported through the Drive API,
        # so their normal file size is not used as a download limit.
        # ---------------------------------------------------------

        raw_size = resolved_file_info.get(
            "size"
        )

        if raw_size and not is_google_export(
            resolved_file_info
        ):

            try:
                if int(raw_size) > MAX_DOWNLOAD_BYTES:

                    reason = (
                        f"File exceeds the "
                        f"{MAX_DOWNLOAD_BYTES} byte "
                        f"download limit."
                    )
                    mark_extraction(
                        ExtractionStatus.FAILURE,
                        reason,
                        resolved_info=resolved_file_info,
                    )

                    skipped_files.append(
                        filename
                    )

                    search_audit[
                        "skipped"
                    ].append({
                        "name": filename,
                        "reason": reason,
                    })

                    print(
                        "[FILE SKIPPED]",
                        filename,
                        reason,
                    )

                    continue

            except (
                TypeError,
                ValueError,
            ):

                print(
                    "[FILE SIZE ERROR]",
                    filename,
                    repr(raw_size),
                )

        try:
            # Pass the resolved target to extraction so the shortcut
            # does not need to be resolved a second time.
            extracted_filename, text = (
                extract_file_text(
                    service,
                    resolved_file_info,
                )
            )

        except DriveFileTooLargeError as exc:

            reason = str(exc)
            mark_extraction(ExtractionStatus.FAILURE, reason, resolved_info=resolved_file_info)

            skipped_files.append(
                filename
            )

            search_audit[
                "skipped"
            ].append({
                "name": filename,
                "reason": reason,
            })

            print(
                "[FILE SKIPPED]",
                filename,
                reason,
            )

            continue

        except DriveFileUnavailableError as exc:

            reason = str(exc)
            mark_extraction(ExtractionStatus.FAILURE, reason, resolved_info=resolved_file_info)

            skipped_files.append(
                filename
            )

            search_audit[
                "skipped"
            ].append({
                "name": filename,
                "reason": reason,
            })

            print(
                "[FILE SKIPPED]",
                filename,
                reason,
            )

            continue

        except HttpError as exc:

            status_code = getattr(
                exc.resp,
                "status",
                None,
            )

            if status_code == 404:
                reason = (
                    "File is unavailable or "
                    "the current account no longer "
                    "has access to it."
                )

            elif status_code == 403:
                reason = (
                    "The current account does not "
                    "have permission to read this file."
                )

            elif status_code in {
                500,
                502,
                503,
                504,
            }:
                reason = (
                    f"Google Drive temporarily "
                    f"returned HTTP {status_code}."
                )

            else:
                reason = (
                    f"Google Drive returned "
                    f"HTTP {status_code}."
                )

            mark_extraction(ExtractionStatus.FAILURE, reason, resolved_info=resolved_file_info)

            skipped_files.append(
                filename
            )

            search_audit[
                "skipped"
            ].append({
                "name": filename,
                "reason": reason,
            })

            print(
                "[FILE SKIPPED]",
                filename,
                reason,
            )

            continue

        except (
            TimeoutError,
            OSError,
        ):

            reason = (
                "File operation timed out "
                "or could not be completed."
            )
            mark_extraction(ExtractionStatus.FAILURE, reason, resolved_info=resolved_file_info)

            skipped_files.append(
                filename
            )

            search_audit[
                "skipped"
            ].append({
                "name": filename,
                "reason": reason,
            })

            print(
                "[FILE SKIPPED]",
                filename,
                reason,
            )

            continue

        except Exception as exc:

            reason = (
                f"{type(exc).__name__}: {exc}"
            )
            mark_extraction(ExtractionStatus.FAILURE, reason, resolved_info=resolved_file_info)

            search_audit[
                "failed"
            ].append({
                "name": filename,
                "reason": reason,
            })

            print(
                "[FILE SKIPPED]",
                filename,
                reason,
            )

            continue

        # Keep the shortcut's visible filename in the search results.
        # The extracted content came from the resolved target.
        extracted_filename = filename

        if not isinstance(text, str):
            text = "" if text is None else str(text)

        # Successfully extracted but no usable text.
        if not text.strip():
            mark_extraction(
                ExtractionStatus.EMPTY,
                text=text,
                resolved_info=resolved_file_info,
            )

            search_audit[
                "empty"
            ].append(
                filename
            )

            print(
                "[EXTRACTION EMPTY]",
                filename,
            )

            continue

        text = text[
            :MAX_CHARS_PER_FILE
        ]
        mark_extraction(
            ExtractionStatus.SUCCESS,
            text=text,
            resolved_info=resolved_file_info,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Extracted text prefix for %s: %r",
                candidate_diagnostic["candidate_id"],
                text[:200],
            )

        score = score_file_relevance(
            file_info,
            question,
            search_plan,
            text,
        )

        extracted_documents.append(
            {
                "name": extracted_filename,
                "modifiedTime": resolved_file_info.get(
                    "modifiedTime",
                    file_info.get(
                        "modifiedTime",
                        "",
                    ),
                ),
                "text": text,
                "score": score,
                "candidate_id": candidate_id,
                "file_id": file_id,
                "mime_type": resolved_file_info.get("mimeType"),
                "extraction_method": candidate_diagnostic["extraction_method"],
                "extraction_status": candidate_diagnostic["extraction_status"],
                "extraction_error": None,
            }
        )

    assessment_result = validate_candidates(
        question,
        extracted_documents,
        search_plan,
        search_audit,
    )
    for document in extracted_documents:
        document["_evidence_assessment"] = assessment_result.assessments[
            document.get("file_id") or document["candidate_id"]
        ]

    if not extracted_documents and (
        search_audit.get("failed")
        or search_audit.get("skipped")
        or search_audit.get("empty")
    ):
        search_audit["state"] = EvidenceState.EXTRACTION_FAILURE.value

    # Highest relevance first.
    extracted_documents.sort(
        key=lambda document: (
            bool(document["_evidence_assessment"].eligible),
            document.get("score", 0),
        ),
        reverse=True,
    )

    documents = []
    total_characters = 0

    # Remove documents that are substantially less relevant than
    # the best matching document, but preserve completeness for
    # explicit-date list/aggregation retrieval where the user is asking
    # for the complete set across the requested range.
    if extracted_documents:

        highest_score = max(document.get("score", 0) for document in extracted_documents)

        if is_list_aggregation_query(search_plan):
            minimum_score = 1
        else:
            minimum_score = max(
                1,
                highest_score - 5,
            )

        extracted_documents = [
            document
            for document in extracted_documents
            if document["_evidence_assessment"].eligible
            or document.get("score", 0) >= minimum_score
        ]

        print(
            "[RELEVANCE FILTER]",
            "highest_score=",
            highest_score,
            "minimum_score=",
            minimum_score,
            "list_aggregation=",
            is_list_aggregation_query(search_plan),
            "remaining=",
            [
                {
                    "name": document["name"],
                    "score": document["score"],
                }
                for document in extracted_documents
            ],
        )

    document_limit = get_document_limit_for_query(search_plan)

    for document in extracted_documents:

        if len(documents) >= document_limit:
            break

        text = document["text"]

        remaining = (
            MAX_TOTAL_CHARS
            - total_characters
        )

        if remaining <= 0:
            break

        text = text[:remaining]

        if not text.strip():
            continue

        total_characters += len(text)

        document["text"] = text

        documents.append(
            document
        )

    # Candidate extraction is not evidence use; the answer boundary records used files.
    search_audit["used"] = []

    print(
        "[SEARCH RESULTS]",
        [
            {
                "name": document["name"],
                "score": document["score"],
            }
            for document in documents
        ],
    )

    return (
        documents,
        skipped_files,
        search_audit,
    )

def collect_drive_documents(
    discord_user_id: int,
    question: str,
    search_plan=None,
):
    """
    Read matching non-trashed, non-folder files accessible to the
    Google account connected to this Discord user.

    Google Drive performs broad candidate retrieval using the
    search-plan terms. Python then extracts supported files and
    ranks them using filenames, complete phrases, and extracted
    content.

    Shared files are intentionally included when the connected
    Google account has permission to read them.
    """

    search_audit = {
        "candidates": [],
        "analyzed": [],
        "used": [],
        "skipped": [],
        "failed": [],
        "empty": [],
    }

    try:
        service = get_drive_service(
            discord_user_id
        )

        drive_query = build_drive_search_query(
            question,
            search_plan,
        )

        return _collect_documents_for_query(
            service,
            drive_query,
            question,
            search_plan,
            search_audit,
        )

    except DriveError:
        raise

    except Exception as exc:

        print(
            "[DRIVE FETCH ERROR]",
            type(exc).__name__,
            repr(exc),
        )

        raise DriveError(
            "Google Drive could not be accessed. "
            "Make sure your Drive is connected and "
            "that the Google authorization is still valid."
        ) from exc