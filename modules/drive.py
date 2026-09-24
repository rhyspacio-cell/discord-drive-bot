import io
import re

from googleapiclient.discovery import build
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


class DriveError(RuntimeError):
    """Raised when Google Drive access or file extraction fails."""
    pass


class DriveFileTooLargeError(DriveError):
    """Raised when a file exceeds the configured download limit."""
    pass


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

    return build("drive", "v3", credentials=credentials)


def download_drive_file(service, file_id: str):
    """Download a normal Drive file."""
    request_ = service.files().get_media(fileId=file_id)

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request_)

    finished = False

    while not finished:
        _, finished = downloader.next_chunk()

        if buffer.tell() > MAX_DOWNLOAD_BYTES:
            raise DriveFileTooLargeError(
                f"File exceeds the {MAX_DOWNLOAD_BYTES} byte download limit."
            )

    return buffer.getvalue()


def extract_file_text(service, file_info):
    """Extract text from a supported Drive file."""
    file_id = file_info["id"]
    mime_type = file_info.get("mimeType", "")
    filename = file_info.get("name", "Unnamed file")

    # Google Docs
    if mime_type == "application/vnd.google-apps.document":
        data = service.files().export(
            fileId=file_id,
            mimeType="text/plain",
        ).execute()

        return filename, data.decode("utf-8", errors="replace")

    # Google Slides
    if mime_type == "application/vnd.google-apps.presentation":
        data = service.files().export(
            fileId=file_id,
            mimeType="text/plain",
        ).execute()

        return filename, data.decode("utf-8", errors="replace")

    # Google Sheets
    if mime_type == "application/vnd.google-apps.spreadsheet":
        data = service.files().export(
            fileId=file_id,
            mimeType="text/csv",
        ).execute()

        return filename, data.decode("utf-8", errors="replace")

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

    raw = download_drive_file(service, file_id)

    # PDF
    if is_pdf:
        reader = PdfReader(io.BytesIO(raw))

        extracted = []
        character_count = 0

        for page in reader.pages:
            page_text = page.extract_text() or ""

            remaining = MAX_CHARS_PER_FILE - character_count

            if remaining <= 0:
                break

            extracted_text = page_text[:remaining]
            extracted.append(extracted_text)
            character_count += len(extracted_text)

        return filename, "\n".join(extracted)

    # DOCX
    if is_docx:
        document = Document(io.BytesIO(raw))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
        ]

        return filename, "\n".join(paragraphs)

    # Normal text files
    if is_text:
        return filename, raw.decode("utf-8", errors="replace")

    return filename, ""


def is_google_export(file_info):
    """Return whether Drive handles the file through a native export."""
    return file_info.get("mimeType", "") in {
        "application/vnd.google-apps.document",
        "application/vnd.google-apps.presentation",
        "application/vnd.google-apps.spreadsheet",
    }


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


def build_drive_search_query(question: str, search_plan=None):
    """
    Build a broad Google Drive candidate query.

    Google Drive performs broad candidate retrieval using OR.

    Python performs the more precise relevance ranking afterward.
    """

    if search_plan is None:
        terms = get_search_terms(question)
        excludes = []

    else:
        required_terms = search_plan.get("required_terms", [])
        phrases = search_plan.get("phrases", [])
        optional_terms = search_plan.get("optional_terms", [])
        excludes = search_plan.get("exclude_terms", [])

        all_positive_values = (
            required_terms
            + phrases
            + optional_terms
        )

        if not all(
            isinstance(term, str) and term.strip()
            for term in all_positive_values
        ):
            raise DriveError(
                "The search plan contains invalid terms."
            )

        if not all(
            isinstance(term, str) and term.strip()
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

        terms = list(required_terms)

        # Break phrases into searchable words.
        #
        # Example:
        # "computer science"
        #
        # becomes:
        # "computer"
        # "science"
        #
        # Python later checks the complete phrase.
        for phrase in phrases:
            phrase_terms = re.findall(
                r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",
                phrase,
            )

            terms.extend(phrase_terms)

        terms.extend(optional_terms)

        # Remove duplicates while preserving order.
        terms = list(
            dict.fromkeys(
                term for term in terms if term
            )
        )

        if not terms:
            terms = get_search_terms(question)

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
        "and mimeType != 'application/vnd.google-apps.folder'"
    )

    # IMPORTANT:
    #
    # Use OR here.
    #
    # Example:
    #
    # diploma OR computer OR science
    #
    # This gives Python a broad candidate pool.
    positive_query = " or ".join(positive_clauses)

    query_parts = [
        base,
        f"({positive_query})",
    ]

    if exclude_clauses:
        query_parts.extend(exclude_clauses)

    final_query = " and ".join(query_parts)

    print("[DRIVE QUERY]", final_query)

    return final_query


def score_file_relevance(
    file_info,
    question,
    search_plan=None,
    text="",
):
    """Score a Drive file using filename and extracted content relevance."""

    filename = file_info.get("name", "").lower()
    content = text.lower()

    filename_words = set(
        re.findall(r"[a-z0-9]+", filename)
    )

    content_words = set(
        re.findall(r"[a-z0-9]+", content)
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
                re.findall(
                    r"[a-z0-9]+",
                    phrase,
                )
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
        search_terms = get_search_terms(question)
        required_terms = search_terms

    score = 0
    reasons = []

    # Score individual search terms.
    for term in search_terms:
        if not term:
            continue

        # Filename substring match.
        if term in filename:
            if term in required_terms:
                score += 35
                reasons.append(f"required filename substring: {term}")
            elif term in optional_terms:
                score += 10
                reasons.append(f"optional filename substring: {term}")
            elif term in context_terms:
                score += 3
                reasons.append(f"context filename substring: {term}")
            else:
                score += 5
                reasons.append(f"filename substring: {term}")

        # Exact filename word match.
        if term in filename_words:
            if term in required_terms:
                score += 30
                reasons.append(f"required filename word: {term}")
            elif term in optional_terms:
                score += 6
                reasons.append(f"optional filename word: {term}")
            elif term in context_terms:
                score += 2
                reasons.append(f"context filename word: {term}")
            else:
                score += 3
                reasons.append(f"filename word: {term}")

        # Exact content word match.
        if term in content_words:
            if term in required_terms:
                score += 15
                reasons.append(f"required content: {term}")
            elif term in optional_terms:
                score += 8
                reasons.append(f"optional content: {term}")
            elif term in context_terms:
                score += 4
                reasons.append(f"context content: {term}")
            else:
                score += 5
                reasons.append(f"content: {term}")

    # Score complete phrases.
    for phrase in phrases:
        if not phrase:
            continue

        # A complete phrase in the filename is a very strong signal.
        if phrase in filename:
            score += 80
            reasons.append(f"phrase filename: {phrase}")

        # A complete phrase in the document content is also strong.
        if phrase in content:
            score += 60
            reasons.append(f"phrase content: {phrase}")

    print(
        "[SCORE DEBUG]",
        file_info.get("name", "Unnamed file"),
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
):
    """
    Retrieve bounded metadata candidates, then extract and rank the best.

    MAX_FILES limits extraction and download work as well as final output.
    """

    candidates = []
    skipped_files = []

    page_token = None

    candidate_limit = max(MAX_FILES * 3, MAX_FILES)

    while len(candidates) < candidate_limit:
        remaining = candidate_limit - len(candidates)
        response = (
            service.files()
            .list(
                q=query,
                pageSize=min(100, remaining),
                pageToken=page_token,
                fields=(
                    "nextPageToken,"
                    "files("
                    "id,"
                    "name,"
                    "mimeType,"
                    "size,"
                    "modifiedTime"
                    ")"
                ),
            )
            .execute()
        )

        page_files = response.get("files", [])
        if not page_files:
            break
        candidates.extend(page_files)

        page_token = response.get("nextPageToken")

        if not page_token:
            break

    print(
        "[DRIVE CANDIDATES]",
        len(candidates),
    )

    metadata_ranked = [
        (
            score_file_relevance(file_info, question, search_plan, text=""),
            file_info,
        )
        for file_info in candidates
    ]
    metadata_ranked.sort(key=lambda item: item[0], reverse=True)
    extraction_candidates = [
        file_info for _, file_info in metadata_ranked[:MAX_FILES]
    ]
    print(
        "[EXTRACTION CANDIDATES]",
        [file_info.get("name", "Unnamed file") for file_info in extraction_candidates],
    )

    extracted_documents = []

    for file_info in extraction_candidates:
        raw_size = file_info.get("size")

        # Skip oversized normal files before downloading them.
        #
        # Google-native files are excluded because they are exported
        # through the Drive API rather than downloaded normally.
        if raw_size and not is_google_export(file_info):
            try:
                if int(raw_size) > MAX_DOWNLOAD_BYTES:
                    filename = file_info.get(
                        "name",
                        "Unnamed file",
                    )

                    skipped_files.append(filename)

                    print(
                        "[FILE SKIPPED]",
                        filename,
                        (
                            f"File exceeds the "
                            f"{MAX_DOWNLOAD_BYTES} byte "
                            f"download limit."
                        ),
                    )

                    continue

            except (TypeError, ValueError):
                print(
                    "[FILE SIZE ERROR]",
                    file_info.get("name"),
                    repr(raw_size),
                )

        try:
            filename, text = extract_file_text(
                service,
                file_info,
            )

        except DriveFileTooLargeError as exc:
            filename = file_info.get(
                "name",
                "Unnamed file",
            )

            skipped_files.append(filename)

            print(
                "[FILE SKIPPED]",
                filename,
                str(exc),
            )

            continue

        except Exception as exc:
            print(
                "[EXTRACTION ERROR]",
                file_info.get("name", "Unnamed file"),
                type(exc).__name__,
                repr(exc),
            )
            print(
                "[FILE ERROR]",
                file_info.get("name"),
                type(exc).__name__,
                repr(exc),
            )

            continue

        if not text.strip():
            print(
                "[EXTRACTION EMPTY]",
                file_info.get("name", "Unnamed file"),
            )
            continue

        text = text[:MAX_CHARS_PER_FILE]

        score = score_file_relevance(
            file_info,
            question,
            search_plan,
            text,
        )

        extracted_documents.append(
            {
                "name": filename,
                "modifiedTime": file_info.get(
                    "modifiedTime",
                    "",
                ),
                "text": text,
                "score": score,
            }
        )

    # Highest relevance first.
    extracted_documents.sort(
        key=lambda document: document.get(
            "score",
            0,
        ),
        reverse=True,
    )

    documents = []
    total_characters = 0

    for document in extracted_documents:
        if len(documents) >= MAX_FILES:
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

        documents.append(document)

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

    return documents, skipped_files


def collect_drive_documents(
    discord_user_id: int,
    question: str,
    search_plan=None,
):
    """
    Read matching non-trashed, non-folder files accessible to the Google
    account connected to this Discord user.

    Google Drive performs broad candidate retrieval using the search-plan
    terms. Python then extracts supported files and ranks them using
    filenames, complete phrases, and extracted content.

    Shared files are intentionally included when the connected Google
    account has permission to read them.
    """

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