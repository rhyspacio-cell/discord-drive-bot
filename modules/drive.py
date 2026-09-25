import io
import random
import re
import time

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
        ]

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


def build_drive_search_query(
    question: str,
    search_plan=None,
):
    """
    Build a broad Google Drive candidate query.

    Google Drive performs broad candidate retrieval using OR.

    Python performs the more precise relevance ranking afterward.
    """

    if search_plan is None:
        terms = get_search_terms(question)
        excludes = []

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

        all_positive_values = (
            required_terms
            + phrases
            + optional_terms
        )

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

        # Break phrases into searchable words.
        for phrase in phrases:
            phrase_terms = re.findall(
                r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",
                phrase,
            )

            terms.extend(
                phrase_terms
            )

        terms.extend(
            optional_terms
        )

        # Remove duplicates while preserving order.
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
        f"({positive_query})",
    ]

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
        search_terms = get_search_terms(
            question
        )

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
            "skipped": [],
            "failed": [],
            "empty": [],
        }

    candidates = []
    skipped_files = []

    page_token = None

    candidate_limit = max(
        MAX_FILES * 3,
        MAX_FILES,
    )

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
        key=lambda item: item[0],
        reverse=True,
    )

    extraction_candidates = [
        file_info
        for _, file_info in metadata_ranked[:MAX_FILES]
    ]

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

    for file_info in extraction_candidates:

        filename = file_info.get(
            "name",
            "Unnamed file",
        )

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

        # Successfully extracted but no usable text.
        if not text.strip():

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

    # Remove documents that are substantially less relevant
    # than the best matching document.
    if extracted_documents:

        highest_score = (
            extracted_documents[0].get(
                "score",
                0,
            )
        )

        minimum_score = max(
            1,
            highest_score - 5,
        )

        extracted_documents = [
            document
            for document in extracted_documents
            if document.get(
                "score",
                0,
            ) >= minimum_score
        ]

        print(
            "[RELEVANCE FILTER]",
            "highest_score=",
            highest_score,
            "minimum_score=",
            minimum_score,
            "remaining=",
            [
                {
                    "name": document["name"],
                    "score": document["score"],
                }
                for document in extracted_documents
            ],
        )

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

        documents.append(
            document
        )

    # These are the files actually sent to the LLM.
    search_audit["used"] = [
        document["name"]
        for document in documents
    ]

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