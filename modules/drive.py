import io
import re

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader
from docx import Document

from modules.config import MAX_CHARS_PER_FILE, MAX_DOWNLOAD_BYTES, MAX_FILES, MAX_TOTAL_CHARS
from modules.storage import load_credentials, save_credentials


class DriveError(RuntimeError):
    """Raised when Google Drive access or file extraction fails."""
    pass


class DriveFileTooLargeError(DriveError):
    """Raised when a file exceeds the configured download limit."""


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

    if mime_type == "application/vnd.google-apps.document":
        data = service.files().export(fileId=file_id, mimeType="text/plain").execute()
        return filename, data.decode("utf-8", errors="replace")

    if mime_type == "application/vnd.google-apps.presentation":
        data = service.files().export(fileId=file_id, mimeType="text/plain").execute()
        return filename, data.decode("utf-8", errors="replace")

    if mime_type == "application/vnd.google-apps.spreadsheet":
        data = service.files().export(fileId=file_id, mimeType="text/csv").execute()
        return filename, data.decode("utf-8", errors="replace")

    is_pdf = mime_type == "application/pdf" or filename.lower().endswith(".pdf")
    is_docx = (
        mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or filename.lower().endswith(".docx")
    )
    text_extensions = (".txt", ".md", ".csv", ".json", ".xml", ".html", ".py", ".js", ".ts", ".css", ".sql")
    is_text = mime_type.startswith("text/") or filename.lower().endswith(text_extensions)

    if not is_pdf and not is_docx and not is_text:
        return filename, ""

    raw = download_drive_file(service, file_id)

    if is_pdf:
        reader = PdfReader(io.BytesIO(raw))
        extracted = []
        character_count = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            remaining = MAX_CHARS_PER_FILE - character_count
            if remaining <= 0:
                break
            extracted.append(page_text[:remaining])
            character_count += len(extracted[-1])
        return filename, "\n".join(extracted)

    if is_docx:
        document = Document(io.BytesIO(raw))
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        return filename, "\n".join(paragraphs)

    if is_text:
        return filename, raw.decode("utf-8", errors="replace")

    return filename, ""


SEARCH_STOP_WORDS = {
    "a", "an", "and", "are", "for", "from", "in", "is", "my", "of",
    "on", "or", "the", "to", "was", "what", "when", "where", "which",
    "who", "with",
}


def build_drive_search_queries(search_query: str):
    """Build strict and broadened Drive queries from natural search terms."""
    terms = [
        term
        for term in re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", search_query.lower())
        if term not in SEARCH_STOP_WORDS
    ]
    if not terms:
        raise DriveError("Enter meaningful search terms to find Drive files.")

    def escape(term):
        return term.replace("\\", "\\\\").replace("'", "\\'")

    clauses = [
        f"(name contains '{escape(term)}' or fullText contains '{escape(term)}')"
        for term in terms
    ]
    base = "trashed = false and mimeType != 'application/vnd.google-apps.folder'"
    strict_query = f"{base} and {' and '.join(clauses)}"
    broad_query = strict_query if len(clauses) == 1 else f"{base} and ({' or '.join(clauses)})"
    return strict_query, broad_query


def _collect_documents_for_query(service, query):
    """Extract readable documents for one Drive query."""
    documents = []
    skipped_files = []
    total_characters = 0
    page_token = None
    seen = 0

    while True:
        response = (
            service.files()
            .list(
                q=query,
                pageSize=min(MAX_FILES - seen, 100) if MAX_FILES - seen > 0 else 1,
                pageToken=page_token,
                fields=(
                    "nextPageToken,"
                    "files("
                    "id,"
                    "name,"
                    "mimeType,"
                    "modifiedTime"
                    ")"
                ),
            )
            .execute()
        )

        for file_info in response.get("files", []):
            if seen >= MAX_FILES:
                break
            seen += 1

            try:
                filename, text = extract_file_text(service, file_info)
            except DriveFileTooLargeError as exc:
                filename = file_info.get("name", "Unnamed file")
                skipped_files.append(filename)
                print("[FILE SKIPPED]", filename, str(exc))
                continue
            except Exception as exc:
                print("[FILE ERROR]", file_info.get("name"), type(exc).__name__, repr(exc))
                continue

            if not text.strip():
                continue

            text = text[:MAX_CHARS_PER_FILE]
            remaining = MAX_TOTAL_CHARS - total_characters
            if remaining <= 0:
                return documents, skipped_files
            text = text[:remaining]
            total_characters += len(text)

            documents.append(
                {
                    "name": filename,
                    "modifiedTime": file_info.get("modifiedTime", ""),
                    "text": text,
                }
            )

        if seen >= MAX_FILES:
            break

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return documents, skipped_files


def collect_drive_documents(discord_user_id: int, search_query: str):
    """
    Read matching non-trashed, non-folder files accessible to the Google
    account connected to this Discord user.

    Shared files are intentionally included when the connected Google account
    has permission to read them. Natural-language search terms are matched
    against both file names and Drive's token-based full-text index. All
    meaningful terms are tried first; if that finds no readable files, the
    search broadens to match any term. Results are limited by MAX_FILES,
    MAX_CHARS_PER_FILE, and MAX_TOTAL_CHARS.
    """
    try:
        service = get_drive_service(discord_user_id)
        strict_query, broad_query = build_drive_search_queries(search_query)
        documents, skipped_files = _collect_documents_for_query(service, strict_query)
        if documents or strict_query == broad_query:
            return documents, skipped_files

        broad_documents, broad_skipped_files = _collect_documents_for_query(service, broad_query)
        return broad_documents, skipped_files + broad_skipped_files

    except DriveError:
        raise
    except Exception as exc:
        print("[DRIVE FETCH ERROR]", type(exc).__name__, repr(exc))
        raise DriveError(
            "Google Drive could not be accessed. "
            "Make sure your Drive is connected and "
            "that the Google authorization is still valid."
            ) from exc
