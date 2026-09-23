import io

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader
from docx import Document

from modules.config import MAX_CHARS_PER_FILE, MAX_FILES, MAX_TOTAL_CHARS
from modules.storage import load_credentials, save_credentials


def get_drive_service(discord_user_id: int):
    """Create a Google Drive API client for one Discord user."""
    credentials = load_credentials(discord_user_id)
    if not credentials:
        raise RuntimeError("Your Google Drive is not connected. Use /connect-drive first.")

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

    raw = download_drive_file(service, file_id)

    if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
        return filename, "\n".join(pages)

    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or filename.lower().endswith(".docx"):
        document = Document(io.BytesIO(raw))
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        return filename, "\n".join(paragraphs)

    text_extensions = (".txt", ".md", ".csv", ".json", ".xml", ".html", ".py", ".js", ".ts", ".css", ".sql")
    if mime_type.startswith("text/") or filename.lower().endswith(text_extensions):
        return filename, raw.decode("utf-8", errors="replace")

    return filename, ""


def collect_drive_documents(discord_user_id: int):
    """Find and extract readable files belonging to this user's connected Drive."""
    try:
        service = get_drive_service(discord_user_id)
        query = "trashed = false and mimeType != 'application/vnd.google-apps.folder'"

        documents = []
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
                    orderBy="modifiedTime desc",
                    fields=(
                        "nextPageToken,"
                        "files("
                        "id,"
                        "name,"
                        "mimeType,"
                        "modifiedTime,"
                        "webViewLink"
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
                except Exception as exc:
                    print("[FILE ERROR]", file_info.get("name"), type(exc).__name__, repr(exc))
                    continue

                if not text.strip():
                    continue

                text = text[:MAX_CHARS_PER_FILE]
                remaining = MAX_TOTAL_CHARS - total_characters
                if remaining <= 0:
                    return documents
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

        return documents

    except Exception as exc:
        print("[DRIVE FETCH ERROR]", type(exc).__name__, repr(exc))
        raise RuntimeError(
            "Google Drive could not be accessed. Make sure your Drive is connected and the Google authorization is still valid."
        ) from exc
