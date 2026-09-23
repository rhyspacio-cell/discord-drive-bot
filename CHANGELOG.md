# Changelog

All notable changes to this project will be documented in this file.
---

## [0.1.0] - 2026-09-24

### Added

#### Discord Bot

* Added Discord bot with slash-command support.
* Added `/connect-drive` command.
* Added `/drive-status` command.
* Added `/summarize-drive` command.
* Added `/disconnect-drive` command.
* Added ephemeral responses for user-specific Drive operations.

#### Google Drive

* Added Google OAuth authentication.
* Added per-Discord-user Google Drive connections.
* Added Google Drive API integration.
* Added support for reading Google Docs.
* Added support for reading Google Sheets.
* Added support for reading Google Slides.
* Added support for PDF files.
* Added support for DOCX files.
* Added support for text and common code/data files.

#### Authentication & Storage

* Added encrypted storage of Google OAuth credentials.
* Added SQLite-based persistent credential storage.
* Added OAuth state values for CSRF protection.
* Added OAuth state expiration and cleanup.
* Added `/disconnect-drive` functionality to remove stored credentials.

#### Local AI

* Added local Ollama integration.
* Added support for the `llama3.2:3b` model.
* Added configurable Ollama base URL.
* Added local document summarization.
* Added configurable summary length limits.

#### Content Protection

* Added maximum file limits.
* Added per-file character limits.
* Added total character limits.
* Added maximum summary length.
* Added instructions to treat Google Drive document contents as untrusted data.
* Added protection against blindly following instructions contained inside Drive documents.

#### Error Handling

* Added dedicated local LLM error handling.
* Added dedicated Google Drive error handling.
* Added per-file extraction error handling.
* Added logging for unexpected errors.
* Added timeouts for local Ollama requests.

#### Project Structure

* Split the application into separate modules for:

  * Configuration
  * Google Drive
  * Local LLM
  * OAuth
  * Credential storage
* Added `.gitignore` to prevent local secrets, databases, caches, and environment files from being committed.

### Security

* Added Fernet encryption for stored Google OAuth credentials.
* Kept `.env` outside version control.
* Kept `client_secret.json` outside version control.
* Kept local SQLite databases outside version control.
* Added OAuth state validation and expiration.
* Added limits on the amount of Drive content sent to the local model.
* Added prompt instructions to prevent document contents from being treated as system instructions.

### Known Limitations

* The bot is currently intended primarily for local/self-hosted use.
* Ollama must be running locally.
* Only selected Google Drive file formats are supported.
* Large files are truncated before summarization.
* The OAuth callback must be correctly configured and reachable.
* The OAuth connection flow should receive additional authorization hardening before production deployment.
* Local model output depends on the selected Ollama model.
* The application has not yet undergone a full production security review.

---