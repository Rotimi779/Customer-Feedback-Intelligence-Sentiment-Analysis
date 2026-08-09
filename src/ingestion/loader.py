"""Robust CSV loading for paths, bytes, and Streamlit-style uploads."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

import pandas as pd

from src.ingestion.schema import DEFAULT_INGESTION_CONFIG, IngestionConfig


class CSVLoadError(ValueError):
    """Base class for user-correctable CSV loading failures."""


class UnsupportedFileTypeError(CSVLoadError):
    """Raised when the supplied file is not named as a CSV."""


class FileSizeLimitError(CSVLoadError):
    """Raised when an upload exceeds the configured byte limit."""


class RowLimitError(CSVLoadError):
    """Raised when a CSV exceeds the configured row limit."""


class EmptyCSVError(CSVLoadError):
    """Raised when a CSV has no readable columns or rows."""


class CSVEncodingError(CSVLoadError):
    """Raised when none of the supported encodings can decode a file."""


class CSVParseError(CSVLoadError):
    """Raised when CSV structure is malformed."""


@runtime_checkable
class UploadLike(Protocol):
    """Minimal interface shared by Streamlit uploads and byte buffers."""

    name: str

    def getvalue(self) -> bytes:
        """Return the complete uploaded payload."""


CSVSource = str | Path | bytes | bytearray | BinaryIO | UploadLike


@dataclass(frozen=True, slots=True)
class LoadedCSV:
    """A loaded CSV and the metadata needed for UI feedback and logging."""

    dataframe: pd.DataFrame
    filename: str
    encoding: str
    size_bytes: int


def _validate_filename(filename: str) -> None:
    if Path(filename).suffix.lower() != ".csv":
        raise UnsupportedFileTypeError(
            "Only CSV files are supported. Save the dataset with a .csv extension."
        )


def _read_source(source: CSVSource, filename: str | None) -> tuple[bytes, str]:
    """Return source bytes and a display filename without importing Streamlit."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        resolved_name = filename or path.name
        _validate_filename(resolved_name)
        try:
            return path.read_bytes(), resolved_name
        except OSError as exc:
            raise CSVLoadError(f"The CSV file could not be read: {exc}") from exc

    if isinstance(source, (bytes, bytearray)):
        resolved_name = filename or "uploaded.csv"
        _validate_filename(resolved_name)
        return bytes(source), resolved_name

    if isinstance(source, UploadLike):
        resolved_name = filename or source.name
        _validate_filename(resolved_name)
        return source.getvalue(), resolved_name

    if hasattr(source, "read"):
        resolved_name = filename or getattr(source, "name", "uploaded.csv")
        _validate_filename(str(resolved_name))
        stream = source
        original_position: int | None = None
        try:
            if hasattr(stream, "tell"):
                original_position = stream.tell()
            if hasattr(stream, "seek"):
                stream.seek(0)
            payload = stream.read()
            if hasattr(stream, "seek") and original_position is not None:
                stream.seek(original_position)
        except OSError as exc:
            raise CSVLoadError(f"The uploaded CSV could not be read: {exc}") from exc

        if not isinstance(payload, (bytes, bytearray)):
            raise CSVLoadError("The uploaded file must provide binary CSV data.")
        return bytes(payload), str(resolved_name)

    raise TypeError(f"Unsupported CSV source type: {type(source).__name__}")


def load_csv(
    source: CSVSource,
    *,
    filename: str | None = None,
    config: IngestionConfig = DEFAULT_INGESTION_CONFIG,
) -> LoadedCSV:
    """Load a CSV using a small, explicit encoding fallback sequence.

    The parser reads at most ``max_rows + 1`` rows. This allows the workflow to
    reject oversized datasets without retaining the full file in memory.
    """
    payload, resolved_name = _read_source(source, filename)

    if not payload:
        raise EmptyCSVError(
            "The uploaded CSV is empty. Add a header row and customer feedback data."
        )

    if len(payload) > config.max_file_size_bytes:
        limit_mb = config.max_file_size_bytes / (1024 * 1024)
        raise FileSizeLimitError(
            f"The uploaded file is larger than the supported {limit_mb:g} MB limit. "
            "Reduce the file size or upload a representative sample."
        )

    decoding_failures: list[str] = []

    for encoding in config.supported_encodings:
        try:
            dataframe = pd.read_csv(
                BytesIO(payload),
                encoding=encoding,
                nrows=config.max_rows + 1,
                on_bad_lines="error",
            )
        except UnicodeDecodeError:
            decoding_failures.append(encoding)
            continue
        except pd.errors.EmptyDataError as exc:
            raise EmptyCSVError(
                "The CSV does not contain readable columns or rows. "
                "Confirm that the first row contains column headers."
            ) from exc
        except pd.errors.ParserError as exc:
            raise CSVParseError(
                "The CSV structure could not be parsed. Check delimiters, quoting, "
                "and whether every row has a consistent number of fields."
            ) from exc
        except (OSError, ValueError) as exc:
            raise CSVLoadError(f"The CSV could not be loaded: {exc}") from exc

        if len(dataframe) > config.max_rows:
            raise RowLimitError(
                f"The dataset exceeds the supported {config.max_rows:,}-row limit. "
                "Upload a representative sample for the interactive MVP."
            )

        if dataframe.empty or len(dataframe.columns) == 0:
            raise EmptyCSVError(
                "The CSV contains headers but no data rows. Add customer feedback "
                "before uploading it."
            )

        return LoadedCSV(
            dataframe=dataframe,
            filename=resolved_name,
            encoding=encoding,
            size_bytes=len(payload),
        )

    attempted = ", ".join(decoding_failures or config.supported_encodings)
    raise CSVEncodingError(
        "The CSV encoding is not supported. Re-save it as UTF-8 and upload it "
        f"again. Attempted encodings: {attempted}."
    )
