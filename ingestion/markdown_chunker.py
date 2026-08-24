"""Header-aware chunking for Markdown documentation."""

import re
from pathlib import Path

from ingestion.ast_chunker import CodeChunk
from ingestion.file_walker import FileInfo


_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def chunk_markdown_file(file_path: str, content: str) -> list[CodeChunk]:
    """Split Markdown at headings while preserving each section's source text."""

    lines = content.splitlines(keepends=True)
    headings = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := _HEADING_PATTERN.match(line.rstrip("\r\n")))
    ]
    if not headings:
        return [_chunk(file_path, content, Path(file_path).name, 1, max(1, len(lines)))]

    chunks: list[CodeChunk] = []
    if headings[0][0] > 0:
        chunks.append(_chunk(file_path, "".join(lines[: headings[0][0]]), "preamble", 1, headings[0][0]))

    for position, (start, match) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        title = match.group(2).strip()
        chunks.append(
            _chunk(file_path, "".join(lines[start:end]), title, start + 1, end, len(match.group(1)))
        )
    return chunks


def chunk_markdown(file_info: FileInfo) -> list[CodeChunk]:
    """Read a Markdown file and return header-aware chunks."""

    content = file_info.path.read_text(encoding="utf-8", errors="replace")
    return chunk_markdown_file(file_info.relative_path, content)


def _chunk(
    file_path: str,
    content: str,
    name: str,
    line_start: int,
    line_end: int,
    header_level: int | None = None,
) -> CodeChunk:
    unique_name = f"{name}@{line_start}"
    return CodeChunk(
        content=content,
        chunk_type="markdown",
        name=name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        language="markdown",
        chunk_id=f"{file_path}::markdown::{unique_name}",
        section_title=name,
        header_level=header_level,
    )
