"""Shared utilities for extracting clean text from LLM responses."""

from typing import Any


def extract_text(response: Any) -> str:
    """Extract clean text from an LLM response, handling all known formats.

    Gemini 3.6+ returns content as a list of content blocks:
        [{'type': 'text', 'text': '...', 'extras': {...}}]

    Older models / other providers return content as a plain string.
    This function normalizes both to a clean string.
    """
    content = getattr(response, "content", response)

    # Case 1: Already a string
    if isinstance(content, str):
        return content

    # Case 2: List of content blocks (Gemini 3.6+ format)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                # Extract text from {'type': 'text', 'text': '...'} blocks
                text = block.get("text", "")
                if text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
            else:
                # Try .text attribute (AIMessageChunk content blocks)
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        if parts:
            return "\n\n".join(parts)

    # Case 3: Fallback — stringify whatever we got
    return str(content)
