"""AST-aware code chunking for Python plus resilient multi-language fallbacks."""

import ast
import re
from pathlib import Path

from pydantic import BaseModel, Field

from ingestion.file_walker import FileInfo


LARGE_FUNCTION_LINES = 200
LARGE_CLASS_LINES = 500
FALLBACK_CHUNK_LINES = 100
FALLBACK_OVERLAP_LINES = 20


class CodeChunk(BaseModel):
    """A semantically meaningful unit of code, ready for indexing."""

    content: str
    chunk_type: str
    name: str
    file_path: str
    line_start: int
    line_end: int
    language: str
    parent_class: str | None = None
    docstring: str | None = None
    params: list[str] = Field(default_factory=list)
    returns: str | None = None
    imports: list[str] = Field(default_factory=list)
    decorators: list[str] = Field(default_factory=list)
    chunk_id: str
    is_large: bool = False
    section_title: str | None = None
    header_level: int | None = None


def chunk_file(file_info: FileInfo) -> list[CodeChunk]:
    """Read and chunk a file according to its detected language."""

    try:
        content = file_info.path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise ValueError(f"Could not read source file: {file_info.relative_path}") from error

    if file_info.language == "python":
        return chunk_python_file(file_info.relative_path, content)
    if file_info.language in {"javascript", "typescript"}:
        return chunk_js_ts_file(file_info.relative_path, content, file_info.language)
    return chunk_generic_file(file_info.relative_path, content, file_info.language)


def chunk_python_file(file_path: str, content: str) -> list[CodeChunk]:
    """Extract module, class, and function chunks from Python source."""

    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return chunk_generic_file(file_path, content, "python")

    lines = content.splitlines(keepends=True)
    imports = [
        _source_for_node(node, lines).strip()
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    chunks = [_module_chunk(file_path, tree, lines, imports)]

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(_function_chunk(file_path, node, lines))
        elif isinstance(node, ast.ClassDef):
            chunks.append(_class_chunk(file_path, node, lines))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(_function_chunk(file_path, child, lines, parent_class=node.name))
    return chunks


def chunk_js_ts_file(file_path: str, content: str, language: str = "javascript") -> list[CodeChunk]:
    """Use Tree-sitter when available; otherwise retain meaningful text chunks."""

    try:
        from tree_sitter import Language, Parser
        import tree_sitter_javascript

        language_capsule = tree_sitter_javascript.language()
        parser = Parser(Language(language_capsule))
        tree = parser.parse(content.encode("utf-8"))
    except (ImportError, AttributeError, TypeError):
        return chunk_generic_file(file_path, content, language)

    source = content.encode("utf-8")
    chunks = [_generic_module_chunk(file_path, content, language)]
    for node in _walk_tree(tree.root_node):
        if node.type not in {"function_declaration", "class_declaration", "method_definition"}:
            continue
        node_content = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode("utf-8") if name_node else f"{node.type}_{node.start_point.row + 1}"
        chunk_type = "class" if node.type == "class_declaration" else "function"
        chunks.append(
            CodeChunk(
                content=node_content,
                chunk_type=chunk_type,
                name=name,
                file_path=file_path,
                line_start=node.start_point.row + 1,
                line_end=node.end_point.row + 1,
                language=language,
                chunk_id=_chunk_id(file_path, chunk_type, name),
                is_large=(node.end_point.row - node.start_point.row + 1) > LARGE_FUNCTION_LINES,
            )
        )
    return chunks


def chunk_generic_file(file_path: str, content: str, language: str = "text") -> list[CodeChunk]:
    """Split unsupported source into definition-oriented or bounded line chunks."""

    lines = content.splitlines(keepends=True)
    if not lines:
        return [_generic_module_chunk(file_path, content, language)]

    boundaries = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^\s*(?:def |class |function |const\s+\w+\s*=|export\s+)", line)
    ]
    if boundaries:
        boundaries.append(len(lines))
        chunks = [_generic_module_chunk(file_path, content, language)]
        for index, end in zip(boundaries, boundaries[1:]):
            block = "".join(lines[index:end])
            name = _generic_name(lines[index], index + 1)
            chunks.append(
                _make_generic_chunk(file_path, block, language, name, index + 1, end)
            )
        return chunks

    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + FALLBACK_CHUNK_LINES, len(lines))
        chunks.append(
            _make_generic_chunk(
                file_path,
                "".join(lines[start:end]),
                language,
                f"chunk_{start + 1}",
                start + 1,
                end,
            )
        )
        if end == len(lines):
            break
        start = end - FALLBACK_OVERLAP_LINES
    return chunks


def _module_chunk(file_path: str, tree: ast.Module, lines: list[str], imports: list[str]) -> CodeChunk:
    docstring = ast.get_docstring(tree)
    definitions = [
        _definition_signature(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    content_parts = []
    if docstring:
        content_parts.append(f"Module docstring:\n{docstring}")
    if imports:
        content_parts.append("Imports:\n" + "\n".join(imports))
    if definitions:
        content_parts.append("Definitions:\n" + "\n".join(definitions))
    return CodeChunk(
        content="\n\n".join(content_parts) or "Module contains no import or definition summary.",
        chunk_type="module",
        name=Path(file_path).name,
        file_path=file_path,
        line_start=1,
        line_end=max(1, len(lines)),
        language="python",
        docstring=docstring,
        imports=imports,
        chunk_id=_chunk_id(file_path, "module", Path(file_path).name),
    )


def _class_chunk(file_path: str, node: ast.ClassDef, lines: list[str]) -> CodeChunk:
    line_start = _node_start_line(node)
    line_end = node.end_lineno or node.lineno
    is_large = line_end - line_start + 1 > LARGE_CLASS_LINES
    methods = [child for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
    content = _class_summary(node) if is_large else _source_for_node(node, lines)
    return CodeChunk(
        content=content,
        chunk_type="class",
        name=node.name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        language="python",
        docstring=ast.get_docstring(node),
        decorators=_decorators(node),
        chunk_id=_chunk_id(file_path, "class", node.name),
        is_large=is_large,
    )


def _function_chunk(
    file_path: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    parent_class: str | None = None,
) -> CodeChunk:
    line_start = _node_start_line(node)
    line_end = node.end_lineno or node.lineno
    return CodeChunk(
        content=_source_for_node(node, lines),
        chunk_type="function",
        name=node.name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        language="python",
        parent_class=parent_class,
        docstring=ast.get_docstring(node),
        params=_parameters(node),
        returns=ast.unparse(node.returns) if node.returns else None,
        decorators=_decorators(node),
        # Methods such as ``__init__`` commonly appear in several classes in
        # one file, so their IDs must include the owning class.
        chunk_id=_chunk_id(
            file_path,
            "function",
            f"{parent_class}.{node.name}" if parent_class else node.name,
        ),
        is_large=line_end - line_start + 1 > LARGE_FUNCTION_LINES,
    )


def _source_for_node(node: ast.AST, lines: list[str]) -> str:
    start = _node_start_line(node)
    end = getattr(node, "end_lineno", None) or getattr(node, "lineno", start)
    return "".join(lines[start - 1 : end])


def _node_start_line(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([getattr(node, "lineno", 1), *(item.lineno for item in decorators)])


def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    params = [
        f"{argument.arg}: {ast.unparse(argument.annotation)}"
        if argument.annotation
        else argument.arg
        for argument in arguments
    ]
    if node.args.vararg:
        params.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        params.append(f"**{node.args.kwarg.arg}")
    return params


def _decorators(node: ast.AST) -> list[str]:
    return [f"@{ast.unparse(item)}" for item in getattr(node, "decorator_list", [])]


def _definition_signature(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(base) for base in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"{prefix} {node.name}({', '.join(_parameters(node))}){returns}"
    return ""


def _class_summary(node: ast.ClassDef) -> str:
    summary = [_definition_signature(node)]
    if docstring := ast.get_docstring(node):
        summary.append(docstring)
    for method in node.body:
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            description = _definition_signature(method)
            if method_docstring := ast.get_docstring(method):
                description += f" — {method_docstring}"
            summary.append(description)
    return "\n".join(summary)


def _generic_module_chunk(file_path: str, content: str, language: str) -> CodeChunk:
    line_count = max(1, len(content.splitlines()))
    return CodeChunk(
        content=f"Module summary for {file_path}",
        chunk_type="module",
        name=Path(file_path).name,
        file_path=file_path,
        line_start=1,
        line_end=line_count,
        language=language,
        chunk_id=_chunk_id(file_path, "module", Path(file_path).name),
    )


def _make_generic_chunk(
    file_path: str, content: str, language: str, name: str, line_start: int, line_end: int
) -> CodeChunk:
    return CodeChunk(
        content=content,
        chunk_type="function",
        name=name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        language=language,
        chunk_id=_chunk_id(file_path, "function", name),
        is_large=line_end - line_start + 1 > LARGE_FUNCTION_LINES,
    )


def _generic_name(line: str, line_number: int) -> str:
    match = re.search(r"(?:def|class|function|const)\s+(\w+)", line)
    return match.group(1) if match else f"definition_{line_number}"


def _chunk_id(file_path: str, chunk_type: str, name: str) -> str:
    return f"{file_path}::{chunk_type}::{name}"


def _walk_tree(node: object):
    yield node
    for child in getattr(node, "children", []):
        yield from _walk_tree(child)
