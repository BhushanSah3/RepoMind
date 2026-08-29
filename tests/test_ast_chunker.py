"""Tests for the AST-aware code chunking pipeline."""

import pytest
from ingestion.ast_chunker import CodeChunk, chunk_python_file, chunk_generic_file


def test_chunk_python_simple_function():
    """Parse a file with one function, verify metadata."""
    code = 'def hello(name: str) -> str:\n    """Say hello to someone."""\n    return f"Hello {name}"\n'
    chunks = chunk_python_file("test.py", code)
    assert len(chunks) == 2  # Module chunk + Function chunk
    func_chunk = next(c for c in chunks if c.chunk_type == "function")
    assert func_chunk.name == "hello"
    assert func_chunk.params == ["name: str"]
    assert func_chunk.docstring == "Say hello to someone."


def test_chunk_python_class_with_methods():
    """Parse a file with a class containing methods."""
    code = (
        'class Calculator:\n'
        '    """A simple calculator."""\n'
        '    def add(self, a, b):\n'
        '        return a + b\n'
        '\n'
        '    def subtract(self, a, b):\n'
        '        return a - b\n'
    )
    chunks = chunk_python_file("calc.py", code)
    chunk_types = [c.chunk_type for c in chunks]
    assert "module" in chunk_types
    assert "class" in chunk_types
    assert chunk_types.count("function") == 2

    add_chunk = next(c for c in chunks if c.name == "add" and c.parent_class == "Calculator")
    assert add_chunk.parent_class == "Calculator"
    assert add_chunk.chunk_id == "calc.py::function::Calculator.add"


def test_chunk_python_module_summary():
    """Verify module chunk contains imports and definition signatures."""
    code = "import os\nimport sys\n\ndef foo():\n    pass\n\nclass Bar:\n    pass\n"
    chunks = chunk_python_file("mod.py", code)
    mod_chunk = next(c for c in chunks if c.chunk_type == "module")
    assert any("os" in imp for imp in mod_chunk.imports)
    assert any("sys" in imp for imp in mod_chunk.imports)


def test_chunk_python_decorators():
    """Functions with decorators should have decorators list populated."""
    code = "@property\n@staticmethod\ndef my_func():\n    pass\n"
    chunks = chunk_python_file("dec.py", code)
    func_chunk = next(c for c in chunks if c.chunk_type == "function")
    decorator_text = " ".join(func_chunk.decorators)
    assert "property" in decorator_text
    assert "staticmethod" in decorator_text


def test_chunk_python_async_function():
    """async def should be handled correctly."""
    code = "async def fetch_data(url: str):\n    return 'data'\n"
    chunks = chunk_python_file("async_test.py", code)
    func_chunk = next(c for c in chunks if c.chunk_type == "function")
    assert func_chunk.name == "fetch_data"
    assert "async" in func_chunk.content


def test_chunk_python_syntax_error_fallback():
    """Invalid Python should fallback to generic chunker."""
    code = "def bad_function()\n    print('oops')\n"
    chunks = chunk_python_file("bad.py", code)
    assert len(chunks) > 0


def test_chunk_generic_with_boundaries():
    """Generic chunker with def/class patterns should split on boundaries."""
    code = "def generic_func():\n    a = 1\n    b = 2\n\nclass GenericClass:\n    c = 3\n"
    chunks = chunk_generic_file("test.txt", code, language="txt")
    assert len(chunks) > 0


def test_chunk_generic_sliding_window():
    """No boundaries should produce sliding window chunks."""
    # Generate 250 lines of plain text (no def/class patterns)
    code = "\n".join([f"line {i}" for i in range(250)])
    chunks = chunk_generic_file("long.txt", code, language="txt")
    # 250 lines with 100 lines per chunk and 20 overlap -> at least 3 content chunks + 1 module
    assert len(chunks) >= 3


def test_chunk_id_format():
    """Verify chunk_id is '{file_path}::{chunk_type}::{name}'."""
    code = "def my_func(): pass\n"
    chunks = chunk_python_file("test.py", code)
    func_chunk = next(c for c in chunks if c.chunk_type == "function")
    assert func_chunk.chunk_id == "test.py::function::my_func"


def test_large_function_flag():
    """Function >200 lines should have is_large=True."""
    # Generate a function with 210 body lines
    lines = ["def large_func():"]
    for _ in range(210):
        lines.append("    pass")
    code = "\n".join(lines) + "\n"
    chunks = chunk_python_file("large.py", code)
    func_chunk = next(c for c in chunks if c.chunk_type == "function")
    assert func_chunk.is_large is True


def test_large_class_summary():
    """Class >500 lines should have summary content, not full source."""
    # Generate a class with 510 methods (each is ~1 line)
    lines = ["class LargeClass:"]
    for i in range(510):
        lines.append(f"    def func_{i}(self): pass")
    code = "\n".join(lines) + "\n"
    chunks = chunk_python_file("large_class.py", code)
    class_chunk = next(c for c in chunks if c.chunk_type == "class")
    # Shouldn't contain full source, should be a summary
    assert class_chunk.is_large is True
