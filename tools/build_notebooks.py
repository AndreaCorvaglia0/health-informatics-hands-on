#!/usr/bin/env python3
"""Build student and solution notebooks from the single sources in ``src/``.

Each notebook is authored ONCE as a Python file in "percent" format
(``src/NN_name.py``).  This script produces two notebooks from it:

* ``notebooks/NN_name.ipynb``            -> what students receive (exercises to fill in)
* ``solutions/NN_name_solution.ipynb``   -> the full walk-through for teachers

Source syntax
-------------
Cells::

    # %% [markdown]
    # Markdown text, every line prefixed with "# " (or a bare "#" for an empty line).

    # %%
    code_cell = 1

    # %% tags=["solution-only"]          <- cell kept only in the solution notebook
    # %% [markdown] tags=["student-only"] <- cell kept only in the student notebook

Exercise blocks inside a code cell::

    #@solution
    loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    #@student
    #| loader = ____  # TODO: build the training DataLoader
    #@end

The solution notebook keeps the lines after ``#@solution``; the student notebook
keeps the lines after ``#@student`` with the ``#| `` prefix removed.  The
``#@student`` part is optional (the block then simply disappears for students).
Indentation of the marker lines is ignored; the ``#| `` prefix is replaced by
nothing, so indentation written *before* ``#|`` is preserved.

Usage::

    python tools/build_notebooks.py            # build everything in src/
    python tools/build_notebooks.py src/03_*.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL_RE = re.compile(r"^# %%(?P<rest>.*)$")
TAGS_RE = re.compile(r"tags\s*=\s*(\[[^\]]*\])")

KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}


def parse_cells(text: str):
    """Yield (cell_type, tags, lines) for each percent cell."""
    cells, current = [], None
    for line in text.splitlines():
        m = CELL_RE.match(line)
        if m:
            rest = m.group("rest")
            ctype = "markdown" if "[markdown]" in rest else "code"
            tags_m = TAGS_RE.search(rest)
            tags = json.loads(tags_m.group(1)) if tags_m else []
            current = {"type": ctype, "tags": tags, "lines": []}
            cells.append(current)
        elif current is not None:
            current["lines"].append(line)
        elif line.strip():
            # header before the first cell marker (e.g. module docstring): ignore
            continue
    return cells


def strip_blank_edges(lines):
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def markdown_lines(lines):
    out = []
    for line in lines:
        if line.startswith("# "):
            out.append(line[2:])
        elif line.rstrip() == "#":
            out.append("")
        elif not line.strip():
            out.append("")
        else:
            raise ValueError(f"Markdown line must start with '# ': {line!r}")
    return strip_blank_edges(out)


def code_lines(lines, variant: str, src_name: str):
    out, mode = [], "both"
    for n, line in enumerate(lines, 1):
        marker = line.strip()
        if marker == "#@solution":
            if mode != "both":
                raise ValueError(f"{src_name}: nested #@solution")
            mode = "solution"
        elif marker == "#@student":
            if mode != "solution":
                raise ValueError(f"{src_name}: #@student outside a #@solution block")
            mode = "student"
        elif marker == "#@end":
            if mode == "both":
                raise ValueError(f"{src_name}: #@end without #@solution")
            mode = "both"
        elif mode == "both":
            out.append(line)
        elif mode == "solution":
            if variant == "solution":
                out.append(line)
        elif mode == "student":
            if variant == "student":
                if "#|" not in line:
                    if line.strip():
                        raise ValueError(f"{src_name}: student line must start with '#| ': {line!r}")
                    out.append("")
                else:
                    indent, _, rest = line.partition("#|")
                    out.append(indent + (rest[1:] if rest.startswith(" ") else rest))
    if mode != "both":
        raise ValueError(f"{src_name}: unterminated #@solution block")
    return strip_blank_edges(out)


def to_source(lines):
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])


def build(src: Path, variant: str) -> dict:
    cells = []
    for i, cell in enumerate(parse_cells(src.read_text(encoding="utf-8"))):
        tags = cell["tags"]
        if "solution-only" in tags and variant != "solution":
            continue
        if "student-only" in tags and variant != "student":
            continue
        keep_tags = [t for t in tags if t not in ("solution-only", "student-only")]
        meta = {"tags": keep_tags} if keep_tags else {}
        if cell["type"] == "markdown":
            lines = markdown_lines(cell["lines"])
            if not lines:
                continue
            cells.append({"cell_type": "markdown", "id": f"c{i:03d}", "metadata": meta,
                          "source": to_source(lines)})
        else:
            lines = code_lines(cell["lines"], variant, src.name)
            if not lines:
                continue
            cells.append({"cell_type": "code", "id": f"c{i:03d}", "metadata": meta,
                          "execution_count": None, "outputs": [], "source": to_source(lines)})
    return {
        "cells": cells,
        "metadata": {"kernelspec": KERNELSPEC,
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sources", nargs="*", type=Path, help="source files (default: src/*.py)")
    args = ap.parse_args(argv)
    sources = args.sources or sorted((ROOT / "src").glob("*.py"))
    if not sources:
        print("no sources found", file=sys.stderr)
        return 1
    for src in sources:
        for variant, folder, suffix in (("student", "notebooks", ""), ("solution", "solutions", "_solution")):
            nb = build(src, variant)
            out = ROOT / folder / f"{src.stem}{suffix}.ipynb"
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
            print(f"{out.relative_to(ROOT)}: {len(nb['cells'])} cells ({n_code} code)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
