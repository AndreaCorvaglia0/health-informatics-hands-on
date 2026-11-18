#!/usr/bin/env python3
"""Execute a notebook top to bottom (smoke test) without modifying it.

Usage::

    HI_DATA_DIR=/path/to/dataset HI_FAST=1 python tools/run_notebook.py solutions/04_*.ipynb

Environment variables understood by the course notebooks:

* ``HI_DATA_DIR``  - folder that contains ``images/``, ``labels/``, ``metadata.csv``
* ``HI_FAST=1``    - "debug run": tiny images, few samples, 1-2 epochs (CPU friendly)
* ``HI_WORK_DIR``  - where notebooks write models/outputs (default: ./outputs)

Use ``--save`` to write the executed copy next to the original (``*.executed.ipynb``).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+", type=Path)
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds")
    ap.add_argument("--save", action="store_true", help="save executed copy as *.executed.ipynb")
    ap.add_argument("--cwd", type=Path, default=None, help="working directory for the kernel")
    args = ap.parse_args(argv)

    failed = 0
    for path in args.notebooks:
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                                resources={"metadata": {"path": str(args.cwd or path.parent)}})
        t0 = time.time()
        try:
            client.execute()
            print(f"OK    {path}  ({time.time() - t0:.0f}s)")
        except CellExecutionError as err:
            failed += 1
            print(f"FAIL  {path}  ({time.time() - t0:.0f}s)\n{str(err)[-3000:]}")
        if args.save:
            out = path.with_suffix(".executed.ipynb")
            nbformat.write(nb, out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
