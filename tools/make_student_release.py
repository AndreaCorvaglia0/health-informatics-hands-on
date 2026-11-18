#!/usr/bin/env python3
"""Assemble what students receive (NO solutions) into ``dist/student`` and zip it.

    python tools/make_student_release.py            # -> dist/student/ and dist/student.zip

Contents: student notebooks, the day-5 starter app and guide, the setup guide, the
syllabus.  Publish it as a separate PUBLIC GitHub repository (so that "Open in Colab"
links work and Streamlit Community Cloud can deploy from students' forks), and/or
upload the notebooks to Kaggle.  Never make the teachers' repository public.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "student"

STUDENT_README = """# Health Informatics - Advanced ML hands-on (student material)

From a CT scan to a deployed web app in 5 days.

1. Do the technical setup: `docs/SETUP.md`
2. Follow the notebooks in order (`notebooks/01 ... 06`), on Kaggle (or Colab)
3. Day 5: build and deploy the web app, guide in `app/README.md`
4. `SYLLABUS.md` lists every topic with references to study further

Legend used everywhere: ✏️ mandatory exercise (hidden hints below it) - 🔍 explore -
🚀 additional challenge - 🌶️ optional, open-ended.

Education and research only: nothing here is a medical device.
"""


def main():
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_notebooks.py")], check=True)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "notebooks").mkdir(parents=True)
    for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
        shutil.copy2(nb, OUT / "notebooks" / nb.name)
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc", ".venv")
    starter = OUT / "app" / "starter"
    shutil.copytree(ROOT / "app" / "starter", starter, ignore=ignore)
    shutil.copy2(ROOT / "app" / "README.md", OUT / "app" / "README.md")
    # students also need the Dockerfile, the Streamlit config and the inference tests on day 5
    # (tests/test_app.py is written for the finished app: they get it from the teachers at the end)
    sol = ROOT / "app" / "solution"
    for extra in ("Dockerfile", ".dockerignore"):
        shutil.copy2(sol / extra, starter / extra)
    shutil.copytree(sol / ".streamlit", starter / ".streamlit", dirs_exist_ok=True)
    (starter / "tests").mkdir(exist_ok=True)
    for t in ("conftest.py", "test_inference.py"):
        if (sol / "tests" / t).exists():
            shutil.copy2(sol / "tests" / t, starter / "tests" / t)
    (OUT / "docs").mkdir()
    shutil.copy2(ROOT / "docs" / "SETUP.md", OUT / "docs" / "SETUP.md")
    shutil.copy2(ROOT / "SYLLABUS.md", OUT / "SYLLABUS.md")
    (OUT / "README.md").write_text(STUDENT_README, encoding="utf-8")

    leaked = [p for p in OUT.rglob("*") if "solution" in p.name.lower()]
    if leaked:
        raise SystemExit(f"solution files leaked into the student release: {leaked}")
    archive = shutil.make_archive(str(OUT), "zip", OUT)
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"{n} files -> {OUT}\n{archive}")


if __name__ == "__main__":
    main()
