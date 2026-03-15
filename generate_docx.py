import re
import subprocess
import tempfile
from pathlib import Path

import pooch

import utils

manuscript_dir = utils.config["manuscript_directory"]
csl_fname = "word/elsevier-vancouver.csl"

csl_filepath = pooch.retrieve(
    url="https://raw.githubusercontent.com/citation-style-language/styles/refs/heads/master/elsevier-vancouver.csl",
    known_hash="md5:da957ad48fa938b4237eb87e2ad1a488",
    path=manuscript_dir,
    fname=csl_fname,
)

tex = (manuscript_dir / "main.tex").read_text()
tex = re.sub(r"\\supercite", r"\\cite", tex)

with tempfile.NamedTemporaryFile(
    mode="w",
    suffix=".tex",
    delete=False,
    encoding="utf-8",
    dir=manuscript_dir,
) as tmp:
    tmp.write(tex)
    tmp_path = Path(tmp.name)

try:
    subprocess.run([
        "pandoc", tmp_path,
        "--from=latex",
        "--to=docx",
        "--citeproc",
        "--bibliography", "references.bib",
        "--csl", csl_fname,
        "--number-sections",
        "--reference-doc", "word/reference.docx",
        "-o", "word/main.docx",
    ], check=True, cwd=manuscript_dir)
finally:
    tmp_path.unlink()
