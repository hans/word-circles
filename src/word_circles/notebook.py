"""Execute a jupytext notebook with parameters (ploomber-engine) and save the executed .ipynb."""

import argparse
import json
import tempfile
from pathlib import Path

import jupytext
from ploomber_engine import execute_notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="jupytext percent-format .py notebook")
    parser.add_argument("output", type=Path, help="executed .ipynb")
    parser.add_argument("--parameters", type=json.loads, default={})
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ipynb = Path(tmp) / f"{args.source.stem}.ipynb"
        jupytext.write(jupytext.read(args.source), ipynb)
        execute_notebook(ipynb, args.output, parameters=args.parameters, log_output=False)


if __name__ == "__main__":
    main()
