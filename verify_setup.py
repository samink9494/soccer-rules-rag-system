#!/usr/bin/env python3
"""Check local files and dependencies needed by the Soccer Rules RAG notebook."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
REQUIRED_MODULES = {
    "chromadb": "chromadb",
    "sentence_transformers": "sentence-transformers",
    "requests": "requests",
}


def check_ollama() -> bool:
    """Check that the local Ollama API responds and exposes the mistral model."""
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            payload = json.load(response)
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"FAIL Ollama: could not reach http://127.0.0.1:11434 ({exc})")
        return False

    models = [model.get("name", "") for model in payload.get("models", [])]
    if any(name == "mistral" or name.startswith("mistral:") for name in models):
        print("PASS Ollama: local API is responding and mistral is available")
        return True
    print("FAIL Ollama: API is responding, but mistral is not installed (run: ollama pull mistral)")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="also check the local Ollama API and the mistral model",
    )
    args = parser.parse_args()

    success = True
    docs = sorted(ROOT.glob("Document *.txt"))
    if len(docs) == 10:
        print(f"PASS documents: found all {len(docs)} rule text files")
    else:
        print(f"FAIL documents: expected 10 'Document *.txt' files, found {len(docs)}")
        success = False

    for module, package in REQUIRED_MODULES.items():
        if importlib.util.find_spec(module) is None:
            print(f"FAIL dependency: {package} is not installed (run: python -m pip install -r requirements.txt)")
            success = False
        else:
            print(f"PASS dependency: {package}")

    if args.ollama and not check_ollama():
        success = False
    elif not args.ollama:
        print("SKIP Ollama: not checked (use --ollama after starting Ollama and pulling mistral)")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
