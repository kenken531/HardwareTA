#!/usr/bin/env python3
"""
main.py — HardwareTA: Local RAG agent for hardware datasheets

Usage:
  python main.py ingest            # parse PDFs → build vector DB
  python main.py ask "question"    # retrieve + answer
  python main.py demo              # run a demo Q&A session
  python main.py download          # download sample ESP32 datasheet
"""

import sys
import textwrap
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def cmd_download():
    from download_sample import main
    main()


def cmd_ingest():
    from ingest import ingest_pdfs
    n = ingest_pdfs()
    if n == 0:
        print("\n[!] Nothing indexed. Run first:\n    python main.py download")


def cmd_ask(question: str):
    from query import answer
    answer(question)


def cmd_demo():
    questions = [
        "What is the maximum operating voltage and current consumption?",
        "What communication interfaces does this chip support?",
        "What is the operating temperature range?",
        "Describe the power consumption in deep sleep mode.",
        "What are the GPIO pin electrical characteristics?",
    ]

    print("\n" + "═" * 68)
    print("  HardwareTA — Demo Q&A Session")
    print("═" * 68)

    from query import answer
    for i, q in enumerate(questions):
        print(f"\n[Question {i+1}/{len(questions)}]")
        result = answer(q)
        # Pause between questions
        if i < len(questions) - 1:
            input("\n  Press ENTER for next question …")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "download":
        cmd_download()
    elif cmd == "ingest":
        cmd_ingest()
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print("Usage: python main.py ask \"your question here\"")
            sys.exit(1)
        cmd_ask(" ".join(sys.argv[2:]))
    elif cmd == "demo":
        cmd_demo()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
