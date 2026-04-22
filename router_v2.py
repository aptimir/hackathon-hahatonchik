"""Правила выбора лучшей версии Markdown между несколькими пайплайнами.

Скрипт сравнивает три варианта одного и того же документа:
- базовую версию
- версию v5.5
- версию без OCR

После анализа нескольких простых метрик качества выбирается один итоговый
Markdown-файл, а также копируются все изображения, на которые он ссылается.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

DRAFT_RE = re.compile(r"\b(?:DRAFT|ЧЕРНОВИК)\b", flags=re.IGNORECASE)
CAPTION_RE = re.compile(r"^\s*Рис\.\s*\d+", flags=re.IGNORECASE | re.MULTILINE)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\((images/[^)]+)\)")
EMPTY_TABLE_RE = re.compile(
    r"(?:^|\n)(?:\|\s*(?:\|\s*)+\n\|\s*(?:[-:]+\s*\|)+)(?=\n|$)",
    flags=re.MULTILINE,
)
ONE_WORD_LINE_RE = re.compile(r"^[A-Za-zА-Яа-яЁё0-9._/%:+\-–—≤≥≈°]+$")
HEADING_RE = re.compile(r"^\s*#+\s+", flags=re.MULTILINE)


def analyze(text: str) -> dict[str, float]:
    """Собирает набор простых метрик качества по Markdown-тексту.

    Args:
        text: Markdown-текст документа.

    Returns:
        Словарь с метриками, которые используются в rule-based выборе.
    """
    lines = text.splitlines()
    nonempty = [ln.strip() for ln in lines if ln.strip()]

    image_count = len(IMAGE_RE.findall(text))
    draft_count = len(DRAFT_RE.findall(text))
    caption_count = len(CAPTION_RE.findall(text))
    empty_table_count = len(EMPTY_TABLE_RE.findall(text))
    heading_count = len(HEADING_RE.findall(text))

    one_word_lines = 0
    short_fragment_lines = 0
    alpha = 0

    for ln in nonempty:
        alpha += sum(ch.isalpha() for ch in ln)
        words = ln.split()

        if len(words) == 1 and ONE_WORD_LINE_RE.fullmatch(ln):
            one_word_lines += 1

        if (
            len(words) <= 2
            and len(ln) <= 18
            and not ln.startswith("#")
            and not ln.startswith("![")
            and "|" not in ln
            and not ln.endswith((".", "!", "?"))
        ):
            short_fragment_lines += 1

    return {
        "text_len": len(text),
        "alpha": alpha,
        "image_count": image_count,
        "draft_count": draft_count,
        "caption_count": caption_count,
        "empty_table_count": empty_table_count,
        "heading_count": heading_count,
        "one_word_lines": one_word_lines,
        "short_fragment_lines": short_fragment_lines,
        "nonempty_lines": len(nonempty),
    }


def pick_version(
    doc_name: str,
    base_text: str,
    v55_text: str,
    noocr_text: str,
) -> str:
    """Выбирает лучшую версию документа по набору эвристик.

    Args:
        doc_name: Имя документа. Используется для логического контекста.
        base_text: Базовая версия Markdown.
        v55_text: Версия от пайплайна `v5.5`.
        noocr_text: Версия без OCR.

    Returns:
        Один из маркеров: `base`, `v55` или `noocr`.
    """
    base = analyze(base_text)
    v55 = analyze(v55_text)
    noocr = analyze(noocr_text)

    base_len = max(base["text_len"], 1)
    v55_ratio = v55["text_len"] / base_len
    noocr_ratio = noocr["text_len"] / base_len

    choice = "v55"

    if (
        v55_ratio < 0.72
        and base["text_len"] - v55["text_len"] > 1200
        and base["caption_count"] <= v55["caption_count"] + 2
    ):
        return "base"

    if (
        base["text_len"] - v55["text_len"] > 1800
        and base["one_word_lines"] <= v55["one_word_lines"] + 10
    ):
        return "base"

    noocr_cleaner = (
        noocr["draft_count"] <= v55["draft_count"]
        and noocr["caption_count"] <= v55["caption_count"]
        and noocr["empty_table_count"] <= v55["empty_table_count"]
        and noocr["one_word_lines"] + noocr["short_fragment_lines"]
            < v55["one_word_lines"] + v55["short_fragment_lines"] - 8
    )

    noocr_not_too_short = (
        noocr_ratio >= 0.82
        or (noocr["text_len"] >= v55["text_len"] - 500)
    )

    noocr_structure_ok = (
        noocr["image_count"] >= max(v55["image_count"] - 1, 0)
        and noocr["heading_count"] >= max(v55["heading_count"] - 2, 0)
    )

    if noocr_cleaner and noocr_not_too_short and noocr_structure_ok:
        choice = "noocr"

    if noocr_ratio < 0.70 and v55["text_len"] >= noocr["text_len"] + 700:
        choice = "v55"

    if noocr["text_len"] < 1500 and v55["text_len"] > 3000:
        choice = "v55"

    if (
        v55["caption_count"] + v55["draft_count"] > noocr["caption_count"] + noocr["draft_count"] + 2
        and noocr["text_len"] >= v55["text_len"] * 0.9
    ):
        choice = "noocr"

    return choice


def main() -> None:
    """Точка входа CLI для выбора лучшей версии каждого документа."""
    parser = argparse.ArgumentParser(description="Rule-based router for best markdown output.")
    parser.add_argument("--output-all", required=True)
    parser.add_argument("--output-v55", required=True)
    parser.add_argument("--output-noocr", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_all = Path(args.output_all)
    out_v55 = Path(args.output_v55)
    out_noocr = Path(args.output_noocr)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "images").mkdir(exist_ok=True)

    doc_names = sorted(p.name for p in out_v55.glob("document_*.md"))
    counts = {"base": 0, "v55": 0, "noocr": 0}

    for doc_name in doc_names:
        p_base = out_all / doc_name
        p_v55 = out_v55 / doc_name
        p_noocr = out_noocr / doc_name

        if not p_base.exists() or not p_v55.exists() or not p_noocr.exists():
            continue

        base_text = p_base.read_text(encoding="utf-8", errors="ignore")
        v55_text = p_v55.read_text(encoding="utf-8", errors="ignore")
        noocr_text = p_noocr.read_text(encoding="utf-8", errors="ignore")

        choice = pick_version(doc_name, base_text, v55_text, noocr_text)
        counts[choice] += 1

        if choice == "base":
            chosen = p_base
        elif choice == "noocr":
            chosen = p_noocr
        else:
            chosen = p_v55

        shutil.copy2(chosen, dest / doc_name)

        text = chosen.read_text(encoding="utf-8", errors="ignore")
        for rel in IMAGE_RE.findall(text):
            src = chosen.parent / rel
            if src.exists():
                dst = dest / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)

        if args.verbose:
            print(f"{doc_name} -> {choice}")

    print("Chosen counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
