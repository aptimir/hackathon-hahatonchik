"""Собирает финальный best-shot submission на основе базового варианта и marker-замен.

Скрипт копирует базовый submission в новую директорию, а затем подменяет
выбранные документы Markdown-файлами из marker-выгрузок, которые показали
лучшее качество при ручной проверке. Для обновленных документов дополнительно
копируются только реально используемые изображения.

Сценарий нужен для быстрой сборки финального варианта без ручной правки
каждого документа.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

BASE_DIR = Path("./submission_variants/swap_probe_mixed_strong")
DEST_DIR = Path("./submission_final_bestshot")

MARKER_ROOTS = {
    "029": Path("./marker_test/document_029"),
    "085": Path("./marker_test/document_085"),
    "095": Path("./marker_test/document_095"),
}

IMAGE_RE = re.compile(r"!\[[^\]]*\]\((images/[^)]+)\)")


def find_marker_md(root: Path) -> Path | None:
    """Находит основной Markdown-файл внутри marker-директории.

    Args:
        root: Корневая директория конкретной marker-выгрузки.

    Returns:
        Путь к наиболее глубокому Markdown-файлу, если он найден,
        иначе None.
    """
    if not root.exists():
        return None
    matches = list(root.rglob("*.md")) + list(root.rglob("*.markdown"))
    if not matches:
        return None
    matches.sort(key=lambda p: (len(p.parts), len(str(p))), reverse=True)
    return matches[0]


def clean_marker_text(text: str) -> str:
    """Очищает marker-markdown от повторяющегося служебного мусора.

    Args:
        text: Исходный Markdown-текст из marker-выгрузки.

    Returns:
        Очищенный Markdown с нормализованными переводами строк и
        схлопнутыми пустыми абзацами.
    """
    bad_line_patterns = [
        r"^\s*DRAFT\s*$",
        r"^\s*НЕ ДЛЯ РАСПРОСТРАНЕНИЯ\s*$",
        r"^\s*.+ · .+ · \d{4}-\d{2}-\d{2}\s*(стр\.\s*\d+)?\s*$",
        r"^\s*(Page\s+\d+|стр\.\s*\d+|—\s*\d+\s*—|\[\s*\d+\s*\]|\d+\s*/\s*\?)\s*$",
    ]

    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    out: list[str] = []

    for line in lines:
        s = line.strip()

        if any(re.fullmatch(pat, s) for pat in bad_line_patterns):
            continue

        out.append(line.rstrip())

    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"
    return cleaned


def copy_referenced_images(md_path: Path, dest_root: Path) -> None:
    """Копирует в итоговую директорию только те изображения, которые есть в Markdown.

    Args:
        md_path: Путь к Markdown-файлу marker-версии.
        dest_root: Корневая директория итогового submission.
    """
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    refs = IMAGE_RE.findall(text)
    for rel in refs:
        src = md_path.parent / rel
        if src.exists():
            dst = dest_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main() -> None:
    """Точка входа для сборки финального best-shot submission."""
    if not BASE_DIR.exists():
        raise SystemExit(f"Base dir not found: {BASE_DIR}")

    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    shutil.copytree(BASE_DIR, DEST_DIR)
    (DEST_DIR / "images").mkdir(parents=True, exist_ok=True)

    replaced: list[str] = []
    skipped: list[str] = []

    for doc_id, marker_root in MARKER_ROOTS.items():
        marker_md = find_marker_md(marker_root)
        if marker_md is None:
            skipped.append(doc_id)
            continue

        text = marker_md.read_text(encoding="utf-8", errors="ignore")
        text = clean_marker_text(text)

        out_md = DEST_DIR / f"document_{doc_id}.md"
        out_md.write_text(text, encoding="utf-8")

        copy_referenced_images(marker_md, DEST_DIR)
        replaced.append(doc_id)

    print("Base:", BASE_DIR)
    print("Output:", DEST_DIR)
    print("Replaced with marker:", ", ".join(replaced) if replaced else "none")
    print("Skipped:", ", ".join(skipped) if skipped else "none")
    print("\nZip it with:")
    print("cd ./submission_final_bestshot && zip -r ../submission_final_bestshot.zip ./* && cd -")


if __name__ == "__main__":
    main()