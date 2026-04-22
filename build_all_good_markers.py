"""Собирает итоговый submission, подменяя выбранные документы marker-версиями.

Скрипт берет базовую директорию с уже собранным submission и заменяет в ней
часть Markdown-файлов версиями из marker-выгрузок, которые вручную были
признаны более качественными. Для подменённых документов также копируются
только реально используемые изображения.

Основные задачи:
- найти основной Markdown-файл внутри marker-директории;
- очистить marker-текст от служебного мусора;
- скопировать только те картинки, на которые есть ссылки в Markdown;
- собрать новый submission в отдельную директорию.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable

BASE = Path("./submission_final_bestshot_v2")
OUT = Path("./submission_final_bestshot_all_good_markers")

MARKER_ROOTS: dict[str, Path] = {
    "031": Path("./marker_batch4_out/document_031"),
    "054": Path("./marker_batch4_out/document_054"),
    "085": Path("./marker_test/document_085"),
    "094": Path("./marker_batch3_out/document_094"),
    "095": Path("./marker_test/document_095"),
    "096": Path("./marker_batch3_out/document_096"),
}

IMAGE_RE = re.compile(r"!\[[^\]]*\]\((images/[^)]+)\)")

BAD_LINE_PATTERNS: tuple[str, ...] = (
    r"^\s*DRAFT\s*$",
    r"^\s*НЕ ДЛЯ РАСПРОСТРАНЕНИЯ\s*$",
    r"^\s*КОНФИДЕНЦИАЛЬНО\s*$",
    r"^\s*.+ · .+ · \d{4}-\d{2}-\d{2}\s*(стр\.\s*\d+)?\s*$",
    r"^\s*(Page\s+\d+|стр\.\s*\d+|—\s*\d+\s*—|\[\s*\d+\s*\]|\d+\s*/\s*\?)\s*$",
)


def _find_markdown_file(root: Path) -> Path | None:
    """Возвращает самый глубокий Markdown-файл в директории marker-выгрузки."""
    if not root.exists():
        return None

    files = list(root.rglob("*.md")) + list(root.rglob("*.markdown"))
    if not files:
        return None

    files.sort(key=lambda path: (len(path.parts), len(str(path))), reverse=True)
    return files[0]


def _iter_clean_lines(text: str) -> Iterable[str]:
    """Итерирует очищенные строки, удаляя известный служебный мусор."""
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")

    for line in normalized_text.splitlines():
        stripped = line.strip()
        if any(re.fullmatch(pattern, stripped) for pattern in BAD_LINE_PATTERNS):
            continue
        yield line.rstrip()


def _clean_text(text: str) -> str:
    """Нормализует marker-markdown перед копированием в итоговый submission."""
    cleaned = "\n".join(_iter_clean_lines(text))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return f"{cleaned}\n"


def _copy_images(md_path: Path, destination_root: Path) -> int:
    """Копирует изображения, на которые есть ссылки в marker-markdown."""
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    copied = 0

    for relative_path in IMAGE_RE.findall(text):
        source = md_path.parent / relative_path
        if not source.exists():
            continue

        target = destination_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1

    return copied


def main() -> None:
    """Собирает итоговый submission, накладывая лучшие marker-файлы поверх базы."""
    if not BASE.exists():
        raise FileNotFoundError(f"Base folder not found: {BASE}")

    if OUT.exists():
        shutil.rmtree(OUT)

    shutil.copytree(BASE, OUT)
    (OUT / "images").mkdir(parents=True, exist_ok=True)

    replaced: list[str] = []
    skipped: list[str] = []
    total_images = 0

    for doc_id, root in MARKER_ROOTS.items():
        markdown_path = _find_markdown_file(root)
        if markdown_path is None:
            skipped.append(doc_id)
            continue

        cleaned_text = _clean_text(
            markdown_path.read_text(encoding="utf-8", errors="ignore")
        )
        output_file = OUT / f"document_{doc_id}.md"
        output_file.write_text(cleaned_text, encoding="utf-8")

        total_images += _copy_images(markdown_path, OUT)
        replaced.append(doc_id)

    print(f"Base: {BASE}")
    print(f"Output: {OUT}")
    print(f"Replaced: {', '.join(replaced) if replaced else 'none'}")
    print(f"Skipped: {', '.join(skipped) if skipped else 'none'}")
    print(f"Copied images: {total_images}")


if __name__ == "__main__":
    main()