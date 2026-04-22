"""Сборка вариантов submission через замену отдельных Markdown-документов.

Скрипт берет базовую директорию с готовым submission и создает несколько
вариантов, в каждом из которых часть документов подменяется версиями из
альтернативных директорий. Также копируются только те изображения, на
которые реально ссылаются обновленные Markdown-файлы.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

DOC_RE = re.compile(r"document_(\d{3})\.md$")

DEFAULT_CONFIG = {
    "base_dir": "./output_v5_5",
    "variants": [
        {
            "name": "swap_probe_all_strong",
            "replacements": [
                {
                    "source_dir": "./output_all",
                    "docs": ["048", "085", "095"]
                }
            ]
        },
        {
            "name": "swap_probe_noocr_strong",
            "replacements": [
                {
                    "source_dir": "./output_v6_1_no_ocr",
                    "docs": ["069", "083", "086", "088", "096"]
                }
            ]
        },
        {
            "name": "swap_probe_mixed_strong",
            "replacements": [
                {
                    "source_dir": "./output_all",
                    "docs": ["085", "095"]
                },
                {
                    "source_dir": "./output_v6_1_no_ocr",
                    "docs": ["083", "086", "088"]
                }
            ]
        },
        {
            "name": "swap_probe_selector_full",
            "replacements": [
                {
                    "source_dir": "./output_all",
                    "docs": ["016", "019", "027", "034", "048", "085", "089", "095"]
                },
                {
                    "source_dir": "./output_v6_1_no_ocr",
                    "docs": ["001", "002", "004", "006", "007", "011", "024", "031", "041", "043", "052", "054", "069", "083", "086", "088", "094", "096"]
                }
            ]
        }
    ]
}


def load_config(path: Path) -> dict:
    """Загружает JSON-конфиг или создает дефолтный при отсутствии файла.

    Args:
        path: Путь до конфигурационного JSON-файла.

    Returns:
        Словарь с параметрами сборки вариантов.
    """
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Created default config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def copy_tree(src: Path, dst: Path) -> None:
    """Полностью копирует директорию с предварительным удалением приемника."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def doc_filename(doc_id: str) -> str:
    """Возвращает имя Markdown-файла по идентификатору документа."""
    return f"document_{int(doc_id):03d}.md"


def collect_referenced_images(md_path: Path) -> set[str]:
    """Собирает список ссылок на изображения из Markdown-файла."""
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    refs = set()
    for m in re.finditer(r"!\[[^\]]*\]\((images/[^)]+)\)", text):
        refs.add(m.group(1))
    return refs


def copy_needed_images(source_dir: Path, dest_dir: Path, md_path: Path) -> None:
    """Копирует только те изображения, которые реально используются в Markdown.

    Args:
        source_dir: Исходная директория варианта.
        dest_dir: Итоговая директория сборки.
        md_path: Markdown-файл, по которому нужно определить набор изображений.
    """
    refs = collect_referenced_images(md_path)
    for rel in refs:
        src = source_dir / rel
        dst = dest_dir / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def build_variant(base_dir: Path, variant: dict, dest_root: Path) -> Path:
    """Собирает один вариант submission по конфигурации замен.

    Args:
        base_dir: Базовая директория submission.
        variant: Блок конфигурации одного варианта.
        dest_root: Корневая директория для выходных вариантов.

    Returns:
        Путь к собранному варианту.
    """
    out_dir = dest_root / variant["name"]
    copy_tree(base_dir, out_dir)

    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    changed_docs: list[str] = []

    for block in variant["replacements"]:
        source_dir = Path(block["source_dir"])
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source dir not found: {source_dir}")

        for doc_id in block["docs"]:
            fname = doc_filename(doc_id)
            src_md = source_dir / fname
            dst_md = out_dir / fname

            if not src_md.exists():
                raise FileNotFoundError(f"Missing source markdown: {src_md}")

            shutil.copy2(src_md, dst_md)
            copy_needed_images(source_dir, out_dir, src_md)
            changed_docs.append(fname)

    print(f"{variant['name']}: {len(changed_docs)} replacements")
    return out_dir


def main() -> None:
    """Точка входа CLI для сборки всех submission-вариантов из конфига."""
    parser = argparse.ArgumentParser(description="Build submission variants by swapping selected documents.")
    parser.add_argument("--config", type=Path, default=Path("swap_config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("./submission_variants"))
    args = parser.parse_args()

    config = load_config(args.config)
    base_dir = Path(config["base_dir"])
    if not base_dir.is_dir():
        raise SystemExit(f"Base dir not found: {base_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    for variant in config["variants"]:
        build_variant(base_dir=base_dir, variant=variant, dest_root=args.output_root)

    print("\nDone.")
    print(f"Variants are in: {args.output_root}")
    print("\nZip example:")
    print("cd ./submission_variants/swap_probe_mixed_strong && zip -r ../../swap_probe_mixed_strong.zip ./* && cd -")


if __name__ == "__main__":
    main()
