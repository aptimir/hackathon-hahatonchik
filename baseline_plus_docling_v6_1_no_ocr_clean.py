"""Конвертер PDF в Markdown на базе Docling с приоритетом no-OCR режима.

Скрипт обрабатывает PDF-документы двумя режимами Docling — без OCR и с OCR —
но в отличие от более ранней версии сначала пытается использовать no-OCR
вариант и обращается к OCR только как к запасному варианту. После извлечения
Markdown применяется набор эвристик для очистки шума, нормализации таблиц,
исправления ссылок на изображения и подавления артефактов OCR.

Основные задачи:
- сохранить Markdown и изображения в формате, совместимом с требованиями;
- уменьшить количество шумовых строк и мусорных блоков;
- аккуратно нормализовать таблицы и битые ссылки на картинки;
- использовать OCR только там, где no-OCR не справляется.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")


def _apply_device_from_argv() -> None:
    """Устанавливает `DOCLING_DEVICE` из аргументов командной строки.

    Переменная окружения должна быть выставлена до импорта Docling.
    Если в аргументах указан `--device auto`, переменная не переопределяется.
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            value = sys.argv[i + 1]
            if value != "auto":
                os.environ["DOCLING_DEVICE"] = value
            return
        if arg.startswith("--device="):
            value = arg.split("=", 1)[1]
            if value != "auto":
                os.environ["DOCLING_DEVICE"] = value
            return


_apply_device_from_argv()


def _patch_cv2_set_num_threads() -> None:
    """Добавляет заглушку `cv2.setNumThreads`, если метода нет.

    Некоторые внутренние части пайплайна Docling ожидают наличие этой
    функции. Если OpenCV установлен в урезанной сборке, метод может
    отсутствовать.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        return
    if not hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads = lambda _nthreads: None  # type: ignore[attr-defined,method-assign]


_patch_cv2_set_num_threads()

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    TableFormerMode,
    TableStructureOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.base import ImageRefMode

_IMG_LINK_RE = re.compile(
    r"images/(image_\d+_[a-f0-9]+\.(?:png|jpe?g))",
    flags=re.IGNORECASE,
)
_MD_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\((images/[^)]+)\)")
_DRAFT_RE = re.compile(r"\b(?:ЧЕРНОВИК|DRAFT)\b", flags=re.IGNORECASE)
_CAPTION_RE = re.compile(r"^\s*Рис\.\s*\d+\..*$", flags=re.IGNORECASE)
_IMAGE_WORD_RE = re.compile(r"^\s*(?:Image|Картинка)\s*$", flags=re.IGNORECASE)
_SINGLE_GARBAGE_LINE_RE = re.compile(
    r"^\s*(?:\.{2,}\d*\.?|\d+[a-zA-Z]?|[a-zA-Z]\d+|\d+[a-zA-Z]+|[a-zA-Z]{1,2})\s*$"
)
_PERCENT_ONLY_RE = re.compile(r"^\s*\d+(?:[.,]\d+)?%\s*$")
_NUMBER_ONLY_RE = re.compile(r"^\s*[\d\s.,:;%°()\-–—/\\+×÷≤≥≈←→]+\s*$")
_EMPTY_TABLE_RE = re.compile(
    r"(?:^|\n)(\|\s*\|\s*\n\|\s*[-: ]+\|\s*)(?=\n|$)",
    flags=re.MULTILINE,
)
_MULTIBLANK_RE = re.compile(r"\n{3,}")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")
_HEADINGISH_RE = re.compile(r"^(Раздел:|Глава\s*-)", flags=re.IGNORECASE)


class QualityStats:
    """Собирает простые эвристики качества по готовому markdown-тексту.

    Атрибуты используются для последующего вычисления итогового score.
    Чем выше score, тем вероятнее, что markdown более чистый и структурный.
    """

    def __init__(self, text: str) -> None:
        """Инициализирует набор метрик для оценки Markdown.

        Args:
            text: Готовый Markdown-текст документа.
        """
        self.text_length = len(text)
        self.lines = text.splitlines()
        self.image_refs = len(_MD_IMAGE_REF_RE.findall(text))
        self.headers = sum(1 for line in self.lines if line.lstrip().startswith("#"))
        self.tables = text.count("| ---") + text.count("|---")
        self.bad_markers = len(_DRAFT_RE.findall(text))
        self.garbage_lines = sum(
            1
            for line in self.lines
            if _is_obvious_noise_line(line.strip())
        )
        self.empty_tables = len(_EMPTY_TABLE_RE.findall(text))

    def score(self) -> float:
        """Возвращает суммарный эвристический score качества Markdown."""
        score = 0.0
        score += min(self.text_length / 400.0, 40.0)
        score += self.headers * 1.5
        score += self.tables * 2.0
        score += self.image_refs * 0.4
        score -= self.bad_markers * 8.0
        score -= self.garbage_lines * 1.5
        score -= self.empty_tables * 6.0
        return score


def _clear_cuda_cache() -> None:
    """Очищает CUDA-кэш PyTorch, если GPU доступен."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def _doc_num_from_stem(stem: str) -> int:
    """Извлекает номер документа из имени вида `document_NNN`.

    Args:
        stem: Имя файла без расширения.

    Returns:
        Номер документа без ведущих нулей.

    Raises:
        ValueError: Если формат имени не соответствует ожидаемому.
    """
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid file stem: {stem}")
    return int(parts[1])


def _move_or_convert_to_png(src: Path, dst: Path) -> None:
    """Перемещает изображение в PNG или конвертирует JPG/JPEG в PNG.

    Args:
        src: Исходный путь к изображению.
        dst: Итоговый путь к PNG-файлу.
    """
    ext = src.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        from PIL import Image

        with Image.open(src) as image:
            image.save(dst, format="PNG")
        src.unlink()
        return
    shutil.move(str(src), str(dst))


def _normalize_image_names(
    markdown: str,
    work_images_dir: Path,
    out_images_dir: Path,
    doc_num: int,
) -> str:
    """Нормализует имена изображений под формат из ТЗ.

    Все найденные изображения переименовываются в формат
    `doc_<id>_image_<order>.png`, а ссылки внутри Markdown обновляются.

    Args:
        markdown: Исходный Markdown.
        work_images_dir: Временная директория с артефактами Docling.
        out_images_dir: Итоговая директория `images/`.
        doc_num: Номер документа.

    Returns:
        Markdown с обновленными ссылками на изображения.
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    old_to_new: dict[str, str] = {}
    order = 1

    for match in _IMG_LINK_RE.finditer(markdown):
        old_name = match.group(1)
        if old_name in old_to_new:
            continue

        src = work_images_dir / old_name
        if not src.is_file():
            continue

        new_name = f"doc_{doc_num}_image_{order}.png"
        old_to_new[old_name] = new_name
        _move_or_convert_to_png(src, out_images_dir / new_name)
        order += 1

    normalized = markdown
    for old_name, new_name in sorted(old_to_new.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = normalized.replace(f"images/{old_name}", f"images/{new_name}")
    return normalized


def _remove_broken_image_refs(markdown: str, out_images_dir: Path) -> str:
    """Удаляет ссылки на изображения, которых нет в итоговой папке.

    Args:
        markdown: Markdown-текст документа.
        out_images_dir: Итоговая директория с изображениями.

    Returns:
        Markdown без битых ссылок на картинки.
    """
    def _replace(match: re.Match[str]) -> str:
        rel_path = match.group(1)
        image_name = Path(rel_path).name
        if (out_images_dir / image_name).is_file():
            return match.group(0)
        return ""

    return _MD_IMAGE_REF_RE.sub(_replace, markdown)


def _is_obvious_noise_line(stripped: str) -> bool:
    """Проверяет, похожа ли строка на шум или мусор."""
    if not stripped:
        return False
    if _DRAFT_RE.fullmatch(stripped):
        return True
    if _CAPTION_RE.fullmatch(stripped):
        return True
    if _IMAGE_WORD_RE.fullmatch(stripped):
        return True
    if _SINGLE_GARBAGE_LINE_RE.fullmatch(stripped):
        return True
    return False


def _cleanup_spaced_words(line: str) -> str:
    """Аккуратно склеивает OCR-строки с большим числом односимвольных фрагментов."""
    stripped = line.strip()
    if "|" in line or stripped.startswith("#") or stripped.startswith("!["):
        return line

    if len(re.findall(r"\b\w\b", stripped)) >= 5:
        return re.sub(r"(?<=\b\w)\s+(?=\w\b)", "", line)

    return line


def _merge_split_lines(lines: list[str]) -> list[str]:
    """Сливает соседние строки, если они похожи на грубый OCR-разрыв слова."""
    merged: list[str] = []
    i = 0

    while i < len(lines):
        cur = lines[i].rstrip()
        nxt = lines[i + 1].lstrip() if i + 1 < len(lines) else ""

        if (
            cur
            and nxt
            and "|" not in cur
            and "|" not in nxt
            and not cur.startswith("#")
            and not nxt.startswith("#")
            and not cur.startswith("![")
            and not nxt.startswith("![")
            and not cur.startswith("- ")
            and not nxt.startswith("- ")
        ):
            cur_words = cur.split()
            nxt_words = nxt.split()

            cur_last = cur_words[-1] if cur_words else ""
            nxt_first = nxt_words[0] if nxt_words else ""

            if (
                cur_last.isalpha()
                and nxt_first.isalpha()
                and 5 <= len(cur_last) <= 14
                and 2 <= len(nxt_first) <= 8
                and len(cur_words) == 1
                and len(nxt_words) == 1
            ):
                merged.append(cur + nxt)
                i += 2
                continue

        merged.append(lines[i])
        i += 1

    return merged


def _safe_join_split_words_in_line(line: str) -> str:
    """Склеивает части слов внутри одной строки по консервативным правилам."""
    stripped = line.strip()

    if (
        not stripped
        or "|" in line
        or stripped.startswith("#")
        or stripped.startswith("![")
        or stripped.startswith("- ")
    ):
        return line

    parts = re.split(r"(\s+)", line)
    result: list[str] = []
    i = 0

    while i < len(parts):
        token = parts[i]

        if i + 2 < len(parts):
            sep = parts[i + 1]
            nxt = parts[i + 2]

            if (
                sep.isspace()
                and _WORD_RE.fullmatch(token or "")
                and _WORD_RE.fullmatch(nxt or "")
            ):
                left = token
                right = nxt

                if len(left) >= 6 and 1 <= len(right) <= 2:
                    right_l = right.lower()
                    if right_l in {"ся", "сь", "d", "s"}:
                        result.append(left + right)
                        i += 3
                        continue
                    if right_l == "я" and left.lower().endswith("с"):
                        result.append(left + right)
                        i += 3
                        continue

                if len(left) >= 6 and 3 <= len(right) <= 4:
                    endings = {
                        "тель", "ение", "ость", "ание", "овать",
                        "ized", "tion", "ing", "ский", "чная", "емый",
                    }
                    if right.lower() in endings:
                        result.append(left + right)
                        i += 3
                        continue

        result.append(token)
        i += 1

    return "".join(result)


def _promote_headingish_lines(lines: list[str]) -> list[str]:
    """Поднимает строки, похожие на заголовки, до уровня `##`."""
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped
            and "|" not in line
            and not stripped.startswith("#")
            and not stripped.startswith("![")
            and _HEADINGISH_RE.match(stripped)
        ):
            out.append(f"## {stripped}")
        else:
            out.append(line)
    return out


def _drop_garbage_lines(lines: list[str]) -> list[str]:
    """Удаляет строки, похожие на шум, артефакты OCR и изолированный мусор."""
    cleaned: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue

        prev_line = lines[idx - 1].strip() if idx > 0 else ""
        next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""

        if _is_obvious_noise_line(stripped):
            continue

        if _PERCENT_ONLY_RE.fullmatch(stripped) or _NUMBER_ONLY_RE.fullmatch(stripped):
            if (not prev_line or _is_obvious_noise_line(prev_line) or prev_line.startswith("![")) and (
                not next_line or _is_obvious_noise_line(next_line) or next_line.startswith("#") or next_line.startswith("![")
            ):
                continue

        cleaned.append(line)
    return cleaned


def _table_row_cells(line: str) -> list[str]:
    """Разбивает строку Markdown-таблицы на ячейки без крайних разделителей."""
    parts = [p.strip() for p in line.strip().split("|")]
    if len(parts) >= 3:
        return parts[1:-1]
    return []


def _is_bad_table_block(block: list[str]) -> bool:
    """Определяет, выглядит ли таблица невалидной или шумовой."""
    pipe_lines = [line for line in block if "|" in line]
    if len(pipe_lines) < 2:
        return True

    cells = [_table_row_cells(line) for line in pipe_lines]
    cells = [row for row in cells if row]
    if not cells:
        return True

    total = 0
    bad = 0
    for row in cells:
        for cell in row:
            total += 1
            if not cell:
                bad += 1
                continue
            if _HEADINGISH_RE.search(cell):
                return True
            if _MD_IMAGE_REF_RE.search(cell):
                return True
            if _is_obvious_noise_line(cell):
                bad += 1

    if total == 0:
        return True

    if total <= 6 and bad / total >= 0.7:
        return True

    if max(len(row) for row in cells) <= 1 and total <= 4:
        return True

    return False


def _normalize_tables(markdown: str) -> str:
    """Очищает пустые и плохие таблицы, а также выравнивает строки таблиц."""
    markdown = _EMPTY_TABLE_RE.sub("", markdown)

    lines = markdown.splitlines()
    result: list[str] = []
    i = 0

    while i < len(lines):
        if "|" not in lines[i]:
            result.append(lines[i])
            i += 1
            continue

        block: list[str] = []
        j = i
        while j < len(lines) and "|" in lines[j]:
            segments = [segment.strip() for segment in lines[j].split("|")]
            if len(segments) >= 3:
                block.append("| " + " | ".join(segments[1:-1]) + " |")
            else:
                block.append(lines[j])
            j += 1

        if not _is_bad_table_block(block):
            result.extend(block)

        i = j

    return "\n".join(result)


def _remove_caption_after_images(lines: list[str]) -> list[str]:
    """Удаляет подписи формата `Рис. N.` сразу после Markdown-картинок."""
    out: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        prev_line = out[-1].strip() if out else ""
        if _CAPTION_RE.fullmatch(stripped) and prev_line.startswith("!["):
            continue
        out.append(line)
    return out


def postprocess_markdown(markdown: str, out_images_dir: Path) -> str:
    """Выполняет полную постобработку Markdown после Docling.

    Args:
        markdown: Исходный Markdown от конвертера.
        out_images_dir: Директория, где лежат итоговые изображения.

    Returns:
        Очищенный и нормализованный Markdown.
    """
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    text = _DRAFT_RE.sub("", text)
    text = text.replace("\u00a0", " ")

    lines = text.splitlines()
    lines = _merge_split_lines(lines)
    lines = [_cleanup_spaced_words(line) for line in lines]
    lines = [_safe_join_split_words_in_line(line) for line in lines]
    lines = _promote_headingish_lines(lines)
    lines = _drop_garbage_lines(lines)
    lines = _remove_caption_after_images(lines)

    text = "\n".join(lines)
    text = _normalize_tables(text)
    text = _remove_broken_image_refs(text, out_images_dir)

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = _MULTIBLANK_RE.sub("\n\n", text)
    text = text.strip() + "\n"
    return text


def _build_converter(
    no_ocr: bool,
    no_table_structure: bool,
    full_quality: bool,
) -> DocumentConverter:
    """Создает и настраивает экземпляр `DocumentConverter` для PDF.

    Args:
        no_ocr: Отключать ли OCR.
        no_table_structure: Отключать ли восстановление структуры таблиц.
        full_quality: Использовать ли более качественный режим.

    Returns:
        Готовый `DocumentConverter`.
    """
    if full_quality:
        images_scale = 1.0
        table_options = TableStructureOptions(mode=TableFormerMode.ACCURATE)
    else:
        images_scale = 0.88
        table_options = TableStructureOptions(mode=TableFormerMode.FAST)

    pipeline_options = ThreadedPdfPipelineOptions(
        do_ocr=not no_ocr,
        do_table_structure=not no_table_structure,
        generate_picture_images=True,
        images_scale=images_scale,
        table_structure_options=table_options,
        accelerator_options=AcceleratorOptions(),
    ).model_copy(deep=True)

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


def _extract_single_markdown(
    pdf_path: Path,
    output_dir: Path,
    converter: DocumentConverter,
    mode_name: str,
) -> str:
    """Извлекает Markdown для одного PDF в одном режиме конвертации.

    Args:
        pdf_path: Путь к PDF-документу.
        output_dir: Итоговая директория вывода.
        converter: Инициализированный Docling-конвертер.
        mode_name: Имя режима, используемое для временной директории.

    Returns:
        Нормализованный Markdown-текст документа.
    """
    stem = pdf_path.stem
    doc_num = _doc_num_from_stem(stem)

    result = converter.convert(str(pdf_path))
    document = result.document

    with tempfile.TemporaryDirectory(prefix=f"docling_{mode_name}_{stem}_") as tmp:
        work_dir = Path(tmp)
        md_work = work_dir / f"{stem}.md"

        document.save_as_markdown(
            md_work,
            artifacts_dir=Path("images"),
            image_mode=ImageRefMode.REFERENCED,
        )

        if not md_work.exists():
            raise FileNotFoundError(f"Markdown was not created: {md_work}")

        text = md_work.read_text(encoding="utf-8")
        text = _normalize_image_names(
            text,
            work_images_dir=work_dir / "images",
            out_images_dir=output_dir / "images",
            doc_num=doc_num,
        )
        text = postprocess_markdown(text, output_dir / "images")
        return text


def _choose_best_markdown(
    pdf_path: Path,
    output_dir: Path,
    no_ocr_converter: DocumentConverter,
    ocr_converter: DocumentConverter,
) -> str:
    """Сначала пытается извлечь Markdown без OCR, а OCR использует только как запасной вариант."""
    try:
        return _extract_single_markdown(
            pdf_path=pdf_path,
            output_dir=output_dir,
            converter=no_ocr_converter,
            mode_name="no_ocr",
        )
    except Exception:
        return _extract_single_markdown(
            pdf_path=pdf_path,
            output_dir=output_dir,
            converter=ocr_converter,
            mode_name="ocr",
        )


def convert_pdf(
    pdf_path: Path,
    output_dir: Path,
    no_ocr_converter: DocumentConverter,
    ocr_converter: DocumentConverter,
) -> None:
    """Конвертирует один PDF в Markdown и сохраняет результат на диск."""
    markdown = _choose_best_markdown(
        pdf_path=pdf_path,
        output_dir=output_dir,
        no_ocr_converter=no_ocr_converter,
        ocr_converter=ocr_converter,
    )
    out_md = output_dir / f"{pdf_path.stem}.md"
    out_md.write_text(markdown, encoding="utf-8")


def main() -> None:
    """Точка входа CLI для пакетной обработки pdf-документов."""
    parser = argparse.ArgumentParser(description="Baseline+ v6.1 no_ocr-first (Docling): PDF -> Markdown")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with PDF files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--max-files", type=int, default=None, help="Limit the number of files")
    parser.add_argument(
        "--no-table-structure",
        action="store_true",
        help="Disable TableFormer if OpenCV / table inference is problematic",
    )
    parser.add_argument(
        "--full-quality",
        action="store_true",
        help="Use full quality for image scale and table mode",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device for Docling models",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already have a markdown file in the output directory",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(args.input_dir.glob("*.pdf"))
    if args.max_files is not None:
        pdf_files = pdf_files[: args.max_files]

    if not pdf_files:
        print(f"ERROR: no PDF files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.skip_existing:
        before = len(pdf_files)
        pdf_files = [pdf for pdf in pdf_files if not (args.output_dir / f"{pdf.stem}.md").exists()]
        skipped = before - len(pdf_files)
        if skipped:
            print(f"Skip-existing: skipped {skipped}, remaining {len(pdf_files)}")
        if not pdf_files:
            print("Nothing to process.")
            sys.exit(0)

    device_name = os.environ.get("DOCLING_DEVICE", "auto")
    print("Initializing Docling...")
    print(f"Device (DOCLING_DEVICE): {device_name}")
    print(
        "Loading layout/OCR/table weights may take a few minutes. "
        "Both parsing modes will be initialized."
    )

    no_ocr_converter = _build_converter(
        no_ocr=True,
        no_table_structure=args.no_table_structure,
        full_quality=args.full_quality,
    )
    ocr_converter = _build_converter(
        no_ocr=False,
        no_table_structure=args.no_table_structure,
        full_quality=args.full_quality,
    )

    no_ocr_converter.initialize_pipeline(InputFormat.PDF)
    ocr_converter.initialize_pipeline(InputFormat.PDF)

    print(f"Found {len(pdf_files)} PDF files\n")

    for index, pdf_path in enumerate(pdf_files, 1):
        print(f"[{index}/{len(pdf_files)}] {pdf_path.name}...", end=" ", flush=True)
        try:
            convert_pdf(
                pdf_path=pdf_path,
                output_dir=args.output_dir,
                no_ocr_converter=no_ocr_converter,
                ocr_converter=ocr_converter,
            )
            print("OK")
        except Exception as error:  # noqa: BLE001
            print(f"ERROR: {error}")
        finally:
            _clear_cuda_cache()

    print(f"\nDone! Results are in: {args.output_dir}")


if __name__ == "__main__":
    main()