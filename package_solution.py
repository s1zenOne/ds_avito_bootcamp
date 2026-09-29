"""Проверяю мой ответ и собираю небольшой архив для GitHub.

Я включаю код, документацию и локальные веса моделей. Исходные Parquet-файлы
и промежуточные индексы остаются вне архива: они велики, а проверяющий может
положить свой набор в ``dataset/`` и повторить предсказание по README.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from solution import validate_answer


ROOT = Path(__file__).resolve().parent
PACKAGE_FILES = (
    ".gitignore",
    "README.md",
    "PROGRESS.md",
    "requirements.txt",
    "answer.csv",
    "retrieval.py",
    "solution.py",
    "ranker_experiment.py",
    "improved_predict.py",
    "prototype_experiment.py",
    "prototype_predict.py",
    "package_solution.py",
    "models/catboost_v5.cbm",
    "models/catboost_v6.cbm",
    "models/catboost_v9.cbm",
    "models/catboost_v10.cbm",
    "models/lambdarank.txt",
)


def project_path(path):
    """Я разрешаю относительные пути от корня проекта, независимо от cwd."""
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--tag", default="final_recall_0_835023")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--answer", type=Path, default=ROOT / "answer.csv",
                        help="Answer CSV to include as answer.csv inside the archive")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md",
                        help="Student README to include in the archive")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports",
                        help="Directory of small validation reports to include under reports/")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.tag):
        raise ValueError("Тег версии может содержать только буквы, цифры, точку, _ и -")

    # Проверяю именно выбранные файлы: overrides не требуют наличия
    # исходных answer.csv и README.md рядом со скриптом.
    answer_path = project_path(args.answer)
    readme_path = project_path(args.readme)
    sources = {name: (answer_path if name == "answer.csv" else
                      readme_path if name == "README.md" else ROOT / name)
               for name in PACKAGE_FILES}
    reports_dir = project_path(args.reports_dir)
    if reports_dir.exists():
        if not reports_dir.is_dir():
            raise NotADirectoryError(reports_dir)
        for source in sorted(reports_dir.rglob("*")):
            if source.is_file():
                name = "reports/" + source.relative_to(reports_dir).as_posix()
                sources[name] = source
    missing = [str(source) for source in sources.values() if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"Не хватает файлов для архива: {missing}")

    # Перед упаковкой я повторяю проверки формата и совпадения ID с корпусом.
    data_dir = project_path(args.data_dir)
    queries = pd.read_parquet(data_dir / "benchmark_queries.parquet")
    items = pd.read_parquet(data_dir / "benchmark_items.parquet", columns=["item_id"])
    checked = validate_answer(answer_path, queries, items)

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"avito_bootcamp_solution_{args.tag}.zip"
    # Сохранённые версии я не перезаписываю: для нового архива нужен новый tag.
    if archive.exists():
        raise FileExistsError(f"Архив уже существует: {archive}; выберите другой --tag")
    prefix = "avito_bootcamp_solution"
    manifest = {
        "format_version": 1,
        "answer_sha256": checked["sha256"],
        "answer_rows": checked["rows"],
        # JSON преобразует числовые ключи в строки; фиксирую это заранее,
        # чтобы обратное чтение manifest.json можно было сравнить строго.
        "candidates_per_query": {str(count): rows for count, rows in
                                 checked["candidates_per_query"].items()},
        "files": {},
        "note": "Parquet data are supplied separately; no external inference API is used.",
    }
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=9) as zipped:
        for name, source in sources.items():
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest["files"][name] = {"bytes": source.stat().st_size, "sha256": digest}
            zipped.write(source, arcname=f"{prefix}/{name}")
        zipped.writestr(f"{prefix}/manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    # Я проверяю CRC, список файлов и SHA-256 каждого упакованного файла,
    # включая модели, инструкции, контрольную точку и локальные метрики.
    with ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise RuntimeError("Архив повреждён")
        expected = {f"{prefix}/{name}" for name in manifest["files"]}
        expected.add(f"{prefix}/manifest.json")
        if set(zipped.namelist()) != expected:
            raise RuntimeError("Список файлов в ZIP не совпал с manifest.json")
        packed_manifest = json.loads(zipped.read(f"{prefix}/manifest.json"))
        if packed_manifest != manifest:
            raise RuntimeError("В ZIP попал неверный manifest.json")
        for name, record in manifest["files"].items():
            packed = zipped.read(f"{prefix}/{name}")
            if len(packed) != record["bytes"] or hashlib.sha256(packed).hexdigest() != record["sha256"]:
                raise RuntimeError(f"Не совпали размер или SHA-256 файла {name}")
        if manifest["files"]["answer.csv"]["sha256"] != checked["sha256"]:
            raise RuntimeError("В архив попал не тот answer.csv")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                      "answer_sha256": checked["sha256"], "rows": checked["rows"],
                      "files_verified": len(manifest["files"])},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
