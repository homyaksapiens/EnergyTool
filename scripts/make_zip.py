"""Сборка ZIP для автосабмита из bot/ с проверками по регламенту организаторов.

    .venv\\Scripts\\python scripts\\make_zip.py   →  dist/submission.zip

В архив идут только корневые .py из bot/ и requirements.txt — вложенные папки платформа не переносит.
"""
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOT = ROOT / "bot"
ALLOWED = ROOT / "data" / "requirements.txt"
OUT = ROOT / "dist" / "submission.zip"
SECRET_RE = re.compile(r"sk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._-]{20,}")


def parse_requirements(path: Path) -> dict[str, str | None]:
    reqs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, sep, version = line.partition("==")
            reqs[name.strip().lower().replace("_", "-")] = version.strip() if sep else None
    return reqs


def main() -> None:
    errors = []
    files = sorted(BOT.glob("*.py"))
    if not (BOT / "solution.py").exists():
        errors.append("нет bot/solution.py")
    for path in files:
        text = path.read_text(encoding="utf-8")
        try:
            compile(text, str(path), "exec")
        except SyntaxError as e:
            errors.append(f"{path.name}: синтаксическая ошибка: {e}")
        if SECRET_RE.search(text):
            errors.append(f"{path.name}: похоже на API-ключ в коде")

    reqs = parse_requirements(BOT / "requirements.txt")
    for name, version in reqs.items():
        if not version:
            errors.append(f"requirements.txt: версия {name} не зафиксирована через ==")
    if ALLOWED.exists():
        allowed = parse_requirements(ALLOWED)
        for name, version in reqs.items():
            if name not in allowed:
                errors.append(f"requirements.txt: пакета {name} нет в разрешённом списке")
            elif version and allowed[name] != version:
                errors.append(f"requirements.txt: {name}=={version}, а разрешена версия {allowed[name]}")
    else:
        print("Внимание: нет data/requirements.txt — список разрешённых пакетов не проверен")

    if errors:
        print("Сборка остановлена:\n  " + "\n  ".join(errors))
        sys.exit(1)

    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)
        archive.write(BOT / "requirements.txt", "requirements.txt")
    names = [p.name for p in files] + ["requirements.txt"]
    print(f"Готово: {OUT} ({OUT.stat().st_size / 1024:.1f} КБ)\n  " + "\n  ".join(names))


if __name__ == "__main__":
    main()
