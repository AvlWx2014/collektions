default:
  just --list

clean:
  @find . \( -name "coverage.xml" -o -name "junit.xml" -o -name ".coverage" -o -name "htmlcov" \) -exec rm -rf {} \;

format:
  uv run ruff format --config ./pyproject.toml .

format-check:
  uv run ruff format --config ./pyproject.toml --check .

lint:
  uv run ruff check --config ./pyproject.toml --fix .

lint-check:
  uv run ruff check --config ./pyproject.toml .

lint-unsafe:
  uv run ruff check  --config ./pyproject.toml --unsafe-fixes .

provision:
  uv sync --dev --all-groups

sync:
  uv sync

type-check:
  uv run mypy --config-file ./pyproject.toml src/

test:
  uv run pytest

verify: format lint format type-check vuln-check test

vuln-check:
  uv run bandit -rc pyproject.toml src/ tests/
