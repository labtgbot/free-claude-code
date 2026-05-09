"""Tests for cli/entrypoints.py — fcc-init scaffolding logic."""

from pathlib import Path
from unittest.mock import patch


def _env_value(content: str, key: str) -> str | None:
    prefix = f"{key}="
    for line in content.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip().strip("\"'")
    return None


def _env_with_auth_token(content: str, token: str) -> str:
    prefix = "ANTHROPIC_AUTH_TOKEN="
    lines = content.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f'ANTHROPIC_AUTH_TOKEN="{token}"{newline}'
            return "".join(lines)
    raise AssertionError("Template must define ANTHROPIC_AUTH_TOKEN")


def _run_init(tmp_home: Path) -> tuple[str, Path]:
    """Run init() with home directory redirected to tmp_home. Returns (printed output, env_file path)."""
    from cli.entrypoints import init

    env_file = tmp_home / ".config" / "free-claude-code" / ".env"
    printed: list[str] = []

    with (
        patch("pathlib.Path.home", return_value=tmp_home),
        patch(
            "builtins.print",
            side_effect=lambda *a: printed.append(" ".join(str(x) for x in a)),
        ),
    ):
        init()

    return "\n".join(printed), env_file


def test_init_creates_env_file(tmp_path: Path) -> None:
    """init() creates .env from the bundled template when it doesn't exist yet."""
    output, env_file = _run_init(tmp_path)

    assert env_file.exists()
    assert env_file.stat().st_size > 0
    assert str(env_file) in output


def test_init_copies_template_content(tmp_path: Path) -> None:
    """init() writes the canonical template with only the auth token generated."""
    template = (Path(__file__).resolve().parents[2] / ".env.example").read_text(
        encoding="utf-8"
    )
    with patch("cli.entrypoints.secrets.token_urlsafe", return_value="generated-token"):
        _, env_file = _run_init(tmp_path)

    assert env_file.read_text("utf-8") == _env_with_auth_token(
        template, "generated-token"
    )


def test_init_generates_unique_auth_token_by_default(tmp_path: Path) -> None:
    """init() does not scaffold a shared public token into new configs."""
    _, first_env_file = _run_init(tmp_path / "first")
    _, second_env_file = _run_init(tmp_path / "second")

    first_token = _env_value(first_env_file.read_text("utf-8"), "ANTHROPIC_AUTH_TOKEN")
    second_token = _env_value(
        second_env_file.read_text("utf-8"), "ANTHROPIC_AUTH_TOKEN"
    )

    assert first_token is not None
    assert second_token is not None
    assert first_token != ""
    assert first_token != "freecc"
    assert second_token != first_token
    assert len(first_token) >= 32


def test_env_example_does_not_ship_known_shared_auth_token() -> None:
    """The checked-in template must not publish a reusable proxy auth token."""
    template = (Path(__file__).resolve().parents[2] / ".env.example").read_text(
        encoding="utf-8"
    )

    assert _env_value(template, "ANTHROPIC_AUTH_TOKEN") != "freecc"


def test_env_template_loader_uses_root_template_in_source_checkout() -> None:
    """Source checkout fallback uses the root .env.example as the single source."""
    from cli.entrypoints import _load_env_template

    template = (Path(__file__).resolve().parents[2] / ".env.example").read_text(
        encoding="utf-8"
    )

    assert _load_env_template() == template


def test_init_creates_parent_directories(tmp_path: Path) -> None:
    """init() creates ~/.config/free-claude-code/ even if it doesn't exist."""
    config_dir = tmp_path / ".config" / "free-claude-code"
    assert not config_dir.exists()

    _run_init(tmp_path)

    assert config_dir.is_dir()


def test_init_skips_if_env_already_exists(tmp_path: Path) -> None:
    """init() does not overwrite an existing .env and prints a warning."""
    # Create it first
    _run_init(tmp_path)

    env_file = tmp_path / ".config" / "free-claude-code" / ".env"
    env_file.write_text("existing content", encoding="utf-8")

    output, _ = _run_init(tmp_path)

    assert env_file.read_text("utf-8") == "existing content"
    assert "already exists" in output


def test_init_prints_next_step_hint(tmp_path: Path) -> None:
    """init() tells the user to run free-claude-code after editing .env."""
    output, _ = _run_init(tmp_path)

    assert "free-claude-code" in output
