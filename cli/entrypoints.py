"""CLI entry points for the installed package."""

from __future__ import annotations

import secrets
from pathlib import Path

_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
_AUTH_TOKEN_BYTES = 32


def _load_env_template() -> str:
    """Load the canonical root env template from package resources or source."""
    import importlib.resources

    packaged = importlib.resources.files("cli").joinpath("env.example")
    if packaged.is_file():
        return packaged.read_text("utf-8")

    source_template = Path(__file__).resolve().parents[1] / ".env.example"
    if source_template.is_file():
        return source_template.read_text(encoding="utf-8")

    raise FileNotFoundError("Could not find bundled or source .env.example template.")


def _with_generated_auth_token(template: str) -> str:
    """Return an env file template with a fresh proxy auth token."""
    token_line = f'{_AUTH_TOKEN_ENV}="{secrets.token_urlsafe(_AUTH_TOKEN_BYTES)}"'
    prefix = f"{_AUTH_TOKEN_ENV}="
    lines = template.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            stripped = line.rstrip("\r\n")
            line_ending = line[len(stripped) :]
            lines[index] = f"{token_line}{line_ending}"
            return "".join(lines)

    suffix = "" if template.endswith("\n") or not template else "\n"
    return f"{template}{suffix}{token_line}\n"


def serve() -> None:
    """Start the FastAPI server (registered as `free-claude-code` script)."""
    import uvicorn

    from cli.process_registry import kill_all_best_effort
    from config.settings import get_settings

    settings = get_settings()
    try:
        uvicorn.run(
            "api.app:create_asgi_app",
            factory=True,
            host=settings.host,
            port=settings.port,
            log_level="debug",
            timeout_graceful_shutdown=5,
        )
    finally:
        kill_all_best_effort()


def init() -> None:
    """Scaffold config at ~/.config/free-claude-code/.env (registered as `fcc-init`)."""
    config_dir = Path.home() / ".config" / "free-claude-code"
    env_file = config_dir / ".env"

    if env_file.exists():
        print(f"Config already exists at {env_file}")
        print("Delete it first if you want to reset to defaults.")
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    template = _load_env_template()
    env_file.write_text(_with_generated_auth_token(template), encoding="utf-8")
    print(f"Config created at {env_file}")
    print(
        "Edit it to set your API keys and model preferences, then run: free-claude-code"
    )
