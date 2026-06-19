from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig.

    1. Resolve the repo root or default to the current file parent.
    2. Load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    data_dir = root / "data"

    # Read env vars or use sensible defaults
    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "400"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    provider_name = os.getenv("LLM_PROVIDER", "openai")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # Simple logic to grab the appropriate key based on provider
    api_key = None
    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    elif provider_name == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider_name == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
    elif provider_name == "custom":
        api_key = os.getenv("CUSTOM_API_KEY")

    base_url = os.getenv("LLM_BASE_URL") or os.getenv("CUSTOM_BASE_URL")

    model_config = ProviderConfig(
        provider=provider_name,
        model_name=model_name,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url
    )

    judge_config = ProviderConfig(
        provider=os.getenv("JUDGE_PROVIDER", "openai"),
        model_name=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL")
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_config,
        judge_model=judge_config
    )

