"""Aide partagée des exemples : chargement .env + création du provider.

Chaque exemple accepte `--provider` / `--model` ; sans argument, le premier
provider dont la clé est présente dans .env est choisi (gemini, deepseek,
openai, anthropic).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Console Windows : éviter UnicodeEncodeError sur les accents/emoji des démos.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 — stream non reconfigurable (redirigé)
        pass

from autoagent import ModelConfig, create_provider  # noqa: E402

DEFAULTS = {
    "gemini": "gemini-2.5-flash",
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
    "kimi": "kimi-k3",   # écrasable : --model ou KIMI_MODEL dans .env
}
KEYS = {
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "kimi": "KIMI_API_KEY",
}

# Endpoints OpenAI-compatibles : le « provider » wire est openai, seul le
# base_url change. Même recette pour Groq, Ollama (http://localhost:11434/v1),
# vLLM, etc. — ajoute une entrée ici et une clé dans .env, c'est tout.
OPENAI_COMPATIBLES = {
    "kimi": ("https://api.moonshot.ai/v1", "KIMI_API_KEY", "KIMI_MODEL"),
}


def load_env() -> None:
    env = ROOT / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] not in ("'", '"'):  # commentaire inline `CLE=val  # note`
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        os.environ.setdefault(key.strip(), value.strip('"').strip("'"))


def make_provider(argv: list[str] | None = None, timeout: float = 180.0):
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    args, _extra = parser.parse_known_args(argv)  # tolère les args propres à l'exemple
    name = args.provider or next((n for n, k in KEYS.items() if os.getenv(k)), None)
    if name is None:
        sys.exit("Aucune clé LLM dans .env (GEMINI_API_KEY / DEEPSEEK_API_KEY / ...).")
    if name in OPENAI_COMPATIBLES:
        base_url, key_env, model_env = OPENAI_COMPATIBLES[name]
        model = args.model or os.getenv(model_env) or DEFAULTS[name]
        print(f"[provider: {name} (openai-compatible) / {model}]\n")
        return create_provider(ModelConfig(
            provider="openai", model=model, base_url=base_url,
            api_key_env=key_env, timeout=timeout,
        ))
    model = args.model or DEFAULTS[name]
    print(f"[provider: {name} / {model}]\n")
    return create_provider(ModelConfig(provider=name, model=model, timeout=timeout))
