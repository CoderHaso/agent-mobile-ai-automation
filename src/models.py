"""Registry of supported Groq and DeepSeek models with metadata.

Used by:
  - The CLI / GUI model picker (display labels, pricing).
  - `LLMClient.from_choice(provider, model)` to build a client.

Pricing reflects the public list prices as of May 2026 (USD per 1M tokens).
Stars are subjective relative ratings within this list:
  - quality : reasoning ability on long-tail / agentic tasks (1=basic … 5=frontier)
  - speed   : tokens per second observed on the provider (1=slow … 5=very fast)

Add or update an entry here and it shows up in the picker automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ModelInfo:
    provider: str          # "groq" | "deepseek"
    slug: str              # API model identifier (passed verbatim to the API)
    label: str             # human-readable name
    quality: int           # 1-5
    speed: int             # 1-5
    input_per_m: float     # USD per 1M input tokens
    output_per_m: float    # USD per 1M output tokens
    context_k: int         # context window (in K tokens)
    notes: str = ""

    # ---- presentation helpers -------------------------------------------
    @staticmethod
    def _stars(n: int) -> str:
        n = max(0, min(5, int(n)))
        return "★" * n + "☆" * (5 - n)

    @staticmethod
    def _stars_ascii(n: int) -> str:
        n = max(0, min(5, int(n)))
        return "*" * n + "." * (5 - n)

    @property
    def quality_stars(self) -> str:
        return self._stars(self.quality)

    @property
    def speed_stars(self) -> str:
        return self._stars(self.speed)

    @property
    def quality_stars_ascii(self) -> str:
        return self._stars_ascii(self.quality)

    @property
    def speed_stars_ascii(self) -> str:
        return self._stars_ascii(self.speed)

    @property
    def cost_label(self) -> str:
        return f"${self.input_per_m:.2f} in / ${self.output_per_m:.2f} out / 1M"

    def short_label(self) -> str:
        """One-liner for use in QComboBox / Rich tables."""
        return (
            f"{self.label}  "
            f"Q{self.quality_stars}  S{self.speed_stars}  "
            f"{self.cost_label}  ({self.context_k}K ctx)"
        )

    def detail_label(self) -> str:
        return (
            f"Provider: {self.provider}\n"
            f"API slug: {self.slug}\n"
            f"Quality:  {self.quality_stars} ({self.quality}/5)\n"
            f"Speed:    {self.speed_stars} ({self.speed}/5)\n"
            f"Cost:     {self.cost_label}\n"
            f"Context:  {self.context_k}K tokens\n"
            f"Notes:    {self.notes or '—'}"
        )


# --------------------------------------------------------------------------- #
# Groq — sorted from cheapest+lightest to most powerful
# Source: https://wow.groq.com/pricing  (May 2026 list prices)
# --------------------------------------------------------------------------- #

GROQ_MODELS: List[ModelInfo] = [
    ModelInfo(
        provider="groq",
        slug="llama-3.1-8b-instant",
        label="Llama 3.1 8B Instant",
        quality=2, speed=5,
        input_per_m=0.05, output_per_m=0.08, context_k=128,
        notes="Cheapest + fastest. OK for trivial tasks, weak at agentic reasoning.",
    ),
    ModelInfo(
        provider="groq",
        slug="openai/gpt-oss-20b",
        label="GPT-OSS 20B",
        quality=3, speed=5,
        input_per_m=0.075, output_per_m=0.30, context_k=128,
        notes="Solid open-weights model from OpenAI; very fast, cheap.",
    ),
    ModelInfo(
        provider="groq",
        slug="meta-llama/llama-4-scout-17b-16e-instruct",
        label="Llama 4 Scout 17B×16E",
        quality=4, speed=5,
        input_per_m=0.11, output_per_m=0.34, context_k=128,
        notes="Llama 4 small MoE. Great quality/cost — good DEFAULT for this agent.",
    ),
    ModelInfo(
        provider="groq",
        slug="openai/gpt-oss-120b",
        label="GPT-OSS 120B",
        quality=5, speed=4,
        input_per_m=0.15, output_per_m=0.60, context_k=128,
        notes="Strong open-weights flagship; cheap for its capability.",
    ),
    ModelInfo(
        provider="groq",
        slug="meta-llama/llama-4-maverick-17b-128e-instruct",
        label="Llama 4 Maverick 17B×128E",
        quality=5, speed=4,
        input_per_m=0.20, output_per_m=0.60, context_k=128,
        notes="Llama 4 large MoE — recommended for the hardest UI flows.",
    ),
    ModelInfo(
        provider="groq",
        slug="qwen/qwen3-32b",
        label="Qwen3 32B",
        quality=4, speed=4,
        input_per_m=0.29, output_per_m=0.59, context_k=128,
        notes="Strong reasoning, good multilingual (TR/CN/EN).",
    ),
    ModelInfo(
        provider="groq",
        slug="llama-3.3-70b-versatile",
        label="Llama 3.3 70B Versatile",
        quality=4, speed=4,
        input_per_m=0.59, output_per_m=0.79, context_k=128,
        notes="Mature, reliable mainstream Llama 3.3. The classic safe choice.",
    ),
    ModelInfo(
        provider="groq",
        slug="moonshotai/kimi-k2-instruct",
        label="Kimi K2 (Moonshot)",
        quality=5, speed=3,
        input_per_m=1.00, output_per_m=3.00, context_k=256,
        notes="Premium. 256K context — best when prompts are huge.",
    ),
]

# --------------------------------------------------------------------------- #
# DeepSeek — sorted from cheapest+fastest to premium
# Source: https://api-docs.deepseek.com/quick_start/pricing  (May 2026)
# --------------------------------------------------------------------------- #

DEEPSEEK_MODELS: List[ModelInfo] = [
    ModelInfo(
        provider="deepseek",
        slug="deepseek-v4-flash",
        label="DeepSeek V4 Flash [Vision]",
        quality=4, speed=4,
        input_per_m=0.14, output_per_m=0.28, context_k=1000,
        notes="DEFAULT. 1M context, supports thinking + non-thinking modes. Native Vision.",
    ),
    ModelInfo(
        provider="deepseek",
        slug="deepseek-chat",
        label="DeepSeek Chat (legacy)",
        quality=4, speed=4,
        input_per_m=0.14, output_per_m=0.28, context_k=1000,
        notes="Legacy alias → maps to V4 Flash non-thinking. Will be removed.",
    ),
    ModelInfo(
        provider="deepseek",
        slug="deepseek-reasoner",
        label="DeepSeek Reasoner (legacy)",
        quality=5, speed=3,
        input_per_m=0.14, output_per_m=0.55, context_k=1000,
        notes="Legacy alias → maps to V4 Flash thinking mode. Will be removed.",
    ),
    ModelInfo(
        provider="deepseek",
        slug="deepseek-v4-pro",
        label="DeepSeek V4 Pro [Vision]",
        quality=5, speed=3,
        input_per_m=0.435, output_per_m=0.87, context_k=1000,
        notes="Frontier-tier reasoning (75% off through May 31, 2026). Native Vision.",
    ),
]


ALL_MODELS: List[ModelInfo] = GROQ_MODELS + DEEPSEEK_MODELS

PROVIDERS = ("groq", "deepseek")


def by_provider(provider: str) -> List[ModelInfo]:
    return [m for m in ALL_MODELS if m.provider == provider]


def find(provider: str, slug: str) -> Optional[ModelInfo]:
    for m in ALL_MODELS:
        if m.provider == provider and m.slug == slug:
            return m
    return None


def default_for(provider: str) -> ModelInfo:
    """Reasonable starting choice per provider for an agentic UI task."""
    if provider == "groq":
        # Llama 4 Scout — best quality/cost for agentic loops.
        return find("groq", "meta-llama/llama-4-scout-17b-16e-instruct") or GROQ_MODELS[0]
    if provider == "deepseek":
        return find("deepseek", "deepseek-v4-flash") or DEEPSEEK_MODELS[0]
    raise ValueError(f"Unknown provider: {provider}")
