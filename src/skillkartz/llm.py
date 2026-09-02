"""Optional Claude-backed narration.

The design doc is explicit that "the LLM supports conversation and reasoning,
while market figures come from approved data and deterministic formulas"
(section 2). So every number, threshold decision, roadmap and governance check in
this project is produced by deterministic Python. The LLM is used only to phrase
the final answer more naturally, and it is entirely optional: with no API key (or
``SKILLKARTZ_LLM`` unset) the pipeline falls back to a deterministic template and
runs fully offline.

Enable it with::

    export SKILLKARTZ_LLM=1
    export ANTHROPIC_API_KEY=sk-ant-...      # or `ant auth login`
    # optional: export SKILLKARTZ_MODEL=claude-opus-5
"""

from __future__ import annotations

from typing import Optional

from .config import SETTINGS


class LLMClient:
    def __init__(self, enabled: Optional[bool] = None, model: Optional[str] = None) -> None:
        self.model = model or SETTINGS.llm_model
        self._want = SETTINGS.llm_enabled if enabled is None else enabled
        self._client = None
        self._error: Optional[str] = None
        if self._want:
            self._init_client()

    def _init_client(self) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError:
            self._error = "the 'anthropic' package is not installed (pip install anthropic)"
            return
        try:
            self._client = anthropic.Anthropic()
        except Exception as exc:  # pragma: no cover - depends on local creds
            self._error = f"could not construct Anthropic client: {exc}"

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def status(self) -> str:
        if self.available:
            return f"LLM narration ON (model={self.model})"
        if not self._want:
            return "LLM narration OFF (deterministic template)"
        return f"LLM narration requested but unavailable: {self._error}"

    def narrate(self, system: str, prompt: str, max_tokens: int = 700) -> Optional[str]:
        """Return polished prose, or None to signal 'use the template'."""
        if not self.available:
            return None
        try:  # pragma: no cover - network path
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
            return "".join(parts).strip() or None
        except Exception as exc:  # pragma: no cover
            self._error = f"request failed: {exc}"
            return None
