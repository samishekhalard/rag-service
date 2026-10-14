"""
Chat backends behind one protocol, with token accounting.

Every reply carries tokens in and out so the eval harness can report cost per
query without guessing. Ollama reports token counts in its response; for
backends that don't, the counts are estimated from characters and the eval
output marks the variant accordingly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import Settings


@dataclass
class ChatReply:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    estimated: bool = False


class ChatModel(Protocol):
    name: str

    def complete(self, system: str, user: str) -> ChatReply: ...


def _estimate(text: str) -> int:
    return max(1, len(text) // 4)


class CloudChat:
    """OpenAI-compatible /chat/completions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.name = settings.cloud_chat_model

    def complete(self, system: str, user: str) -> ChatReply:
        response = httpx.post(
            f"{self.settings.cloud_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.cloud_api_key}"},
            json={
                "model": self.name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
            },
            timeout=self.settings.request_timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        return ChatReply(
            text=payload["choices"][0]["message"]["content"],
            tokens_in=int(usage.get("prompt_tokens") or _estimate(system + user)),
            tokens_out=int(usage.get("completion_tokens") or 0),
            estimated="usage" not in payload,
        )


class OllamaChat:
    """Ollama /api/chat, non-streaming."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.name = settings.ollama_chat_model

    def complete(self, system: str, user: str) -> ChatReply:
        response = httpx.post(
            f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
            json={
                "model": self.name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self.settings.request_timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        return ChatReply(
            text=payload["message"]["content"],
            tokens_in=int(payload.get("prompt_eval_count") or _estimate(system + user)),
            tokens_out=int(payload.get("eval_count") or 0),
            estimated="prompt_eval_count" not in payload,
        )


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class ExtractiveChat:
    """
    Offline stand-in used by the test suite and by retrieval-only evaluation.

    It does no generation: it returns the sentences from the context that
    share the most terms with the question. That makes answer metrics a
    measurement of retrieval quality with generation held constant, which is
    the comparison you want before blaming the model for a bad answer.
    """

    name = "extractive"

    def complete(self, system: str, user: str) -> ChatReply:
        question, _, context = user.partition("\n\nContext:\n")
        # Only content words count. Scoring on raw token overlap lets "the",
        # "is" and "what" decide the answer, so the longest sentence wins
        # regardless of topic — which is exactly what happened before this
        # filter existed, and what the integration test now pins.
        terms = {t for t in re.findall(r"\w+", question.lower()) if len(t) > 3}
        sentences = [s.strip() for s in _SENTENCE.split(context) if s.strip()]
        if not sentences:
            return ChatReply(
                text="No relevant context found.", tokens_in=_estimate(user),
                tokens_out=8, estimated=True,
            )

        def score(sentence: str) -> float:
            tokens = {t for t in re.findall(r"\w+", sentence.lower()) if len(t) > 3}
            if not tokens or not terms:
                return 0.0
            # Normalised by sentence length so a long sentence cannot win on
            # volume alone.
            return len(terms & tokens) / math.sqrt(len(tokens))

        scores = [score(s) for s in sentences]
        best = max(range(len(sentences)), key=lambda i: (scores[i], -i))
        if scores[best] == 0.0:
            text = "No relevant context found."
        else:
            # Take the best sentence and the one after it. Technical prose
            # states the subject in one sentence and the value in the next
            # ("Fault F-310 overpressure. Discharge pressure exceeded 16 bar."),
            # so the sentence that matches the question often does not hold
            # the answer. Picking the two highest scorers instead returns two
            # topic sentences and no value — which the integration test caught.
            text = " ".join(sentences[best : best + 2])
        return ChatReply(text=text, tokens_in=_estimate(user), tokens_out=_estimate(text), estimated=True)


def build_chat(settings: Settings) -> ChatModel:
    if settings.provider == "fake":
        return ExtractiveChat()
    if settings.provider == "local":
        return OllamaChat(settings)
    if not settings.cloud_api_key:
        raise RuntimeError(
            "provider is 'cloud' but RAG_CLOUD_API_KEY is empty. "
            "Set the key, or use RAG_PROVIDER=local for Ollama."
        )
    return CloudChat(settings)
