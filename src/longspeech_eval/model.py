from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from .config import ModelConfig


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


class MiniCPMOStreamingModel:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

        self.model = AutoModel.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            attn_implementation=config.attn_implementation,
            torch_dtype=_torch_dtype(config.dtype),
            init_vision=False,
            init_audio=True,
            init_tts=False,
        )
        self.model.eval()
        self.model.to(config.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=True,
        )

    def reset(self) -> None:
        if hasattr(self.model, "reset_session"):
            self.model.reset_session()

    def prefill(
        self,
        *,
        session_id: str,
        content: list[Any],
        is_last_chunk: bool,
        omni_mode: bool = False,
    ) -> None:
        msg = {"role": "user", "content": content}
        self.model.streaming_prefill(
            session_id=session_id,
            msgs=[msg],
            tokenizer=self.tokenizer,
            omni_mode=omni_mode,
            use_tts_template=False,
            enable_thinking=False,
            is_last_chunk=is_last_chunk,
        )

    def generate(
        self,
        *,
        session_id: str,
        max_new_tokens: int,
        do_sample: bool = False,
    ) -> str:
        iterator = self.model.streaming_generate(
            session_id=session_id,
            tokenizer=self.tokenizer,
            generate_audio=False,
            use_tts_template=False,
            enable_thinking=False,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
        )

        parts: list[str] = []
        for item in iterator:
            text = self._extract_text(item)
            if text:
                parts.append(text)
        return "".join(parts).strip()

    @staticmethod
    def _extract_text(item: Any) -> str:
        # Official text-only streaming path currently yields (text_chunk, is_finished).
        if isinstance(item, tuple):
            if item and isinstance(item[0], str):
                return item[0]
            return ""
        if isinstance(item, dict):
            value = item.get("text", "")
            return value if isinstance(value, str) else ""
        value = getattr(item, "text", "")
        return value if isinstance(value, str) else ""
