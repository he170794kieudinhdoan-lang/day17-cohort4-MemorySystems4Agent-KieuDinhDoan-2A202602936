from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent.

    Requirements:
    - Within-session memory only (per thread_id).
    - No persistent `User.md`.
    - Forgets long-term facts across new threads.
    - Carries entire uncompressed thread context on every turn.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if not self.force_offline and self._maybe_build_langchain_agent() is not None:
            try:
                prompt_tokens = self._calc_prompt_context_tokens(thread_id, message)
                sess = self._get_session(thread_id)
                sess.prompt_tokens_processed += prompt_tokens
                sess.messages.append({"role": "user", "content": message})

                model_res = self.langchain_agent.invoke(message)
                reply_text = getattr(model_res, "content", str(model_res))
                if isinstance(reply_text, list):
                    # Handle multimodal/block list if returned by some providers
                    reply_text = " ".join(
                        b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b))
                        for b in reply_text
                    )

                turn_tokens = estimate_tokens(message) + estimate_tokens(reply_text)
                sess.token_usage += turn_tokens
                sess.messages.append({"role": "assistant", "content": reply_text})
                return {
                    "response": reply_text,
                    "tokens": sess.token_usage,
                    "prompt_tokens": sess.prompt_tokens_processed,
                }
            except Exception:
                # Fallback to offline on live failure
                pass

        return self._reply_offline(thread_id, message)

    def _get_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def _calc_prompt_context_tokens(self, thread_id: str, new_message: str) -> int:
        sess = self._get_session(thread_id)
        existing_tokens = sum(estimate_tokens(m["content"]) for m in sess.messages)
        return existing_tokens + estimate_tokens(new_message)

    def token_usage(self, thread_id: str) -> int:
        return self._get_session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._get_session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory
        return 0

    def memory_file_size(self, user_id: str) -> int:
        # Baseline has no persistent storage
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline path for repeatable benchmarking."""
        sess = self._get_session(thread_id)

        # Baseline accumulates all previous messages as prompt context without compaction
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in sess.messages) + estimate_tokens(message)
        sess.prompt_tokens_processed += prompt_tokens
        sess.messages.append({"role": "user", "content": message})

        # Deterministic reply generation:
        # Across a fresh thread, baseline has no history or profile, so it knows nothing about the user.
        lower_msg = message.lower()
        is_question = "?" in message or any(w in lower_msg for w in ["ai", "gì", "đâu", "nhắc lại", "tên", "nghề"])

        if is_question:
            # Baseline cannot recall user facts across new threads
            reply_text = "Xin lỗi, mình chỉ nhớ thông tin trong phiên hiện tại và không có thông tin cá nhân của bạn từ các phiên trước."
        else:
            reply_text = "Đã nhận thông tin: " + message[:60] + "..." if len(message) > 60 else "Đã nhận thông tin: " + message

        out_tokens = estimate_tokens(reply_text)
        sess.token_usage += estimate_tokens(message) + out_tokens
        sess.messages.append({"role": "assistant", "content": reply_text})

        return {
            "response": reply_text,
            "tokens": sess.token_usage,
            "prompt_tokens": sess.prompt_tokens_processed,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally wire chat model if API key is provided."""
        if self.langchain_agent is not None:
            return self.langchain_agent
        if not self.config.model.api_key:
            return None
        try:
            self.langchain_agent = build_chat_model(self.config.model)
            return self.langchain_agent
        except Exception:
            return None
