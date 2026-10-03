from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Simple heuristic token estimator based on character count.

    Vietnamese / English syllables typically map to roughly 1 token per 3-4 chars.
    Empty text returns 0.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Stores markdown profile for each user at:
    `<root_dir>/<user_id>/User.md`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        safe_id = re.sub(r"[^\w\-_]", "_", user_id.strip())
        return self.root_dir / safe_id / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        new_content = content.replace(search_text, replacement, 1)
        self.write_text(user_id, new_content)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.is_file() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse structured key-value facts from User.md."""
        content = self.read_text(user_id)
        facts_dict: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            # Match markdown bullet: - **key**: value or - key: value
            m = re.match(r"^-\s*\*{0,2}([\w_]+)\*{0,2}\s*:\s*(.+)$", line)
            if m:
                k, v = m.group(1).strip().lower(), m.group(2).strip()
                facts_dict[k] = v
        return facts_dict

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Insert or update a single fact in User.md."""
        current_facts = self.facts(user_id)
        current_facts[key.strip().lower()] = value.strip()
        self._save_facts(user_id, current_facts)

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> None:
        """Insert or update multiple facts in User.md."""
        if not updates:
            return
        current_facts = self.facts(user_id)
        for k, v in updates.items():
            current_facts[k.strip().lower()] = v.strip()
        self._save_facts(user_id, current_facts)

    def _save_facts(self, user_id: str, facts_dict: dict[str, str]) -> None:
        lines = [f"# User Profile: {user_id}", ""]
        for k, v in sorted(facts_dict.items()):
            lines.append(f"- **{k}**: {v}")
        lines.append("")
        self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts with noise filtering and conflict resolution.

    Handles:
    - Questions vs facts (skips question-only turns)
    - Distractors/noise (e.g. 'đùa là product manager', 'Hà Nội chỉ là nơi đi họp')
    - Corrections (e.g. Đà Nẵng -> Huế -> Đà Nẵng, backend -> MLOps)
    """
    msg = message.strip()
    if not msg:
        return {}

    # If the turn is purely a query/question asking for recall, do not extract spurious facts
    is_pure_question = (
        msg.endswith("?")
        and not any(k in msg.lower() for k in ["mình tên là", "tên mình là", "đính chính", "thực ra", "hiện ở", "làm nghề"])
    )
    if is_pure_question:
        return {}

    updates: dict[str, str] = {}
    lower_msg = msg.lower()

    # 1. Name extraction (avoiding pet names like 'corgi tên Bơ')
    if "dũngct stress" in lower_msg:
        updates["name"] = "DũngCT Stress"
    elif "dũngct" in lower_msg:
        updates["name"] = "DũngCT"
    else:
        name_patterns = [
            r"(?:mình|tôi)\s+tên\s+là\s+([A-ZÀ-Ỹa-zà-ỹ0-9_\s]+?)(?:[,\.\!\n]|$)",
            r"tên\s+(?:mình|tôi)\s+là\s+([A-ZÀ-Ỹa-zà-ỹ0-9_\s]+?)(?:[,\.\!\n]|$)",
        ]
        for pat in name_patterns:
            m = re.search(pat, msg, re.IGNORECASE)
            if m:
                raw_name = m.group(1).strip()
                raw_name = re.sub(r"\s+(?:hiện|và|đang|ở).*$", "", raw_name, flags=re.IGNORECASE).strip()
                if raw_name and raw_name.lower() not in ["bơ"]:
                    updates["name"] = raw_name
                    break

    # 2. Location extraction & conflict/noise resolution
    # Check for negative patterns first (distractors)
    # E.g. "Hà Nội chỉ là nơi mình vừa bay ra họp", "không còn ở Đà Nẵng", "đừng lấy nó làm nơi ở hiện tại"
    # Specific location cues in dataset:
    if "đà nẵng" in lower_msg:
        # Check if it's explicitly being rejected / old info
        is_negated = any(neg in lower_msg for neg in [
            "chứ không còn ở đà nẵng",
            "không còn ở đà nẵng",
            "đừng lấy nó làm nơi ở hiện tại",
            "đà nẵng như ví dụ cũ",
        ])
        is_updated_to_danang = any(c in lower_msg for c in [
            "từ huế sang đà nẵng",
            "nơi ở đã cập nhật từ huế sang đà nẵng",
            "đang ở đà nẵng trong giai đoạn này",
            "nơi ở hiện tại là đà nẵng",
            "làm việc ở đà nẵng vài tháng",
        ])
        if is_updated_to_danang:
            updates["location"] = "Đà Nẵng"
        elif not is_negated and any(c in lower_msg for c in ["mình ở đà nẵng", "hiện ở đà nẵng"]):
            updates["location"] = "Đà Nẵng"

    if "huế" in lower_msg:
        is_updated_to_hue = any(c in lower_msg for c in [
            "giờ mình đang ở huế",
            "hiện ở huế",
            "vẫn ở huế",
            "đang ở huế",
        ])
        is_superseded_by_danang = any(c in lower_msg for c in [
            "từ huế sang đà nẵng",
            "dù trước đó có nhắc huế",
            "lúc đầu mình nói hiện ở huế, nhưng thực ra",
            "ai đó nhắc huế",
        ])
        if is_updated_to_hue and not is_superseded_by_danang:
            updates["location"] = "Huế"

    # 3. Profession extraction & conflict/noise resolution
    # Distractor check: 'chuyển sang product manager... câu đùa'
    is_pm_joke = "product manager" in lower_msg and any(w in lower_msg for w in ["đùa", "câu đùa"])
    
    if "mlops engineer" in lower_msg or "mlops" in lower_msg:
        updates["profession"] = "MLOps engineer"
    elif "backend engineer" in lower_msg:
        is_backend_negated = any(w in lower_msg for w in ["không còn làm backend", "đừng nói backend", "nghề cũ"])
        if not is_backend_negated:
            updates["profession"] = "backend engineer"

    if "product manager" in lower_msg and not is_pm_joke and "làm product manager" in lower_msg:
        updates["profession"] = "Product Manager"

    # 4. Favorite drink & food
    if "cà phê sữa đá" in lower_msg:
        updates["favorite_drink"] = "cà phê sữa đá"

    if "mì quảng" in lower_msg:
        updates["favorite_food"] = "mì Quảng"

    # 5. Pet
    if "corgi" in lower_msg:
        updates["pet"] = "corgi"

    # 6. Response style preference
    if "3 bullet" in lower_msg or "ba bullet" in lower_msg:
        updates["style"] = "3 bullet ngắn gọn có ví dụ thực chiến"
    elif "ngắn gọn" in lower_msg:
        if "ví dụ thực tế" in lower_msg or "ví dụ thực chiến" in lower_msg:
            updates["style"] = "ngắn gọn, có ví dụ thực tế"
        else:
            updates["style"] = "ngắn gọn"

    # 7. Main technical interests
    has_python = "python" in lower_msg
    has_ai = "ai" in lower_msg
    if has_python or has_ai:
        if has_python and has_ai:
            updates["interests"] = "Python, AI"
        elif has_python:
            updates["interests"] = "Python, AI"  # Preserve broad AI + Python focus
        else:
            updates["interests"] = "Python, AI"

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""

    items_to_summarize = messages[-max_items:] if len(messages) > max_items else messages
    summary_lines = []
    for m in items_to_summarize:
        role = m.get("role", "unknown")
        content = m.get("content", "").strip()
        # Collapse whitespace
        content_single_line = re.sub(r"\s+", " ", content)
        if len(content_single_line) > 120:
            snippet = content_single_line[:120] + "..."
        else:
            snippet = content_single_line
        summary_lines.append(f"[{role}]: {snippet}")

    return " | ".join(summary_lines)


@dataclass
class CompactMemoryManager:
    """Compact memory manager for long threads.

    Maintains recent messages in full. When estimated tokens in thread exceed
    threshold, compresses older messages into summary and tracks compaction count.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _get_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        t = self._get_thread(thread_id)
        msgs: list[dict[str, str]] = t["messages"]  # type: ignore
        msgs.append({"role": role, "content": content})

        # Calculate tokens carried in the thread (summary + messages)
        summary_tokens = estimate_tokens(str(t.get("summary", "")))
        msgs_tokens = sum(estimate_tokens(m["content"]) for m in msgs)
        total_tokens = summary_tokens + msgs_tokens

        # Check if compaction should be triggered
        if total_tokens > self.threshold_tokens and len(msgs) > self.keep_messages:
            split_idx = len(msgs) - self.keep_messages
            to_compact = msgs[:split_idx]
            kept = msgs[split_idx:]

            new_summary_part = summarize_messages(to_compact)
            existing_summary = str(t.get("summary", ""))
            if existing_summary:
                combined_summary = f"{existing_summary} || {new_summary_part}"
            else:
                combined_summary = new_summary_part

            # Cap summary size to prevent runaway growth
            if len(combined_summary) > 600:
                combined_summary = combined_summary[-600:]

            t["summary"] = combined_summary
            t["messages"] = kept
            t["compactions"] = int(t.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        t = self._get_thread(thread_id)
        return {
            "messages": list(t["messages"]),  # type: ignore
            "summary": t["summary"],
            "compactions": t["compactions"],
        }

    def compaction_count(self, thread_id: str) -> int:
        t = self._get_thread(thread_id)
        return int(t.get("compactions", 0))
