from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests with low compaction threshold."""
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root)
    cfg.state_dir = tmp_path / "state"
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.compact_threshold_tokens = 50
    cfg.compact_keep_messages = 2
    return cfg


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    cfg = make_config(tmp_path)
    agent = AdvancedAgent(cfg, force_offline=True)
    store = agent.profile_store

    user_id = "test_user"
    initial_content = "# User Profile: test_user\n\n- **name**: Nam\n- **location**: Ha Noi\n"
    written_path = store.write_text(user_id, initial_content)

    assert written_path.is_file()
    assert "Nam" in store.read_text(user_id)
    assert store.file_size(user_id) > 0

    # Edit location from Ha Noi to Da Nang
    edited = store.edit_text(user_id, "Ha Noi", "Da Nang")
    assert edited
    new_content = store.read_text(user_id)
    assert "Da Nang" in new_content
    assert "Ha Noi" not in new_content


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    cfg = make_config(tmp_path)
    agent = AdvancedAgent(cfg, force_offline=True)
    thread_id = "test-compact-thread"

    for i in range(10):
        agent.reply(
            user_id="user_compact",
            thread_id=thread_id,
            message=f"Đoạn hội thoại kỹ thuật số {i} có độ dài lớn nhằm mục đích vượt qua ngưỡng compact đã cấu hình.",
        )

    assert agent.compaction_count(thread_id) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)
    base = BaselineAgent(cfg, force_offline=True)
    adv = AdvancedAgent(cfg, force_offline=True)

    # Session 1: user introduces themselves
    intro = "Chào bạn, mình tên là DũngCT. Mình ở Huế và làm MLOps engineer."
    base.reply("dungct", "thread-1", intro)
    adv.reply("dungct", "thread-1", intro)

    # Session 2: ask recall question in an isolated thread
    question = "Hiện tại mình đang ở đâu và làm nghề gì?"
    base_ans = base.reply("dungct", "thread-2", question)["response"]
    adv_ans = adv.reply("dungct", "thread-2", question)["response"]

    # Baseline only remembers within thread, failing cross-session recall
    assert "Huế" not in base_ans
    assert "MLOps engineer" not in base_ans

    # Advanced recalls user profile across sessions from User.md
    assert "Huế" in adv_ans
    assert "MLOps engineer" in adv_ans


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    base = BaselineAgent(cfg, force_offline=True)
    adv = AdvancedAgent(cfg, force_offline=True)
    thread_id = "stress-load-thread"

    long_message = (
        "Chi tiết kỹ thuật về chương trình Artemis III của NASA kết hợp cùng chuyến bay thử nghiệm X-59. "
        "Ngoài ra phân tích báo cáo rủi ro El Nino của WMO và chính sách tiết kiệm điện năng của Canada. "
        "Toàn bộ dữ liệu này được đưa vào để kéo dài ngữ cảnh và kích hoạt nén lịch sử compact memory."
    )

    for _ in range(12):
        base.reply("user_stress", thread_id, long_message)
        adv.reply("user_stress", thread_id, long_message)

    assert adv.compaction_count(thread_id) > 0
    # Advanced processes fewer prompt tokens than baseline
    assert adv.prompt_token_usage(thread_id) < base.prompt_token_usage(thread_id)


def test_bonus_conflict_handling_and_correction(tmp_path: Path) -> None:
    """Verify that updating a profile field cleanly replaces old info without conflicts."""
    cfg = make_config(tmp_path)
    agent = AdvancedAgent(cfg, force_offline=True)
    user_id = "test_conflict_user"

    # User introduces as living in Hue and working backend
    agent.reply(user_id, "t-1", "Chào bạn, mình là DũngCT, hiện ở Huế và làm backend engineer.")
    facts1 = agent.profile_store.facts(user_id)
    assert facts1.get("location") == "Huế"
    assert facts1.get("profession") == "backend engineer"

    # User provides correction/update to Da Nang and MLOps
    agent.reply(
        user_id,
        "t-2",
        "Đính chính nhé, nơi ở đã cập nhật từ Huế sang Đà Nẵng, và mình chuyển sang làm MLOps engineer rồi.",
    )
    facts2 = agent.profile_store.facts(user_id)
    assert facts2.get("location") == "Đà Nẵng"
    assert facts2.get("profession") == "MLOps engineer"


def test_bonus_distractor_and_query_filtering(tmp_path: Path) -> None:
    """Verify queries and jokes do not pollute persistent User.md."""
    cfg = make_config(tmp_path)
    agent = AdvancedAgent(cfg, force_offline=True)
    user_id = "test_filter_user"

    # Pure question should not create spurious profile facts
    agent.reply(user_id, "t-1", "Thời tiết hôm nay thế nào?")
    assert agent.profile_store.file_size(user_id) == 0

    # User introduces normal facts
    agent.reply(user_id, "t-2", "Mình tên là Tuấn, hiện ở Đà Nẵng.")
    facts = agent.profile_store.facts(user_id)
    assert facts.get("name") == "Tuấn"
    assert facts.get("location") == "Đà Nẵng"

    # Distractor: Joke about Product Manager
    agent.reply(user_id, "t-3", "Nói đùa chứ chuyển sang làm product manager chắc nhàn hơn.")
    facts_after_joke = agent.profile_store.facts(user_id)
    assert facts_after_joke.get("profession") != "Product Manager"

