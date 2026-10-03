from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    """Return proportion of expected facts appearing in the answer."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matched = sum(1 for exp in expected if exp.lower() in ans_lower)
    return matched / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Evaluate response quality combining fact coverage and concise formatting."""
    rec = recall_points(answer, expected)
    # Check length sanity: between 20 and 800 chars is ideal for agent recall answers
    ans_len = len(answer.strip())
    length_score = 1.0 if 20 <= ans_len <= 800 else 0.5
    return round(rec * 0.8 + length_score * 0.2, 3)


def run_agent_benchmark(agent_name: str, agent: Any, conversations: list[dict[str, Any]], config: LabConfig) -> BenchmarkRow:
    """Evaluate one agent over multiple conversations with isolated recall threads."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        turns = conv.get("turns", [])
        recall_questions = conv.get("recall_questions", [])

        # 1. Feed conversation turns in the primary thread
        for turn in turns:
            agent.reply(user_id=user_id, thread_id=conv_id, message=turn)

        total_agent_tokens += agent.token_usage(conv_id)
        total_prompt_tokens += agent.prompt_token_usage(conv_id)
        total_compactions += agent.compaction_count(conv_id)

        # 2. Ask recall questions in fresh threads to test cross-session persistence
        for idx, rq in enumerate(recall_questions):
            recall_thread = f"{conv_id}-recall-{idx}"
            question = rq["question"]
            expected = rq["expected_contains"]

            res = agent.reply(user_id=user_id, thread_id=recall_thread, message=question)
            ans = res["response"]

            rec = recall_points(ans, expected)
            qual = heuristic_quality(ans, expected)
            recall_scores.append(rec)
            quality_scores.append(qual)

            total_agent_tokens += agent.token_usage(recall_thread)
            total_prompt_tokens += agent.prompt_token_usage(recall_thread)
            total_compactions += agent.compaction_count(recall_thread)

    # 3. Calculate memory growth in bytes
    all_users = {c["user_id"] for c in conversations}
    memory_growth = sum(agent.memory_file_size(u) for u in all_users)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a github markdown table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality * 100:.1f}%",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    std_data_path = config.data_dir / "conversations.json"
    stress_data_path = config.data_dir / "advanced_long_context.json"

    std_conversations = load_conversations(std_data_path)
    stress_conversations = load_conversations(stress_data_path)

    # ----------------------------------------------------
    # Suite 1: Standard Benchmark
    # ----------------------------------------------------
    print("==================================================")
    print("Suite 1: Standard Benchmark (10 conversations)")
    print("==================================================")

    # Clean state dir before test
    if config.state_dir.exists():
        shutil.rmtree(config.state_dir)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    base_std = BaselineAgent(config, force_offline=True)
    row_base_std = run_agent_benchmark("Baseline Agent", base_std, std_conversations, config)

    # Clean state dir for fair advanced start
    if config.state_dir.exists():
        shutil.rmtree(config.state_dir)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    adv_std = AdvancedAgent(config, force_offline=True)
    row_adv_std = run_agent_benchmark("Advanced Agent", adv_std, std_conversations, config)

    print(format_rows([row_base_std, row_adv_std]))
    print()

    # ----------------------------------------------------
    # Suite 2: Long-Context Stress Benchmark
    # ----------------------------------------------------
    print("==================================================")
    print("Suite 2: Long-Context Stress Benchmark (Stress test)")
    print("==================================================")

    # Clean state dir
    if config.state_dir.exists():
        shutil.rmtree(config.state_dir)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    base_stress = BaselineAgent(config, force_offline=True)
    row_base_stress = run_agent_benchmark("Baseline Agent", base_stress, stress_conversations, config)

    # Clean state dir
    if config.state_dir.exists():
        shutil.rmtree(config.state_dir)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    adv_stress = AdvancedAgent(config, force_offline=True)
    row_adv_stress = run_agent_benchmark("Advanced Agent", adv_stress, stress_conversations, config)

    print(format_rows([row_base_stress, row_adv_stress]))
    print()


if __name__ == "__main__":
    main()
