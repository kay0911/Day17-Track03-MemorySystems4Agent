from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


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
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear."""
    if not expected:
        return 1.0
    matched = 0
    ans_lower = answer.lower()
    for fact in expected:
        if fact.lower() in ans_lower:
            matched += 1
    return matched / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Add a lightweight quality score for offline mode."""
    rec = recall_points(answer, expected)
    if rec > 0.0:
        if len(answer) > 20:
            return min(1.0, rec + 0.1)
        return rec
    return 0.0


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations."""
    import shutil
    # Clean up the state directory before running to start fresh
    profiles_dir = config.state_dir / "profiles"
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    # Reset internal agent state for a clean run
    if hasattr(agent, "sessions"):
        agent.sessions = {}
    if hasattr(agent, "thread_tokens"):
        agent.thread_tokens = {}
    if hasattr(agent, "thread_prompt_tokens"):
        agent.thread_prompt_tokens = {}
    if hasattr(agent, "compact_memory"):
        agent.compact_memory.state = {}
        
    total_tokens_only = 0
    total_prompt_tokens = 0
    recall_scores = []
    quality_scores = []
    compactions = 0
    
    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        turns = conv["turns"]
        
        # Feed all turns in the same thread
        for turn in turns:
            agent.reply(user_id, conv_id, turn)
            
        # Accumulate token usage for the conversation thread
        total_tokens_only += agent.token_usage(conv_id)
        total_prompt_tokens += agent.prompt_token_usage(conv_id)
        compactions += agent.compaction_count(conv_id)
        
        # Ask recall questions in a fresh thread
        recall_qs = conv.get("recall_questions", [])
        for i, q in enumerate(recall_qs):
            recall_thread_id = f"{conv_id}-recall-{i}"
            res = agent.reply(user_id, recall_thread_id, q["question"])
            answer = res.get("response", "")
            
            r_score = recall_points(answer, q["expected_contains"])
            q_score = heuristic_quality(answer, q["expected_contains"])
            
            recall_scores.append(r_score)
            quality_scores.append(q_score)
            
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    # Calculate memory file size growth
    user_ids = {conv["user_id"] for conv in conversations}
    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        for uid in user_ids:
            memory_growth += agent.memory_file_size(uid)
            
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_tokens_only,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=compactions
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a markdown table or tabulated output."""
    from tabulate import tabulate
    table_data = []
    for r in rows:
        table_data.append([
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality * 100:.1f}%",
            r.memory_growth_bytes,
            r.compactions
        ])
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions"
    ]
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both benchmark suites."""
    import os
    config = load_config(Path(__file__).resolve().parent.parent)

    std_convs = load_conversations(config.data_dir / "conversations.json")
    stress_convs = load_conversations(config.data_dir / "advanced_long_context.json")
    
    force_offline = os.getenv("LLM_OFFLINE", "False").lower() == "true"
    print(f"Executing in {'OFFLINE' if force_offline else 'ONLINE'} mode...")
    
    baseline = BaselineAgent(config, force_offline=force_offline)
    advanced = AdvancedAgent(config, force_offline=force_offline)
    
    print("=== RUNNING STANDARD BENCHMARK ===")
    baseline_std = run_agent_benchmark("Baseline Agent", baseline, std_convs, config)
    advanced_std = run_agent_benchmark("Advanced Agent", advanced, std_convs, config)
    print(format_rows([baseline_std, advanced_std]))
    print()
    
    print("=== RUNNING LONG-CONTEXT STRESS BENCHMARK ===")
    baseline_stress = run_agent_benchmark("Baseline Agent", baseline, stress_convs, config)
    advanced_stress = run_agent_benchmark("Advanced Agent", advanced, stress_convs, config)
    print(format_rows([baseline_stress, advanced_stress]))


if __name__ == "__main__":
    main()

