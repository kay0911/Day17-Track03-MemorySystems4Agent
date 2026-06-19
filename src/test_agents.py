from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def make_config(tmp_path: Path):
    """Build an isolated config for tests."""
    config = load_config(Path(__file__).resolve().parent.parent)
    config.state_dir = tmp_path
    (tmp_path / "profiles").mkdir(parents=True, exist_ok=True)
    config.compact_threshold_tokens = 40
    config.compact_keep_messages = 2
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    from memory_store import UserProfileStore
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test-user"
    
    # Write
    store.write_text(user_id, "hello world")
    assert store.read_text(user_id) == "hello world"
    
    # Edit
    changed = store.edit_text(user_id, "world", "friend")
    assert changed is True
    assert store.read_text(user_id) == "hello friend"
    
    # Size
    assert store.file_size(user_id) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    user_id = "test-user"
    thread_id = "test-thread"
    
    # Total characters should exceed 160 characters (approx 40 tokens)
    msg = "A" * 100
    agent.reply(user_id, thread_id, msg)
    agent.reply(user_id, thread_id, msg)
    agent.reply(user_id, thread_id, msg)
    
    assert agent.compaction_count(thread_id) > 0
    ctx = agent.compact_memory.context(thread_id)
    assert len(ctx["messages"]) <= 2
    assert len(ctx["summary"]) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    
    user_id = "user-123"
    
    # Baseline run
    baseline.reply(user_id, "session-1", "Chào bạn, mình tên là DũngCT.")
    res_base = baseline.reply(user_id, "session-2", "Tên mình là gì?")
    assert "dũngct" not in res_base["response"].lower()
    
    # Advanced run
    advanced.reply(user_id, "session-1", "Chào bạn, mình tên là DũngCT.")
    res_adv = advanced.reply(user_id, "session-2", "Tên mình là gì?")
    assert "dũngct" in res_adv["response"].lower()


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    
    user_id = "test-user"
    thread_id = "thread-long"
    
    # Generate messages that exceed threshold so advanced triggers compaction
    msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 5  # ~200 characters each
    
    for _ in range(5):
        baseline.reply(user_id, thread_id, msg)
        advanced.reply(user_id, thread_id, msg)
        
    assert advanced.prompt_token_usage(thread_id) < baseline.prompt_token_usage(thread_id)

