from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None
        self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        if self.force_offline or not self.langchain_agent:
            return self._reply_offline(user_id, thread_id, message)
        
        try:
            # Sync user profile updates to disk even in live mode helper
            new_facts = extract_profile_updates(message)
            if new_facts:
                self.profile_store.upsert_facts(user_id, new_facts)
                
            # Append to compact memory
            self.compact_memory.append(thread_id, "user", message)
            
            # Read profile & context
            profile_text = self.profile_store.read_text(user_id)
            ctx = self.compact_memory.context(thread_id)
            summary = ctx.get("summary", "")
            recent_msgs = ctx.get("messages", [])
            
            # Build messages
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            system_prompt = f"Bạn là Advanced Agent.\n\nThông tin người dùng thu thập được:\n{profile_text}\n\nHãy trả lời ngắn gọn và tập trung vào các câu hỏi dựa trên thông tin trên."
            
            messages = [SystemMessage(content=system_prompt)]
            if summary:
                messages.append(SystemMessage(content=f"Tóm tắt các lượt hội thoại trước đó: {summary}"))
                
            for m in recent_msgs[:-1]: # exclude the current user message which is added at the end
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                else:
                    messages.append(AIMessage(content=m["content"]))
                    
            messages.append(HumanMessage(content=message))
            
            # Invoke LLM
            response = self.langchain_agent.invoke(messages)
            reply_text = response.content
            
            # Log tokens
            prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
            self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
            
            reply_tokens = estimate_tokens(reply_text)
            self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens
            
            # Append assistant reply
            self.compact_memory.append(thread_id, "assistant", reply_text)
            
            return {"response": reply_text}
        except Exception:
            return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Implement the deterministic advanced path."""
        # 1. Extract stable profile facts from the incoming message
        new_facts = extract_profile_updates(message)
        
        # 2. Persist those facts into User.md
        if new_facts:
            self.profile_store.upsert_facts(user_id, new_facts)
        
        # 3. Append the message into compact memory
        self.compact_memory.append(thread_id, "user", message)
        
        # 4. Estimate prompt-context load from User.md + summary + recent messages
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        
        # 5. Generate a response that can answer long-term recall questions
        reply_text = self._offline_response(user_id, thread_id, message)
        
        # 6. Append the assistant reply and update token counters
        self.compact_memory.append(thread_id, "assistant", reply_text)
        
        reply_tokens = estimate_tokens(reply_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens
        
        return {"response": reply_text}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn."""
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "")
        messages = ctx.get("messages", [])
        
        profile_tokens = estimate_tokens(profile_text)
        summary_tokens = estimate_tokens(summary)
        messages_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        
        return profile_tokens + summary_tokens + messages_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory."""
        facts = self.profile_store.facts(user_id)
        if not facts:
            return "Tôi không biết thông tin này."
        
        msg_lower = message.lower()
        parts = []
        
        if "tên" in msg_lower:
            parts.append(f"Tên bạn là {facts.get('name', '')}")
        if "ở đâu" in msg_lower or "nơi ở" in msg_lower or "huế" in msg_lower or "hà nội" in msg_lower:
            parts.append(f"nơi ở hiện tại là {facts.get('location', '')}")
        if "nghề" in msg_lower or "công việc" in msg_lower or "product manager" in msg_lower:
            parts.append(f"nghề nghiệp hiện tại là {facts.get('profession', '')}")
        if "uống" in msg_lower or "đồ uống" in msg_lower:
            parts.append(f"đồ uống yêu thích là {facts.get('favorite_drink', '')}")
        if "món ăn" in msg_lower or "ăn" in msg_lower:
            parts.append(f"món ăn yêu thích là {facts.get('favorite_food', '')}")
        if "nuôi" in msg_lower or "con gì" in msg_lower:
            parts.append(f"bạn nuôi một bé {facts.get('pet', '')}")
        if "style" in msg_lower or "trả lời" in msg_lower or "bullet" in msg_lower:
            parts.append(f"style trả lời thích là {facts.get('response_style', '')}")
        if "quan tâm" in msg_lower or "sở thích" in msg_lower or "thích" in msg_lower or "hữu ích" in msg_lower:
            if facts.get("interests"):
                parts.append(f"mối quan tâm chính là {facts.get('interests', '')}")
                
        if not parts:
            return "Chào bạn, tôi là Advanced Agent và đã ghi nhận thông tin của bạn."
        return ", ".join(parts) + "."

    def _maybe_build_langchain_agent(self):
        """Optionally wire model builder here."""
        try:
            self.langchain_agent = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None

