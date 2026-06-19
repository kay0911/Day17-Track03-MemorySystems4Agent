from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if self.force_offline or not self.langchain_agent:
            return self._reply_offline(thread_id, message)
        
        try:
            session = self.sessions.setdefault(thread_id, SessionState())
            
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            messages = [SystemMessage(content="Bạn là Baseline Agent. Hãy trả lời ngắn gọn.")]
            for m in session.messages:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                else:
                    messages.append(AIMessage(content=m["content"]))
            messages.append(HumanMessage(content=message))
            
            response = self.langchain_agent.invoke(messages)
            reply_text = response.content
            
            session.messages.append({"role": "user", "content": message})
            session.prompt_tokens_processed += sum(estimate_tokens(msg.content) for msg in messages)
            session.token_usage += estimate_tokens(reply_text)
            session.messages.append({"role": "assistant", "content": reply_text})
            
            return {"response": reply_text}
        except Exception:
            return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent token count for one thread."""
        session = self.sessions.get(thread_id)
        if not session:
            return 0
        return session.token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Estimate how much prompt context this baseline kept processing."""
        session = self.sessions.get(thread_id)
        if not session:
            return 0
        return session.prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _extract_facts_from_messages(self, messages: list[dict[str, str]]) -> dict[str, str]:
        """Extract facts purely from messages in the current thread."""
        facts = {}
        for msg in messages:
            if msg["role"] == "user":
                updates = extract_profile_updates(msg["content"])
                facts.update(updates)
        return facts

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Implement a simple offline behavior."""
        session = self.sessions.setdefault(thread_id, SessionState())
        
        # Calculate prompt tokens processed before adding the new user message
        prompt_tokens_this_turn = sum(estimate_tokens(msg["content"]) for msg in session.messages)
        session.prompt_tokens_processed += prompt_tokens_this_turn
        
        # Append new user message
        session.messages.append({"role": "user", "content": message})
        
        # Build answer using only within-session facts
        facts = self._extract_facts_from_messages(session.messages)
        msg_lower = message.lower()
        parts = []
        
        if any(k in msg_lower for k in ["tên", "ở đâu", "nơi ở", "nghề", "công việc", "uống", "món ăn", "ăn", "nuôi", "con gì", "style", "trả lời"]):
            if "tên" in msg_lower:
                if facts.get("name"):
                    parts.append(f"Tên bạn là {facts['name']}")
                else:
                    parts.append("Tôi không biết tên bạn")
            if "ở đâu" in msg_lower or "nơi ở" in msg_lower:
                if facts.get("location"):
                    parts.append(f"nơi ở hiện tại là {facts['location']}")
                else:
                    parts.append("Tôi không biết nơi ở của bạn")
            if "nghề" in msg_lower or "công việc" in msg_lower:
                if facts.get("profession"):
                    parts.append(f"nghề nghiệp của bạn là {facts['profession']}")
                else:
                    parts.append("Tôi không biết nghề nghiệp của bạn")
            if "uống" in msg_lower or "đồ uống" in msg_lower:
                if facts.get("favorite_drink"):
                    parts.append(f"đồ uống yêu thích là {facts['favorite_drink']}")
                else:
                    parts.append("Tôi không biết đồ uống yêu thích của bạn")
            if "món ăn" in msg_lower or "ăn" in msg_lower:
                if facts.get("favorite_food"):
                    parts.append(f"món ăn yêu thích là {facts['favorite_food']}")
                else:
                    parts.append("Tôi không biết món ăn yêu thích của bạn")
            if "nuôi" in msg_lower or "con gì" in msg_lower:
                if facts.get("pet"):
                    parts.append(f"bạn nuôi một bé {facts['pet']}")
                else:
                    parts.append("Tôi không biết bạn nuôi con gì")
            if "style" in msg_lower or "trả lời" in msg_lower:
                if facts.get("response_style"):
                    parts.append(f"style trả lời thích là {facts['response_style']}")
                else:
                    parts.append("Tôi không biết style trả lời bạn thích")
            reply_text = ", ".join(parts) + "."
        else:
            reply_text = "Chào bạn, tôi là Baseline Agent và đã ghi nhận thông tin."
            
        # Append assistant reply
        session.messages.append({"role": "assistant", "content": reply_text})
        
        # Update response tokens
        reply_tokens = estimate_tokens(reply_text)
        session.token_usage += reply_tokens
        
        return {"response": reply_text}

    def _maybe_build_langchain_agent(self):
        """Optionally wire model builder here."""
        try:
            self.langchain_agent = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None

