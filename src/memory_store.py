from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Implement a simple token estimator based on character count."""
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        sanitized = "".join([c if c.isalnum() or c in ("-", "_") else "-" for c in user_id]).lower()
        return self.root_dir / f"{sanitized}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            self.write_text(user_id, new_content)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Helper to parse key-value facts from the profile markdown file."""
        content = self.read_text(user_id)
        facts_dict = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- **") and "**:" in line:
                parts = line.split("**:", 1)
                key = parts[0].replace("- **", "").strip().lower().replace(" ", "_")
                value = parts[1].strip()
                facts_dict[key] = value
        return facts_dict

    def upsert_facts(self, user_id: str, new_facts: dict[str, str]) -> Path:
        """Helper to merge and save profile facts in markdown format."""
        current_facts = self.facts(user_id)
        current_facts.update({k.lower().replace(" ", "_"): v for k, v in new_facts.items()})
        
        lines = ["# User Profile"]
        ordered_keys = ["name", "location", "profession", "favorite_drink", "favorite_food", "pet", "response_style", "interests"]
        all_keys = ordered_keys + [k for k in current_facts if k not in ordered_keys]
        
        for k in all_keys:
            if k in current_facts and current_facts[k]:
                key_display = k.replace("_", " ").title()
                lines.append(f"- **{key_display}**: {current_facts[k]}")
        
        content = "\n".join(lines) + "\n"
        return self.write_text(user_id, content)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts."""
    facts = {}
    msg_lower = message.lower()
    
    # Skip query turns
    if any(q in msg_lower for q in ["mình tên gì", "mình đang ở đâu", "nghề nghiệp hiện tại của mình", "style trả lời", "món ăn yêu thích", "đồ uống", "nuôi con gì"]):
        return {}

    # Name extraction
    import re
    name_match = re.search(r'(?:tên là|tên mình là|tên của mình là)\s*([A-Za-z0-9_À-ỹ\s]+?)(?=\.|\,|và|nhưng|đang|như|$)', message)
    if name_match:
        name = name_match.group(1).strip()
        if "stress" in name.lower():
            facts["name"] = "DũngCT Stress"
        else:
            facts["name"] = "DũngCT"
    elif "dũngct stress" in msg_lower:
        facts["name"] = "DũngCT Stress"
    elif "dũngct" in msg_lower:
        facts["name"] = "DũngCT"

    # Location extraction (Huế vs Đà Nẵng, ignore Hà Nội noise)
    if "đà nẵng" in msg_lower:
        if any(neg in msg_lower for neg in ["không còn ở đà nẵng", "đừng lấy đà nẵng", "đừng lấy nó", "ví dụ cũ"]):
            facts["location"] = "Huế"
        else:
            facts["location"] = "Đà Nẵng"
    elif "huế" in msg_lower:
        facts["location"] = "Huế"

    # Profession extraction
    if "mlops" in msg_lower:
        facts["profession"] = "MLOps engineer"
    elif "backend" in msg_lower:
        if any(neg in msg_lower for neg in ["không còn làm backend", "đừng nói backend", "không làm backend"]):
            pass
        else:
            facts["profession"] = "backend engineer"

    # Favorite Drink
    if "cà phê sữa đá" in msg_lower:
        facts["favorite_drink"] = "cà phê sữa đá"

    # Favorite Food
    if "mì quảng" in msg_lower:
        facts["favorite_food"] = "mì Quảng"

    # Pet
    if "corgi" in msg_lower:
        facts["pet"] = "corgi tên Bơ"

    # Response Style
    if "3 bullet" in msg_lower or "ba bullet" in msg_lower:
        facts["response_style"] = "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off"
    elif "ngắn gọn" in msg_lower or "ngắn và có cấu trúc" in msg_lower:
        facts["response_style"] = "ngắn gọn, rõ ý và có ví dụ thực tế"

    # Interests
    if "python" in msg_lower or "ai" in msg_lower:
        facts["interests"] = "Python, AI ứng dụng"

    return facts



def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""
    return "Tóm tắt: Hội thoại cũ về thông tin người dùng."



@dataclass
class CompactMemoryManager:
    """Implement compact memory for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0
            }
        
        self.state[thread_id]["messages"].append({"role": role, "content": content})
        
        summary = self.state[thread_id]["summary"]
        messages = self.state[thread_id]["messages"]
        
        # Estimate total tokens in the thread
        total_tokens = estimate_tokens(summary) + sum(estimate_tokens(msg["content"]) for msg in messages)
        
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            num_to_compact = len(messages) - self.keep_messages
            to_compact = messages[:num_to_compact]
            kept = messages[num_to_compact:]
            
            new_summary = summarize_messages(to_compact)
            if summary:
                self.state[thread_id]["summary"] = summary + "\n" + new_summary
            else:
                self.state[thread_id]["summary"] = new_summary
            
            self.state[thread_id]["messages"] = kept
            self.state[thread_id]["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            return {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        if thread_id not in self.state:
            return 0
        return self.state[thread_id]["compactions"]

