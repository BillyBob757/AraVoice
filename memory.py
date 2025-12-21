import json
import os
import logging

logger = logging.getLogger("voice-agent")

MEMORY_FILE = "agent_memory.json"

class MemoryManager:
    @staticmethod
    def load_memory() -> str:
        """Loads usage history and summarizes it for context injection."""
        if not os.path.exists(MEMORY_FILE):
            return ""
        
        try:
            with open(MEMORY_FILE, "r") as f:
                history = json.load(f)
                
            # Convert last 5 interactions into a context string
            # Format: 'User said: ... \n Agent said: ...'
            if not history:
                return ""
            
            recent = history[-500:] # Keep last 500 turns for long-term memory (xAI has 128K context)
            memory_str = "\n[Previous Conversation Memory]:\n"
            for turn in recent:
                memory_str += f"{turn['role']}: {turn['text']}\n"
            
            return memory_str
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return ""

    @staticmethod
    def save_turn(role: str, text: str):
        """Appends a single turn to the memory file."""
        entry = {"role": role, "text": text}
        history = []
        
        # Read existing
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    history = json.load(f)
            except:
                history = []
        
        # Append and Save
        history.append(entry)
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
