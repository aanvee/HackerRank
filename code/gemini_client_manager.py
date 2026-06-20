import os
import logging
import threading
import time
from google import genai

class GeminiClientManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.keys = []
        self.clients = []
        self.status = []  # List of {"name": ..., "key": ..., "status": "available"/"unavailable", "unavailable_until": 0.0}
        self.current_index = 0
        self._initialized = False
        
    def _ensure_initialized(self):
        """Lazy initialization - loads keys on first use (after load_dotenv has run)."""
        if self._initialized:
            return
        with self.lock:
            if self._initialized:
                return
            self._load_keys()
            self._initialized = True

    def _load_keys(self):
        found_keys = []
        # Check GEMINI_API_KEY_1 to GEMINI_API_KEY_5
        for i in range(1, 6):
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                found_keys.append((f"KEY {i}", key))
                
        # Fallback to GEMINI_API_KEY if no numbered keys are found
        if not found_keys:
            fallback_key = os.getenv("GEMINI_API_KEY")
            if fallback_key:
                found_keys.append(("DEFAULT KEY", fallback_key))
                
        self.keys = found_keys
        self.clients = [genai.Client(api_key=key_val) for _, key_val in found_keys]
        self.status = [
            {
                "name": name, 
                "key": key_val, 
                "status": "available", 
                "unavailable_until": 0.0
            } for name, key_val in found_keys
        ]
            
        logging.info(f"GeminiClientManager initialized. Loaded {len(found_keys)} API keys.")

    def reload_keys(self):
        """Force reload of keys (e.g. after load_dotenv)."""
        with self.lock:
            self._load_keys()
            self._initialized = True

    def get_client(self):
        self._ensure_initialized()
        with self.lock:
            if not self.keys:
                raise Exception("No Gemini API keys currently available. (None loaded from environment variables)")
                
            now = time.time()
            num_keys = len(self.keys)
            
            # Search for available key starting at current_index
            for i in range(num_keys):
                idx = (self.current_index + i) % num_keys
                stat = self.status[idx]
                
                # Check for cooldown expiration
                if stat["status"] == "unavailable" and now > stat["unavailable_until"]:
                    stat["status"] = "available"
                    logging.info(f"[{stat['name']}] Status reset to available after cooldown.")
                    
                if stat["status"] == "available":
                    self.current_index = idx
                    logging.debug(f"[{stat['name']}] Activated.")
                    return self.clients[idx]
                    
            # All keys are exhausted
            raise Exception("No Gemini API keys currently available.")

    def rotate_key(self, failed_client, reason):
        self._ensure_initialized()
        with self.lock:
            failed_idx = None
            for idx, client in enumerate(self.clients):
                if client == failed_client:
                    failed_idx = idx
                    break
                    
            if failed_idx is None:
                return
                
            stat = self.status[failed_idx]
            name = stat["name"]
            now = time.time()
            
            reason_upper = str(reason).upper()
            is_quota = "RESOURCE_EXHAUSTED" in reason_upper or "429" in reason_upper or "QUOTA" in reason_upper
            is_unavailable = "UNAVAILABLE" in reason_upper or "503" in reason_upper
            
            if is_quota:
                stat["status"] = "unavailable"
                stat["unavailable_until"] = now + 90.0  # Cool down for 90 seconds
                print(f"[{name}] Quota exhausted")
                logging.warning(f"[{name}] Quota exhausted ({reason}). Marking unavailable for 90s.")
            elif is_unavailable:
                stat["status"] = "unavailable"
                stat["unavailable_until"] = now + 15.0  # Cool down for 15 seconds
                print(f"[{name}] Request failed, switching")
                logging.warning(f"[{name}] Activated and request failed (503/UNAVAILABLE). Switching to next key.")
            else:
                logging.warning(f"[{name}] General error: {reason}. Rotating client key.")
                
            # Move to next key index
            self.current_index = (failed_idx + 1) % len(self.keys)

# Global manager instance (lazy - keys loaded on first use)
client_manager = GeminiClientManager()

