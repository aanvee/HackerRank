import re
import json
import time
import logging
import hashlib
from pathlib import Path

# Setup Cache Location relative to repo root
# Repo root is parent of parent (i.e. parent of code/)
REPO_ROOT = Path(__file__).parent.parent.resolve()
CACHE_FILE = REPO_ROOT / "cache" / "claim_cache.json"
_cache_data = None

def get_repo_root():
    return REPO_ROOT

def load_cache():
    global _cache_data
    if _cache_data is not None:
        return _cache_data
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _cache_data = json.load(f)
                return _cache_data
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
    _cache_data = {}
    return _cache_data

def save_cache():
    global _cache_data
    if _cache_data is None:
        return
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_cache_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving cache: {e}")

def get_cache_key(prefix, *args):
    hasher = hashlib.md5()
    for arg in args:
        if isinstance(arg, bytes):
            hasher.update(arg)
        else:
            hasher.update(str(arg).encode("utf-8"))
    return f"{prefix}_{hasher.hexdigest()}"

def get_cached_response(key):
    cache = load_cache()
    return cache.get(key)

def set_cached_response(key, val):
    cache = load_cache()
    cache[key] = val
    save_cache()

def print_startup_summary():
    from gemini_client_manager import client_manager
    client_manager._ensure_initialized()
    cache = load_cache()
    num_keys = len(client_manager.keys)
    num_cache = len(cache)
    print(f"Loaded {num_keys} Gemini API keys")
    print(f"Cache entries: {num_cache}")
    logging.info(f"Startup Summary: Loaded {num_keys} Gemini API keys, Cache entries: {num_cache}")

# Global counter to track API usage and token metrics
API_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "calls": 0,
    "images": 0
}

# Setup logging
def setup_logging():
    log_file = Path(__file__).parent / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

# Allowed values and constraints
ALLOWED_ISSUES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown"
]

ALLOWED_SEVERITIES = ["none", "low", "medium", "high", "unknown"]
ALLOWED_STATUSES = ["supported", "contradicted", "not_enough_information"]
ALLOWED_RISK_FLAGS = [
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required"
]

ALLOWED_PARTS = {
    "car": [
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown"
    ],
    "laptop": [
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base", "body", "unknown"
    ],
    "package": [
        "box", "package_corner", "package_side", "seal", "label", "contents", "item", "unknown"
    ]
}

# Clean and parse JSON safely
def parse_json_safely(text):
    text = text.strip()
    # Strip markdown code blocks
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt simple cleanup for trailing commas before brackets/braces
        cleaned = re.sub(r",\s*([\]}])", r"\1", text)
        return json.loads(cleaned)

# Retry decorator with exponential backoff and transparent key rotation
def retry_api_call(max_retries=5, initial_backoff=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            from gemini_client_manager import client_manager
            
            backoff = initial_backoff
            args_list = list(args)
            
            for attempt in range(max_retries):
                # Always fetch the currently active client dynamically
                try:
                    active_client = client_manager.get_client()
                except Exception as ex:
                    # If no keys are available
                    logging.critical(f"No Gemini API keys currently available.")
                    raise ex
                
                # Replace the client positional argument (args[0]) with the active client
                if args_list and hasattr(args_list[0], "models"):
                    args_list[0] = active_client
                
                try:
                    return func(*args_list, **kwargs)
                except Exception as e:
                    err_msg = str(e).upper()
                    is_rate_limit = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "QUOTA" in err_msg or "UNAVAILABLE" in err_msg or "503" in err_msg
                    
                    logging.warning(f"API call failed on attempt {attempt + 1}/{max_retries}: {e}")
                    
                    # Notify manager of failure to trigger rotation
                    try:
                        client_manager.rotate_key(active_client, e)
                    except Exception as rot_e:
                        logging.error(f"Error updating key rotation state: {rot_e}")

                    if attempt == max_retries - 1:
                        raise e
                    
                    if is_rate_limit:
                        # Sleep slightly to let rotation settle
                        sleep_time = 1.5 + (backoff * 0.5)
                        time.sleep(sleep_time)
                    else:
                        time.sleep(backoff)
                    backoff *= 2
        return wrapper
    return decorator
