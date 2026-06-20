import logging
import pandas as pd
from pathlib import Path
from utils import get_repo_root, ALLOWED_RISK_FLAGS

_user_history_cache = None

def load_user_history():
    global _user_history_cache
    if _user_history_cache is not None:
        return _user_history_cache
        
    repo_root = get_repo_root()
    csv_path = repo_root / "dataset" / "user_history.csv"
    try:
        if csv_path.exists():
            _user_history_cache = pd.read_csv(csv_path)
            return _user_history_cache
        else:
            logging.warning(f"user_history.csv not found at {csv_path}")
            return None
    except Exception as e:
        logging.error(f"Error loading user history: {e}")
        return None

def evaluate_user_risk(user_id):
    df = load_user_history()
    if df is None:
        return []

    user_row = df[df["user_id"] == user_id]
    if user_row.empty:
        return []

    row = user_row.iloc[0]
    history_flags_str = str(row.get("history_flags", ""))
    
    if pd.isna(row.get("history_flags")) or not history_flags_str or history_flags_str.lower() == "none":
        history_flags = []
    else:
        history_flags = [f.strip().lower() for f in history_flags_str.split(";") if f.strip()]
        
    return history_flags

def build_risk_flags(user_id, vision_result):
    flags = [str(f).lower().strip() for f in vision_result.get("risk_flags", [])]
    flags = [f for f in flags if f != "none"]

    user_flags = evaluate_user_risk(user_id)
    flags.extend(user_flags)

    if not vision_result.get("valid_image", True):
        flags.append("manual_review_required")
        
    if not vision_result.get("evidence_standard_met", True):
        reason = str(vision_result.get("evidence_standard_met_reason", "")).lower()
        if "angle" in reason or "view" in reason:
            flags.append("wrong_angle")
        if "crop" in reason or "obstruct" in reason or "cut" in reason or "block" in reason:
            flags.append("cropped_or_obstructed")
        flags.append("manual_review_required")

    valid_flags = []
    for f in flags:
        if f in ALLOWED_RISK_FLAGS and f != "none":
            valid_flags.append(f)
            
    valid_flags = sorted(list(set(valid_flags)))

    if not valid_flags:
        return ["none"]
    return valid_flags