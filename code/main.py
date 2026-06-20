import os
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from utils import setup_logging, get_repo_root, print_startup_summary
from gemini_client_manager import client_manager
from claim_processor import process_claim

def main():
    # 1. Setup paths and env vars
    repo_root = get_repo_root()
    env_path = repo_root / ".env"
    load_dotenv(env_path)
    
    setup_logging()
    logging.info("Starting Claim Verification System...")

    # 2. Print Startup Summary and check keys
    print_startup_summary()
    
    client_manager._ensure_initialized()
    if not client_manager.keys:
        logging.error("No Gemini API keys loaded from environment variables (GEMINI_API_KEY_1-5 or GEMINI_API_KEY).")
        return
        
    client = client_manager.get_client()

    # 3. Load dataset
    claims_csv_path = repo_root / "dataset" / "claims.csv"
    if not claims_csv_path.exists():
        logging.error(f"claims.csv not found at: {claims_csv_path}")
        return
        
    df = pd.read_csv(claims_csv_path)
    logging.info(f"Loaded {len(df)} claims from claims.csv")

    # 4. Process each row
    output_rows = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing claims"):
        try:
            # Re-obtain active client to match rotation state
            active_client = client_manager.get_client()
            result = process_claim(active_client, row)
            output_rows.append(result)
        except Exception as e:
            logging.error(f"Error processing row index {index} (user_id: {row.get('user_id')}): {e}")
            output_rows.append({
                "user_id": row.get("user_id"),
                "image_paths": row.get("image_paths"),
                "user_claim": row.get("user_claim"),
                "claim_object": row.get("claim_object"),
                "evidence_standard_met": "false",
                "evidence_standard_met_reason": "Execution error during processing.",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": f"System error during claim evaluation: {e}",
                "supporting_image_ids": "none",
                "valid_image": "false",
                "severity": "unknown"
            })

    # 5. Export to output.csv in the correct schema order
    output_df = pd.DataFrame(output_rows)
    cols_order = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
        "issue_type", "object_part", "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity"
    ]
    output_df = output_df[cols_order]
    
    output_csv_path = repo_root / "output.csv"
    output_df.to_csv(output_csv_path, index=False)
    logging.info(f"Successfully wrote {len(output_df)} predictions to {output_csv_path}")

if __name__ == "__main__":
    main()