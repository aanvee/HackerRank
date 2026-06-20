import logging
import pandas as pd
from pathlib import Path
from utils import get_repo_root

# Load requirements from CSV
def load_evidence_requirements():
    repo_root = get_repo_root()
    csv_path = repo_root / "dataset" / "evidence_requirements.csv"
    try:
        if csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            logging.warning(f"evidence_requirements.csv not found at {csv_path}")
            return None
    except Exception as e:
        logging.error(f"Error loading evidence requirements: {e}")
        return None

def get_evidence_requirements_text(claim_object, claimed_issue, claimed_part, num_images):
    df = load_evidence_requirements()
    if df is None:
        return "Verify that the images clearly show the claimed object and relevant parts to assess the damage."

    applicable_rules = []
    
    # 1. Add general rules
    general_rules = df[(df["claim_object"] == "all") & (df["applies_to"] == "general claim review")]
    for _, row in general_rules.iterrows():
        applicable_rules.append(f"- {row['requirement_id']}: {row['minimum_image_evidence']}")

    trust_rules = df[(df["claim_object"] == "all") & (df["applies_to"] == "reviewability")]
    for _, row in trust_rules.iterrows():
        applicable_rules.append(f"- {row['requirement_id']}: {row['minimum_image_evidence']}")

    # 2. Add multi-image rule if applicable
    if num_images > 1:
        multi_rules = df[(df["claim_object"] == "all") & (df["applies_to"] == "multi-image rows")]
        for _, row in multi_rules.iterrows():
            applicable_rules.append(f"- {row['requirement_id']}: {row['minimum_image_evidence']}")

    # 3. Add object-specific rules based on issue or part
    obj_rules = df[df["claim_object"] == claim_object]
    for _, row in obj_rules.iterrows():
        applies_to = row["applies_to"].lower()
        
        # Car logic
        if claim_object == "car":
            if "dent" in applies_to or "scratch" in applies_to:
                if claimed_issue in ["dent", "scratch"]:
                    applicable_rules.append(f"- {row['requirement_id']}: {row['minimum_image_evidence']}")
            elif "crack" in applies_to or "broken" in applies_to or "missing" in applies_to:
                if claimed_issue in ["crack", "glass_shatter", "broken_part", "missing_part"]:
                    applicable_rules.append(f"- {row['requirement_id']}: {row['minimum_image_evidence']}")
            elif "identity" in applies_to or "orientation" in applies_to:
                # Car identity rule applies generally to check vehicle matches
                applicable_rules.append(f"- {row['requirement_id']}: {row['minimum_image_evidence']}")
                
        # Laptop logic
        elif claim_object == "laptop":
            if "screen" in applies_to or "keyboard" in applies_to or "trackpad" in applies_to:
                if claimed_part in ["screen", "keyboard", "trackpad"] or claimed_issue in ["stain", "crack", "glass_shatter"]:
                    applicable_rules.append(f"- {row['requirement_id']}: {row['minimum_image_evidence']}")
            elif "hinge" in applies_to or "lid" in applies_to or "corner" in applies_to or "body" in applies_to or "port" in applies_to:
                if claimed_part in ["hinge", "lid", "corner", "port", "base", "body"] or claimed_issue in ["dent"]:
                    applicable_rules.append(f"- {row['requirement_id']}: {row['minimum_image_evidence']}")

        # Package logic
        elif claim_object == "package":
            if "crushed" in applies_to or "torn" in applies_to or "seal" in applies_to:
                if claimed_issue in ["crushed_packaging", "torn_packaging", "broken_part"] or claimed_part in ["seal", "box", "package_corner"]:
                    applicable_rules.append(f"- {row['requirement_id']}: {row['minimum_image_evidence']}")
            elif "water" in applies_to or "stain" in applies_to or "label" in applies_to:
                if claimed_issue in ["water_damage", "stain"] or claimed_part in ["label", "package_side"]:
                    applicable_rules.append(f"- {row['requirement_id']}: {row['minimum_image_evidence']}")
            elif "contents" in applies_to or "inner" in applies_to:
                if claimed_part in ["contents", "item"] or claimed_issue in ["missing_part"]:
                    applicable_rules.append(f"- {row['requirement_id']}: {row['minimum_image_evidence']}")

    # If no specific rules are found, append all object specific rules as fallback
    if len(applicable_rules) <= 3 and len(obj_rules) > 0:
        for _, row in obj_rules.iterrows():
            applicable_rules.append(f"- {row['requirement_id']}: {row['minimum_image_evidence']}")

    return "\n".join(applicable_rules)
