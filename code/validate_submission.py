import sys
import pandas as pd
from pathlib import Path

# Setup paths relative to script location
code_dir = Path(__file__).parent.resolve()
repo_root = code_dir.parent.resolve()

sys.path.append(str(code_dir))
from utils import ALLOWED_PARTS, ALLOWED_ISSUES, ALLOWED_RISK_FLAGS, ALLOWED_STATUSES

def validate_submission():
    output_path = repo_root / "output.csv"
    claims_path = repo_root / "dataset" / "claims.csv"

    print("="*60)
    print("SUBMISSION VALIDATION AUDIT")
    print("="*60)

    # 1. Check if output.csv exists
    if not output_path.exists():
        print("✗ ERROR: output.csv does not exist in the repository root!")
        sys.exit(1)
        
    df_output = pd.read_csv(output_path)
    
    # 2. Check row count matches claims.csv
    if claims_path.exists():
        df_claims = pd.read_csv(claims_path)
        expected_rows = len(df_claims)
        actual_rows = len(df_output)
        if actual_rows != expected_rows:
            print(f"✗ ERROR: Row count mismatch! claims.csv has {expected_rows} rows, but output.csv has {actual_rows} rows.")
            sys.exit(1)
        else:
            print(f"✓ Row count matches claims.csv ({expected_rows} rows).")
    else:
        print("⚠ WARNING: claims.csv not found to verify row count.")

    # 3. Check schema columns and ordering
    expected_cols = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
        "issue_type", "object_part", "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity"
    ]
    
    actual_cols = list(df_output.columns)
    if actual_cols != expected_cols:
        print("✗ ERROR: Column list or ordering mismatch!")
        print("Expected:", expected_cols)
        print("Got:     ", actual_cols)
        sys.exit(1)
    else:
        print("✓ Columns and schema ordering are correct.")

    # 4. Check for null or empty values
    errors = 0
    null_cols = df_output.columns[df_output.isnull().any()].tolist()
    if null_cols:
        print(f"✗ ERROR: Null values found in columns: {null_cols}")
        for col in null_cols:
            null_indices = df_output[df_output[col].isnull()].index.tolist()
            print(f"  -> Null rows for {col}: indices {null_indices}")
        errors += 1
    else:
        print("✓ No null or missing values found in output.csv.")

    # 5. Validate enum and standard values
    for idx, row in df_output.iterrows():
        # Validate claim_object
        obj = str(row["claim_object"]).strip().lower()
        if obj not in ["car", "laptop", "package"]:
            print(f"✗ ERROR [Row {idx}]: Invalid claim_object '{obj}'")
            errors += 1
            
        # Validate claim_status
        status = str(row["claim_status"]).strip().lower()
        if status not in ALLOWED_STATUSES:
            print(f"✗ ERROR [Row {idx}]: Invalid claim_status '{status}'")
            errors += 1
            
        # Validate evidence_standard_met
        ev = str(row["evidence_standard_met"]).strip().lower()
        if ev not in ["true", "false"]:
            print(f"✗ ERROR [Row {idx}]: Invalid evidence_standard_met '{ev}' (must be string 'true' or 'false')")
            errors += 1
            
        # Validate valid_image
        vi = str(row["valid_image"]).strip().lower()
        if vi not in ["true", "false"]:
            print(f"✗ ERROR [Row {idx}]: Invalid valid_image '{vi}' (must be string 'true' or 'false')")
            errors += 1

        # Validate issue_type
        issue = str(row["issue_type"]).strip().lower()
        if issue not in ALLOWED_ISSUES:
            print(f"✗ ERROR [Row {idx}]: Invalid issue_type '{issue}'")
            errors += 1
            
        # Validate severity
        sev = str(row["severity"]).strip().lower()
        if sev not in ["none", "low", "medium", "high", "unknown"]:
            print(f"✗ ERROR [Row {idx}]: Invalid severity '{sev}'")
            errors += 1

        # Validate object_part
        part = str(row["object_part"]).strip().lower()
        allowed_parts = ALLOWED_PARTS.get(obj, [])
        if part not in allowed_parts:
            print(f"✗ ERROR [Row {idx}]: Invalid object_part '{part}' for object type '{obj}'")
            errors += 1

        # Validate risk_flags
        rflags = [f.strip().lower() for f in str(row["risk_flags"]).split(";") if f.strip()]
        for flag in rflags:
            if flag not in ALLOWED_RISK_FLAGS:
                print(f"✗ ERROR [Row {idx}]: Invalid risk flag '{flag}'")
                errors += 1

        # Validate supporting_image_ids format
        sids = str(row["supporting_image_ids"]).strip()
        if not sids:
            print(f"✗ ERROR [Row {idx}]: Empty supporting_image_ids field!")
            errors += 1

    # Print final diagnosis
    print("-" * 60)
    if errors > 0:
        print(f"✗ AUDIT FAILED! Found {errors} validation errors. Correct them before submitting.")
        sys.exit(1)
    else:
        print("✓ AUDIT PASSED! output.csv is 100% ready for submission.")
        print("=" * 60)

if __name__ == "__main__":
    validate_submission()
