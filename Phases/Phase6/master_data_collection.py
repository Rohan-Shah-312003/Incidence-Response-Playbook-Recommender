
import subprocess
import sys
from pathlib import Path
import argparse


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent


def run_command(command, description, skip=False):
    if skip:
        print(f"\n⏭️  Skipping: {description}")
        return True

    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)

    try:
        # We set cwd=SCRIPT_DIR so the sub-scripts find their siblings
        result = subprocess.run(
            command, check=True, capture_output=False, text=True, cwd=SCRIPT_DIR
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code {e.returncode}")
        return False


def check_prerequisites():
    print("=" * 70)
    print("CHECKING PREREQUISITES")
    print("=" * 70)

    # These files are in the same folder as this script (Phase6)
    required_files = [
        SCRIPT_DIR / "download_veris_incidents.py",
        SCRIPT_DIR / "scrape_ddos_multisource.py",
        SCRIPT_DIR / "merge_all_datasets.py",
    ]

    missing = []
    for file_path in required_files:
        if file_path.exists():
            print(f"✓ {file_path.name}")
        else:
            print(f"❌ {file_path.name} - NOT FOUND at {file_path}")
            missing.append(file_path.name)

    if missing:
        return False

    # VCDB is in the ROOT directory based on your image
    vcdb_path = ROOT_DIR / "VCDB"
    if vcdb_path.exists():
        print(f"✓ VERIS repository (VCDB) found at {vcdb_path}")
    else:
        print("⚠️  VERIS repository not found in root")

    return True


def clone_veris():
    vcdb_path = ROOT_DIR / "VCDB"
    if vcdb_path.exists():
        return True

    # Run git clone inside the ROOT directory
    return run_command(
        ["git", "clone", "https://github.com/vz-risk/VCDB.git"],
        "Clone VERIS Community Database",
    )


def collect_veris(skip=False):
    return run_command(
        [sys.executable, "download_veris_incidents.py"],
        "Process VERIS incidents",
        skip=skip,
    )


def collect_ddos(skip=False):
    return run_command(
        [sys.executable, "scrape_ddos_multisource.py"],
        "Collect DDoS incidents",
        skip=skip,
    )


def merge_datasets():
    return run_command(
        [sys.executable, "merge_all_datasets.py"],
        "Merge and balance all incident datasets",
    )


def verify_output():
    print("\n" + "=" * 70)
    print("VERIFYING OUTPUT")
    print("=" * 70)

    # Based on your image, 'data' is in the root directory
    output_file = ROOT_DIR / "data" / "real_incidents_balanced.csv"

    if output_file.exists():
        import pandas as pd

        df = pd.read_csv(output_file)
        print(f"✓ Output file created: {output_file}")
        print(f"  Total incidents: {len(df)}")
        return True
    else:
        print(f"❌ Output file not found at: {output_file}")
        return False


def main():
    """
    Main workflow
    """
    parser = argparse.ArgumentParser(
        description="Complete data collection pipeline for IRPR"
    )
    parser.add_argument(
        "--skip-veris", action="store_true", help="Skip VERIS data collection"
    )
    parser.add_argument(
        "--skip-ddos", action="store_true", help="Skip DDoS data collection"
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=600,
        help="Target samples per class (default: 600)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("IRPR - MASTER DATA COLLECTION PIPELINE")
    print("=" * 70)
    print()
    print("This script will:")
    print("  1. Clone VERIS repository (if needed)")
    print("  2. Process VERIS incidents")
    print("  3. Scrape DDoS incidents")
    print("  4. Merge with existing CERT/Enron data")
    print("  5. Balance classes and save final dataset")
    print()

    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please ensure all files are present.")
        sys.exit(1)

    # Clone VERIS if needed
    if not args.skip_veris:
        if not clone_veris():
            print("\n⚠️  VERIS clone failed, but continuing...")

    # Step 1: Collect VERIS data
    if not collect_veris(skip=args.skip_veris):
        print("\n⚠️  VERIS collection failed, but continuing...")

    # Step 2: Collect DDoS data
    if not collect_ddos(skip=args.skip_ddos):
        print("\n⚠️  DDoS collection failed, but continuing...")

    # Step 3: Merge everything
    if not merge_datasets():
        print("\n❌ Dataset merging failed")
        sys.exit(1)

    # Step 4: Verify output
    if not verify_output():
        print("\n❌ Output verification failed")
        sys.exit(1)

    # Success
    print("\n" + "=" * 70)
    print("✓ DATA COLLECTION PIPELINE COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Review dataset:")
    print("     cat ./data/real_incidents_balanced_v2_summary.txt")
    print()
    print("  2. Retrain models:")
    print("     python Phases/Phase5/train_and_evaluate_pipeline.py")
    print()
    print("  3. Evaluate improvement:")
    print("     python Phases/Phase5/generate_plots.py")
    print()
    print("  4. Update config to use new dataset:")
    print("     Edit train_and_evaluate_pipeline.py")
    print("     Change: data_path='./data/real_incidents_balanced.csv'")
    print()


if __name__ == "__main__":
    main()
