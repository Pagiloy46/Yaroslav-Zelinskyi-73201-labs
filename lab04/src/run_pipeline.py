
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    src_dir = root / "lab04" / "src"

    steps = [
        "ingest_data.py",
        "validate_data.py",
        "audit_data.py",
        "preprocess_data.py",
        "finalize_data.py",
    ]

    for step in steps:
        script_path = src_dir / step

        print(f"\n{'=' * 60}")
        print(f"Running {step}")
        print("=" * 60)

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=root,
            check=False,
        )

        if result.returncode != 0:
            print(f"\nPipeline stopped: {step} failed.")
            raise SystemExit(result.returncode)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
