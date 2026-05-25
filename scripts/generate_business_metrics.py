#!/usr/bin/env python3
"""
Command-line tool to generate business metrics report from inference logs.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.business_metrics import generate_business_metrics_report


def main():
    parser = argparse.ArgumentParser(description="Generate business metrics report from inference logs")
    parser.add_argument("--log-file", type=Path, default=Path("artifacts/logs/predictions.jsonl"),
                        help="Path to inference logs JSONL file")
    parser.add_argument("--output", type=Path, default=Path("reports/business_metrics.md"),
                        help="Output markdown file")
    
    args = parser.parse_args()
    
    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        print(f"Tip: Run the inference API first to generate predictions")
        return 1
    
    print(f"Generating business metrics report from {args.log_file}...")
    md = generate_business_metrics_report(args.log_file, args.output)
    
    if args.output:
        print(f"Report saved to {args.output}")
    
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
