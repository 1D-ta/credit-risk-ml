#!/usr/bin/env python3
"""
Validate that the development environment is properly set up.
Run this after cloning the repository to ensure all dependencies are installed.
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version is 3.9+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_imports():
    """Check that all required packages can be imported"""
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("prometheus_client", "prometheus_client"),
        ("pytest", "pytest"),
        ("requests", "requests"),
    ]
    
    all_ok = True
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            print(f"❌ {package_name} not installed")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """Check that required directories and files exist"""
    project_root = Path(__file__).resolve().parents[1]
    
    required_paths = [
        "data/raw",
        "data/schemas/schema.json",
        "artifacts/models",
        "artifacts/reports",
        "artifacts/logs",
        "credit_risk_ml/__init__.py",
        "inference/__init__.py",
        "training/__init__.py",
        "monitoring/__init__.py",
        "governance",
        "tests",
        "requirements.txt",
        "Makefile",
    ]
    
    all_ok = True
    for path_str in required_paths:
        path = project_root / path_str
        if path.exists():
            print(f"✓ {path_str} exists")
        else:
            print(f"❌ {path_str} missing")
            all_ok = False
    
    return all_ok

def check_data_files():
    """Check that required data files exist"""
    project_root = Path(__file__).resolve().parents[1]
    
    data_file = project_root / "data" / "raw" / "german_credit_raw.txt"
    if data_file.exists():
        print(f"✓ Raw data file exists")
        return True
    else:
        print(f"⚠ Raw data file missing: {data_file}")
        print(f"  This is expected if you haven't added the data yet.")
        print(f"  Download from: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)")
        return True  # Not a critical error

def main():
    print("=" * 60)
    print("Credit Risk ML - Setup Validation")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_imports),
        ("Project Structure", check_project_structure),
        ("Data Files", check_data_files),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 40)
        results.append(check_func())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All checks passed! Environment is ready.")
        print("\nNext steps:")
        print("  1. Run 'make generate-temporal' to create timestamped data")
        print("  2. Run 'make train' to train the model")
        print("  3. Run 'make test' to verify everything works")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nTo install dependencies:")
        print("  make setup    # Create venv and install")
        print("  OR")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())