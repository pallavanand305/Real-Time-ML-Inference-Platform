#!/usr/bin/env python
"""Verify that the development environment is set up correctly"""

import sys
from pathlib import Path


def check_python_version() -> bool:
    """Check if Python version is 3.11+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.11+)")
        return False


def check_directory_structure() -> bool:
    """Check if required directories exist"""
    required_dirs = [
        "src",
        "src/inference_service",
        "src/model_server",
        "src/drift_detector",
        "src/common",
        "tests",
        "tests/unit",
        "tests/integration",
        "config",
        "k8s",
        "terraform",
        "docker",
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
            all_exist = False

    return all_exist


def check_config_files() -> bool:
    """Check if configuration files exist"""
    required_files = [
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        ".gitignore",
        ".pre-commit-config.yaml",
        "pytest.ini",
        "docker-compose.yaml",
        "config/development.yaml",
        "config/staging.yaml",
        "config/production.yaml",
        "config/prometheus.yaml",
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_exist = False

    return all_exist


def check_imports() -> bool:
    """Check if key dependencies can be imported"""
    dependencies = [
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
        ("redis", "Redis"),
        ("yaml", "PyYAML"),
    ]

    all_imported = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (not installed)")
            all_imported = False

    return all_imported


def main() -> None:
    """Run all verification checks"""
    print("=" * 60)
    print("Real-Time ML Inference Platform - Setup Verification")
    print("=" * 60)

    print("\n1. Checking Python version...")
    python_ok = check_python_version()

    print("\n2. Checking directory structure...")
    dirs_ok = check_directory_structure()

    print("\n3. Checking configuration files...")
    files_ok = check_config_files()

    print("\n4. Checking Python dependencies...")
    imports_ok = check_imports()

    print("\n" + "=" * 60)
    if python_ok and dirs_ok and files_ok and imports_ok:
        print("✓ All checks passed! Environment is ready.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗ Some checks failed. Please review the output above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
