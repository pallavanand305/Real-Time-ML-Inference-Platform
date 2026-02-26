#!/usr/bin/env python
"""Check if all Docker services are healthy"""

import sys
import time
from typing import Dict, List

import httpx


def check_service(name: str, url: str, timeout: int = 5) -> bool:
    """Check if a service is responding"""
    try:
        response = httpx.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✓ {name} is healthy")
            return True
        else:
            print(f"✗ {name} returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ {name} is not responding: {e}")
        return False


def main() -> None:
    """Check all services"""
    services = {
        "MLflow": "http://localhost:5000/health",
        "Prometheus": "http://localhost:9090/-/healthy",
        "Grafana": "http://localhost:3000/api/health",
    }

    print("=" * 60)
    print("Checking Docker Services Health")
    print("=" * 60)
    print("\nWaiting for services to start...")
    time.sleep(5)

    results: Dict[str, bool] = {}
    for name, url in services.items():
        print(f"\nChecking {name}...")
        results[name] = check_service(name, url)

    print("\n" + "=" * 60)
    healthy = sum(results.values())
    total = len(results)

    if healthy == total:
        print(f"✓ All services are healthy ({healthy}/{total})")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"✗ Some services are unhealthy ({healthy}/{total})")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
