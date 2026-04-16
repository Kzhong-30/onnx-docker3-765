#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("Running Image Classification Model Performance Test Suite")
    print("=" * 60)
    
    os.makedirs("test_results", exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--html=test_results/test_report.html",
        "--self-contained-html",
        "--junitxml=test_results/test_report.xml"
    ]
    
    print(f"\nExecuting: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd, capture_output=False)
    
    print("-" * 60)
    if result.returncode == 0:
        print("✅ All tests passed!")
        print(f"📊 HTML report generated: test_results/test_report.html")
        print(f"📊 XML report generated: test_results/test_report.xml")
    else:
        print(f"❌ Some tests failed with exit code: {result.returncode}")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
