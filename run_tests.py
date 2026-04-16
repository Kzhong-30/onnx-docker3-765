#!/usr/bin/env python3
"""
自动化测试脚本 - 生成HTML测试报告
"""

import subprocess
import sys
import os
from datetime import datetime


def run_tests():
    """运行所有测试并生成HTML报告"""
    
    # 创建报告目录
    report_dir = "test_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # 生成带时间戳的报告文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_dir, f"test_report_{timestamp}.html")
    
    # 构建pytest命令
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--html=" + report_file,
        "--self-contained-html",
        "--tb=short",
        "-r", "a",  # 显示所有测试的简短摘要
    ]
    
    print("=" * 60)
    print("开始运行模型性能测试套件")
    print("=" * 60)
    print(f"报告将保存到: {report_file}")
    print("-" * 60)
    
    # 运行测试
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    print("-" * 60)
    if result.returncode == 0:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败")
    print(f"📊 测试报告: {report_file}")
    print("=" * 60)
    
    return result.returncode


def run_specific_tests(test_type):
    """运行特定类型的测试"""
    
    report_dir = "test_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_dir, f"test_report_{test_type}_{timestamp}.html")
    
    test_files = {
        "loading": "tests/test_model_loading.py",
        "latency": "tests/test_inference_latency.py",
        "accuracy": "tests/test_accuracy.py",
    }
    
    if test_type not in test_files:
        print(f"未知的测试类型: {test_type}")
        print(f"可用的类型: {', '.join(test_files.keys())}")
        return 1
    
    cmd = [
        sys.executable, "-m", "pytest",
        test_files[test_type],
        "-v",
        "--html=" + report_file,
        "--self-contained-html",
        "--tb=short",
    ]
    
    print(f"运行 {test_type} 测试...")
    result = subprocess.run(cmd, capture_output=False, text=True)
    print(f"报告保存到: {report_file}")
    
    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type in ["loading", "latency", "accuracy"]:
            sys.exit(run_specific_tests(test_type))
        else:
            print(f"用法: python run_tests.py [loading|latency|accuracy]")
            sys.exit(1)
    else:
        sys.exit(run_tests())
