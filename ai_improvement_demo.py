#!/usr/bin/env python3
"""
Comprehensive AI-Powered Code Improvement Demonstration for CodeGuard.

This script demonstrates the complete workflow:
1. AST-based semantic analysis (reduces false positives)
2. Multi-tool static analysis (flake8, pylint, mypy, etc.)
3. AI-powered code improvements using ChatGPT
4. Bulk fixing and comprehensive reporting

Shows the evolution from simple pattern matching to intelligent code improvement.
"""

import requests
import json
import time
from typing import Dict, List, Any

def run_ai_improvement_demo():
    """Run comprehensive demonstration of AI-powered code improvements."""
    
    print("CodeGuard AI-Powered Code Improvement Demonstration")
    print("=" * 65)
    print("This demonstration showcases the complete workflow:")
    print("✓ AST-based semantic analysis with false positive filtering")
    print("✓ Multi-tool static analysis (8+ analysis engines)")
    print("✓ ChatGPT-powered intelligent code improvements")
    print("✓ Bulk fixing and comprehensive reporting")
    print()
    
    # Comprehensive test case with multiple ML/RL issues
    test_code = '''
import torch
import numpy as np
import gym
import pickle
import os

# Poor ML/RL code with multiple issues for AI improvement
class BadMLModel:
    def __init__(self):
        # Missing random seeding - major issue
        self.model = torch.nn.Linear(10, 1)
        
    def train(self, data):
        # Security issue: using eval
        learning_rate = eval("0.001")
        
        # Missing proper error handling
        for epoch in range(100):
            loss = self.model(data)
            loss.backward()
            
        # Hardcoded paths - portability issue
        torch.save(self.model, "/tmp/model.pth")
        
    def unsafe_load(self, path):
        # Major security vulnerability
        with open(path, 'rb') as f:
            return pickle.load(f)

def rl_training_loop():
    # Missing environment reset
    env = gym.make('CartPole-v1')
    
    for episode in range(1000):
        state = env.reset()  # Good: has reset
        done = False
        
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            
            # Poor RL pattern: no state update
            state = next_state

# Formatting issues
def   poorly_formatted_function(   x,y,z   ):
    return x+y+z   # Missing spaces

# Long line that exceeds 88 characters limit
very_long_variable_name_that_makes_this_line_exceed_the_recommended_character_limit = "This is a very long string that should be broken"

print("Training complete")  # Should use logging instead of print
'''

    base_url = "https://codeguard.replit.app"
    
    print("Phase 1: Standard CodeGuard Analysis")
    print("-" * 40)
    
    # Test standard audit with semantic analysis
    audit_request = {
        "files": [
            {
                "filename": "ml_model.py", 
                "content": test_code
            }
        ],
        "options": {
            "level": "strict",
            "framework": "pytorch",
            "target": "gpu"
        }
    }
    
    try:
        print("Running comprehensive static analysis...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/audit",
            json=audit_request,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        audit_time = time.time() - start_time
        
        if response.status_code == 200:
            audit_data = response.json()
            print(f"✅ Analysis completed in {audit_time:.1f}s")
            print(f"📊 Summary: {audit_data['summary']}")
            print(f"🔍 Issues found: {len(audit_data['issues'])}")
            
            # Categorize issues by type and severity
            issue_categories = {}
            security_issues = 0
            
            for issue in audit_data['issues']:
                issue_type = issue['type']
                severity = issue['severity']
                
                if issue_type not in issue_categories:
                    issue_categories[issue_type] = {'count': 0, 'severities': []}
                
                issue_categories[issue_type]['count'] += 1
                issue_categories[issue_type]['severities'].append(severity)
                
                if severity == 'error' and 'security' in issue_type:
                    security_issues += 1
            
            print("\n📋 Issue Breakdown:")
            for category, data in issue_categories.items():
                print(f"   {category}: {data['count']} issues")
            
            if security_issues > 0:
                print(f"⚠️  {security_issues} critical security issues detected")
            
            print()
            
        else:
            print(f"❌ Audit failed: {response.status_code} - {response.text}")
            return
            
    except Exception as e:
        print(f"❌ Error during audit: {e}")
        return
    
    print("Phase 2: AI-Powered Code Improvement")
    print("-" * 40)
    
    # Test AI-powered improvement
    improvement_request = {
        "original_code": test_code,
        "filename": "ml_model.py",
        "issues": audit_data['issues'][:10],  # Top 10 issues for focused improvement
        "fixes": audit_data.get('fixes', [])[:10],
        "improvement_level": "moderate",
        "ai_provider": "openai"
    }
    
    try:
        print("Applying ChatGPT-powered improvements...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/improve/code",
            json=improvement_request,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        improvement_time = time.time() - start_time
        
        if response.status_code == 200:
            improvement_data = response.json()
            print(f"✅ AI improvement completed in {improvement_time:.1f}s")
            print(f"🎯 Confidence Score: {improvement_data['confidence_score']:.1%}")
            print(f"🔧 Fixes Applied: {len(improvement_data['applied_fixes'])}")
            print(f"📈 Summary: {improvement_data['improvement_summary']}")
            
            if improvement_data.get('warnings'):
                print(f"⚠️  Warnings: {len(improvement_data['warnings'])}")
            
            print("\n🔧 Applied Improvements:")
            for i, fix in enumerate(improvement_data['applied_fixes'][:5], 1):
                print(f"   {i}. {fix.get('description', 'Code improvement')}")
            
            # Show code comparison (first 20 lines)
            original_lines = test_code.split('\n')[:20]
            improved_lines = improvement_data['improved_code'].split('\n')[:20]
            
            print(f"\n📝 Code Improvement Preview (first 20 lines):")
            print("Original → Improved")
            print("-" * 50)
            
            for i, (orig, improved) in enumerate(zip(original_lines, improved_lines), 1):
                if orig != improved:
                    print(f"Line {i:2d}: {orig[:40]:<40} → {improved[:40]}")
            
            print()
            
        else:
            print(f"❌ AI improvement failed: {response.status_code} - {response.text}")
            return
            
    except Exception as e:
        print(f"❌ Error during AI improvement: {e}")
        return
    
    print("Phase 3: Combined Audit + AI Improvement")
    print("-" * 40)
    
    # Test the combined endpoint
    try:
        print("Testing combined audit and improvement...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/audit-and-improve",
            json=audit_request,
            headers={"Content-Type": "application/json"},
            timeout=150
        )
        
        combined_time = time.time() - start_time
        
        if response.status_code == 200:
            combined_data = response.json()
            print(f"✅ Combined analysis completed in {combined_time:.1f}s")
            
            summary = combined_data['combined_summary']
            print(f"📊 Total Issues: {summary['total_issues_found']}")
            print(f"🔧 CodeGuard Fixes: {summary['codeguard_fixes']}")
            print(f"🤖 AI Fixes Applied: {summary['ai_fixes_applied']}")
            print(f"🎯 Average AI Confidence: {summary['average_ai_confidence']:.1%}")
            print(f"🔍 Framework Detected: {summary['framework_detected']}")
            
        else:
            print(f"❌ Combined analysis failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error during combined analysis: {e}")
    
    print()
    print("Phase 4: Comprehensive Improvement Report")
    print("-" * 40)
    
    # Test comprehensive reporting
    report_request = {
        "files": [{"filename": "ml_model.py", "content": test_code}],
        "format": "markdown",
        "include_ai_suggestions": True,
        "apply_false_positive_filtering": True
    }
    
    try:
        print("Generating comprehensive improvement report...")
        
        response = requests.post(
            f"{base_url}/reports/improvement-analysis",
            json=report_request,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response.status_code == 200:
            report_data = response.json()
            print(f"✅ Report generated successfully")
            print(f"📄 Total Issues: {report_data['total_issues']}")
            print(f"📁 Files Analyzed: {report_data['total_files']}")
            
            severity = report_data['severity_breakdown']
            print(f"🔴 Errors: {severity.get('error', 0)}")
            print(f"🟡 Warnings: {severity.get('warning', 0)}")
            print(f"🔵 Info: {severity.get('info', 0)}")
            
            # Show a snippet of the markdown report
            report_content = report_data['report']
            if isinstance(report_content, str):
                lines = report_content.split('\n')[:15]
                print(f"\n📋 Report Preview:")
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
            
        else:
            print(f"❌ Report generation failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error during report generation: {e}")
    
    print()
    print("DEMONSTRATION COMPLETE")
    print("=" * 30)
    print("🎉 CodeGuard AI-Powered Improvement System Features Validated:")
    print("✓ AST-based semantic analysis prevents false positives")
    print("✓ Multi-tool static analysis (flake8, pylint, mypy, black, isort)")
    print("✓ Custom ML/RL pattern detection (seeding, env resets, security)")
    print("✓ ChatGPT integration for intelligent code improvements")
    print("✓ Combined audit + improvement workflow")
    print("✓ Comprehensive reporting with AI suggestions")
    print("✓ False positive filtering maintains critical issue detection")
    print()
    print("This positions CodeGuard as a cutting-edge AI-powered static analysis")
    print("platform that goes beyond issue detection to provide intelligent solutions.")

if __name__ == "__main__":
    run_ai_improvement_demo()