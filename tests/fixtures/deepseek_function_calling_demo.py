"""
DeepSeek Function Calling Integration Demo for CodeGuard API.
Demonstrates the enhanced multi-tool analysis capabilities with DeepSeek Function Calling.
"""

import json
import sys
import os
sys.path.append('.')

from chatgpt_integration import MultiLLMCodeImprover, CodeImprovementRequest
from models import Issue, Fix

def create_test_cases():
    """Create comprehensive test cases for Function Calling demo."""
    
    # Test Case 1: Security-focused code with multiple vulnerabilities
    security_test_code = """
import pickle
import subprocess
import os

def load_model(filename):
    # Security issue: unsafe deserialization
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def execute_command(user_input):
    # Security issue: command injection
    result = os.system(f"echo {user_input}")
    return result

def process_data(data):
    # Security issue: eval usage
    result = eval(f"process_{data}")
    return result
"""

    # Test Case 2: ML/RL code with best practice issues
    ml_test_code = """
import torch
import numpy as np
import gym

def train_model(data):
    # Missing reproducibility setup
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Memory inefficiency
    device = torch.device('cuda')
    model = model.cuda()
    
    for epoch in range(100):
        loss = model(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

def train_rl_agent():
    env = gym.make('CartPole-v1')
    
    # Missing environment reset
    for episode in range(1000):
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
    
    env.close()
"""

    # Test Case 3: Performance optimization code
    performance_test_code = """
import pandas as pd
import numpy as np

def process_large_dataset(df):
    results = []
    
    # Inefficient loop-based processing
    for index, row in df.iterrows():
        result = row['value'] * 2 + row['other']
        results.append(result)
    
    # Memory inefficient list operations
    large_list = []
    for i in range(1000000):
        large_list.append(i * 2)
    
    return results, large_list

def gpu_training_loop(model, data):
    # GPU memory inefficiency
    for batch in data:
        batch = batch.cuda()
        output = model(batch)
        loss = compute_loss(output)
        # Missing memory cleanup
"""

    return [
        ("security_test.py", security_test_code),
        ("ml_best_practices.py", ml_test_code), 
        ("performance_optimization.py", performance_test_code)
    ]

def run_function_calling_demo():
    """Run comprehensive demonstration of DeepSeek Function Calling."""
    print("🔧 DeepSeek Function Calling Integration Demo")
    print("=" * 60)
    
    # Initialize the improved code analyzer
    improver = MultiLLMCodeImprover()
    
    test_cases = create_test_cases()
    
    for filename, code in test_cases:
        print(f"\n📁 Testing: {filename}")
        print("-" * 40)
        
        # Create mock issues for demonstration
        issues = [
            Issue(
                filename=filename,
                line=5,
                type="security",
                description="Security vulnerability detected",
                source="security_scanner",
                severity="error"
            )
        ]
        
        fixes = [
            Fix(
                filename=filename,
                line=5,
                suggestion="Use safer serialization methods",
                auto_fixable=False
            )
        ]
        
        # Create improvement request
        request = CodeImprovementRequest(
            original_code=code,
            filename=filename,
            issues=issues,
            fixes=fixes,
            improvement_level="aggressive",
            ai_provider="deepseek",
            ai_api_key="demo_key_for_testing"
        )
        
        print(f"📊 Original code length: {len(code)} characters")
        print(f"🎯 Detected issues: {len(issues)}")
        print(f"🔧 Available fixes: {len(fixes)}")
        
        try:
            # Test the Function Calling integration
            print("🚀 Running DeepSeek Function Calling analysis...")
            
            # Since we don't have a real API key, simulate the function calling workflow
            simulate_function_calling_workflow(request)
            
        except Exception as e:
            print(f"⚠️  Demo completed with simulation: {str(e)}")
    
    print("\n✅ DeepSeek Function Calling Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("• Multi-tool security analysis")
    print("• ML/RL best practices generation")
    print("• Performance optimization suggestions")
    print("• Framework-specific improvements")
    print("• Comprehensive error handling")

def simulate_function_calling_workflow(request):
    """Simulate the Function Calling workflow for demonstration."""
    print("🔍 Simulating Function Calling workflow...")
    
    # Simulate security analysis
    print("  📋 analyze_code_security()")
    security_results = {
        "security_issues": ["Pickle usage detected", "Command injection risk"],
        "recommendations": ["Use JSON serialization", "Validate inputs"],
        "risk_level": "high"
    }
    print(f"    ➤ Found {len(security_results['security_issues'])} security issues")
    
    # Simulate ML best practices
    print("  🧠 generate_ml_best_practices()")
    ml_results = {
        "best_practices": ["Add random seeding", "Use device-agnostic code"],
        "code_improvements": ["torch.manual_seed(42)", "device = torch.device(...)"],
        "framework_specific": "pytorch"
    }
    print(f"    ➤ Generated {len(ml_results['best_practices'])} improvements")
    
    # Simulate performance optimization
    print("  ⚡ optimize_code_performance()")
    perf_results = {
        "optimizations": ["Vectorize operations", "Use GPU efficiently"],
        "performance_tips": ["Avoid iterrows()", "Clear GPU cache"],
        "target": "memory"
    }
    print(f"    ➤ Found {len(perf_results['optimizations'])} optimizations")
    
    print("  📝 Generating comprehensive improved code...")
    print("  ✅ Function Calling workflow completed successfully")

def test_function_implementations():
    """Test the individual function implementations."""
    print("\n🧪 Testing Function Implementations")
    print("-" * 40)
    
    improver = MultiLLMCodeImprover()
    
    # Test security analysis
    test_code = "import pickle\ndata = pickle.load(open('file.pkl', 'rb'))"
    security_result = improver._analyze_code_security(test_code, ["pickle"])
    print(f"Security Analysis: {len(security_result['security_issues'])} issues found")
    
    # Test ML best practices
    ml_code = "import torch\nmodel = torch.nn.Linear(10, 1)"
    ml_result = improver._generate_ml_best_practices(ml_code, "pytorch", [])
    print(f"ML Best Practices: {len(ml_result['best_practices'])} recommendations")
    
    # Test performance optimization
    perf_code = "for i in range(1000): results.append(i)"
    perf_result = improver._optimize_code_performance(perf_code, "speed")
    print(f"Performance Optimization: {len(perf_result['optimizations'])} suggestions")

if __name__ == "__main__":
    run_function_calling_demo()
    test_function_implementations()