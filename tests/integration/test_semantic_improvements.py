#!/usr/bin/env python3
"""
Test script to validate semantic analysis improvements.
Tests the specific false positive patterns mentioned in the improvements document.
"""

import requests
import json

# Test cases that should demonstrate semantic analysis improvements
test_cases = [
    {
        "name": "PyTorch Model eval() - Should NOT flag as dangerous",
        "code": """
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
model.eval()  # This should NOT be flagged as dangerous
""",
        "expected_eval_issues": 0,
        "description": "PyTorch model.eval() should be recognized as safe method call"
    },
    
    {
        "name": "Dangerous eval() function - Should flag as dangerous",
        "code": """
user_input = "print('hello')"
result = eval(user_input)  # This SHOULD be flagged as dangerous
""",
        "expected_eval_issues": 1,
        "description": "Direct eval() function call should be flagged as security risk"
    },
    
    {
        "name": "RL Environment with proper reset - Should NOT flag missing reset",
        "code": """
import gym

env = gym.make('CartPole-v1')
for episode in range(10):
    state = env.reset()  # Reset is called properly
    for step in range(100):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        if done:
            break
""",
        "expected_rl_issues": 0,
        "description": "Proper RL environment usage should not flag reset issues"
    },
    
    {
        "name": "RL Environment missing reset - Should flag missing reset",
        "code": """
import gym

env = gym.make('CartPole-v1')
for episode in range(10):
    # Missing env.reset() here - should be flagged
    for step in range(100):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        if done:
            break
""",
        "expected_rl_issues": 1,
        "description": "Missing env.reset() in RL loop should be detected"
    },
    
    {
        "name": "NumPy random with seed - Should NOT flag missing seed",
        "code": """
import numpy as np

np.random.seed(42)  # Seed is set properly
data = np.random.rand(100)
""",
        "expected_seed_issues": 0,
        "description": "NumPy random with proper seeding should not be flagged"
    },
    
    {
        "name": "NumPy random without seed - Should flag missing seed",
        "code": """
import numpy as np

# No seed set - should be flagged
data = np.random.rand(100)
""",
        "expected_seed_issues": 1,
        "description": "NumPy random without seeding should be flagged"
    }
]

def test_semantic_analysis():
    """Test the semantic analysis improvements."""
    print("🧪 Testing Semantic Analysis Improvements")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"📝 Description: {test_case['description']}")
        
        # Prepare request
        request_data = {
            "files": [
                {
                    "filename": f"test_{i}.py",
                    "content": test_case['code']
                }
            ]
        }
        
        try:
            # Make request to audit endpoint
            response = requests.post(
                f"{base_url}/audit",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Analyze results
                eval_issues = []
                rl_issues = []
                seed_issues = []
                
                for issue in data.get('issues', []):
                    description = issue['description'].lower()
                    if 'eval' in description:
                        eval_issues.append(issue)
                    elif 'reset' in description or 'env.step' in description:
                        rl_issues.append(issue)
                    elif 'seed' in description or 'random' in description:
                        seed_issues.append(issue)
                
                print(f"🔍 Found {len(data.get('issues', []))} total issues")
                print(f"   - Eval-related: {len(eval_issues)}")
                print(f"   - RL-related: {len(rl_issues)}")
                print(f"   - Seed-related: {len(seed_issues)}")
                
                # Validate expectations
                success = True
                
                if 'expected_eval_issues' in test_case:
                    expected = test_case['expected_eval_issues']
                    actual = len(eval_issues)
                    if actual == expected:
                        print(f"   ✅ Eval detection: {actual} issues (expected {expected})")
                    else:
                        print(f"   ❌ Eval detection: {actual} issues (expected {expected})")
                        success = False
                
                if 'expected_rl_issues' in test_case:
                    expected = test_case['expected_rl_issues']
                    actual = len(rl_issues)
                    if actual == expected:
                        print(f"   ✅ RL detection: {actual} issues (expected {expected})")
                    else:
                        print(f"   ❌ RL detection: {actual} issues (expected {expected})")
                        success = False
                
                if 'expected_seed_issues' in test_case:
                    expected = test_case['expected_seed_issues']
                    actual = len(seed_issues)
                    if actual == expected:
                        print(f"   ✅ Seed detection: {actual} issues (expected {expected})")
                    else:
                        print(f"   ❌ Seed detection: {actual} issues (expected {expected})")
                        success = False
                
                if success:
                    print(f"   🎯 Test PASSED")
                else:
                    print(f"   🚨 Test FAILED")
                    
                    # Show actual issues for debugging
                    if eval_issues:
                        print("   📋 Eval issues found:")
                        for issue in eval_issues:
                            print(f"      - Line {issue['line']}: {issue['description']}")
                    
                    if rl_issues:
                        print("   📋 RL issues found:")
                        for issue in rl_issues:
                            print(f"      - Line {issue['line']}: {issue['description']}")
                    
                    if seed_issues:
                        print("   📋 Seed issues found:")
                        for issue in seed_issues:
                            print(f"      - Line {issue['line']}: {issue['description']}")
                
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
        
        print("-" * 40)
    
    print("\n🎯 Semantic Analysis Test Complete")
    print("The AST-based system should show significant improvements in:")
    print("• Distinguishing dangerous eval() from safe model.eval()")
    print("• Detecting missing env.reset() in proper context")
    print("• Accurate random seed detection across frameworks")

if __name__ == "__main__":
    test_semantic_analysis()