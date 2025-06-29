"""
Test script for the new AST-based semantic analysis system.
Validates that the semantic analyzer correctly distinguishes between dangerous and safe code patterns.
"""

import requests
import json

def test_semantic_analysis():
    """Test the semantic analysis system with the examples from the false positive report."""
    
    # Sample code that demonstrates the false positive patterns mentioned in the document
    sample_code = '''
import torch
import torch.nn as nn
import numpy as np
import gym
import random

# Safe PyTorch model evaluation - should NOT be flagged as dangerous
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 128)
        
    def forward(self, x):
        return self.encoder(x)

# Create model and set to evaluation mode (safe .eval() method call)
vae = VAE()
vae.eval()  # This should NOT be flagged as dangerous eval()

# Another safe model evaluation
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
model.eval()  # This should also be safe

# Dangerous built-in eval() function - SHOULD be flagged
dangerous_input = "print('hello from eval')"
result = eval(dangerous_input)  # This SHOULD be flagged as dangerous

# RL environment usage pattern - proper reset/step usage
env = gym.make('CartPole-v1')

# Proper episode loop with reset (should be fine)
for episode in range(10):
    obs = env.reset()  # Proper reset at episode start
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)  # Proper step usage
        if done:
            break

# Improper nested loop without reset - should be flagged
for outer_loop in range(5):
    for inner_step in range(20):
        # Missing env.reset() here - semantic analysis should detect this
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

# Random operations without seeding - should be flagged for reproducibility
random_value = random.randint(1, 100)  # Missing seed
numpy_random = np.random.rand(10)      # Missing seed

# Proper seeding (should be fine)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
seeded_value = random.randint(1, 100)  # This should be fine
'''

    test_request = {
        "files": [
            {
                "filename": "semantic_test.py",
                "content": sample_code
            }
        ],
        "options": {
            "level": "strict",
            "framework": "pytorch",
            "target": "gpu"
        }
    }
    
    base_url = "https://codeguard.replit.app"
    
    print("Testing AST-Based Semantic Analysis System")
    print("=" * 50)
    print("This test validates the semantic analyzer can distinguish between:")
    print("✓ Safe model.eval() vs dangerous eval() function calls")
    print("✓ Proper RL environment reset/step patterns vs improper usage")
    print("✓ Missing random seeds vs properly seeded operations")
    print()
    
    try:
        # Test with semantic analysis enabled (default behavior)
        print("🔍 Running semantic analysis...")
        response = requests.post(
            f"{base_url}/audit",
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=45
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analysis completed successfully")
            print(f"📊 Summary: {data['summary']}")
            print(f"🔢 Total Issues Found: {len(data['issues'])}")
            print()
            
            # Analyze the results to validate semantic analysis
            eval_issues = []
            rl_issues = []
            seed_issues = []
            security_issues = []
            
            for issue in data['issues']:
                description = issue['description'].lower()
                if 'eval' in description:
                    eval_issues.append(issue)
                elif 'env.step' in description or 'reset' in description:
                    rl_issues.append(issue)
                elif 'seed' in description:
                    seed_issues.append(issue)
                elif issue['severity'] == 'error' and 'security' in issue.get('type', ''):
                    security_issues.append(issue)
            
            print("🎯 Semantic Analysis Results:")
            print("-" * 30)
            
            # Validate eval() detection
            print(f"📍 Eval-related issues: {len(eval_issues)}")
            for issue in eval_issues:
                line_context = get_line_context(sample_code, issue['line'])
                print(f"   Line {issue['line']}: {issue['description']}")
                print(f"   Code: {line_context.strip()}")
                
                # Check if semantic analysis correctly identified dangerous vs safe eval
                if 'dangerous' in issue['description'].lower():
                    print(f"   ✅ Correctly identified as dangerous")
                else:
                    print(f"   ⚠️  Classification: {issue['severity']}")
                print()
            
            # Validate RL environment analysis
            print(f"🎮 RL Environment issues: {len(rl_issues)}")
            for issue in rl_issues:
                line_context = get_line_context(sample_code, issue['line'])
                print(f"   Line {issue['line']}: {issue['description']}")
                print(f"   Code: {line_context.strip()}")
                print(f"   ✅ Detected improper env.step() without reset")
                print()
            
            # Validate seed detection
            print(f"🎲 Reproducibility issues: {len(seed_issues)}")
            for issue in seed_issues:
                line_context = get_line_context(sample_code, issue['line'])
                print(f"   Line {issue['line']}: {issue['description']}")
                print(f"   Code: {line_context.strip()}")
                print()
            
            # Validate security issues
            print(f"🔒 Security issues: {len(security_issues)}")
            for issue in security_issues:
                line_context = get_line_context(sample_code, issue['line'])
                print(f"   Line {issue['line']}: {issue['description']}")
                print(f"   Code: {line_context.strip()}")
                print(f"   Severity: {issue['severity']}")
                print()
            
            # Summary validation
            print("📋 Validation Summary:")
            print("-" * 20)
            
            # Check if we have the expected pattern detections
            has_dangerous_eval = any('dangerous' in issue['description'].lower() for issue in eval_issues)
            has_rl_issues = len(rl_issues) > 0
            has_missing_seeds = len(seed_issues) > 0
            
            if has_dangerous_eval:
                print("✅ Correctly identified dangerous eval() function call")
            else:
                print("❌ Failed to identify dangerous eval() - may need tuning")
            
            if has_rl_issues:
                print("✅ Detected RL environment usage issues")
            else:
                print("❌ Failed to detect RL environment issues")
            
            if has_missing_seeds:
                print("✅ Found missing random seed issues")
            else:
                print("❌ No seed-related issues detected")
            
            # Count safe model.eval() calls that should NOT be flagged
            safe_eval_lines = get_lines_containing(sample_code, ['.eval()', 'vae.eval()', 'model.eval()'])
            flagged_safe_evals = sum(1 for line_num in safe_eval_lines 
                                   if any(issue['line'] == line_num for issue in eval_issues))
            
            print(f"🛡️  Safe model.eval() calls: {len(safe_eval_lines)} found, {flagged_safe_evals} incorrectly flagged")
            
            if flagged_safe_evals == 0:
                print("✅ No false positives on safe model.eval() calls")
            else:
                print("❌ False positives detected on safe eval methods")
            
            print()
            print("🎉 Semantic Analysis Test Completed!")
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

def get_line_context(code: str, line_number: int) -> str:
    """Get the code at a specific line number."""
    lines = code.split('\n')
    if 1 <= line_number <= len(lines):
        return lines[line_number - 1]
    return ""

def get_lines_containing(code: str, patterns: list) -> list:
    """Get line numbers containing any of the specified patterns."""
    lines = code.split('\n')
    matching_lines = []
    
    for i, line in enumerate(lines, 1):
        if any(pattern in line for pattern in patterns):
            matching_lines.append(i)
    
    return matching_lines

if __name__ == "__main__":
    test_semantic_analysis()