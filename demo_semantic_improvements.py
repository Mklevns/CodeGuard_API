"""
Comprehensive demonstration of AST-based semantic analysis improvements.
Shows how the new system solves the specific false positive patterns mentioned in the analysis document.
"""

import requests
import json
import time

def run_comprehensive_demo():
    """Run a comprehensive demonstration of semantic analysis improvements."""
    
    print("CodeGuard AST-Based Semantic Analysis Demonstration")
    print("=" * 60)
    print("This demonstration validates the new semantic analysis system")
    print("against the specific false positive patterns identified in the")
    print("technical analysis document.")
    print()
    
    # Test cases based on the document examples
    test_cases = [
        {
            "name": "PyTorch Model Evaluation",
            "description": "Tests distinction between safe model.eval() and dangerous eval()",
            "code": '''
import torch
import torch.nn as nn

# Safe PyTorch model evaluation - should NOT be flagged
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 128)

vae = VAE()
vae.eval()  # This is a safe method call, not eval() function

# Another safe case
model = nn.Sequential(nn.Linear(10, 5))
model.eval()  # Also safe

# Dangerous case - SHOULD be flagged
user_input = "print('dangerous')"
result = eval(user_input)  # This should be detected as dangerous
'''
        },
        {
            "name": "RL Environment Usage Patterns", 
            "description": "Tests detection of improper env.step() without reset patterns",
            "code": '''
import gym

env = gym.make('CartPole-v1')

# Proper pattern - should be fine
for episode in range(10):
    obs = env.reset()  # Proper reset
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

# Improper nested pattern - should be flagged
for outer in range(5):
    for inner in range(20):
        # Missing env.reset() here
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)  # This should be flagged
'''
        },
        {
            "name": "Random Seed Detection",
            "description": "Tests detection of unseeded random operations",
            "code": '''
import random
import numpy as np
import torch

# Unseeded operations - should be flagged
random_val = random.randint(1, 100)
numpy_val = np.random.rand(10)
torch_val = torch.rand(5)

# Properly seeded operations - should be fine
random.seed(42)
np.random.seed(42) 
torch.manual_seed(42)

seeded_random = random.randint(1, 100)
seeded_numpy = np.random.rand(10)
seeded_torch = torch.rand(5)
'''
        }
    ]
    
    base_url = "https://codeguard.replit.app"
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print("-" * 40)
        print(f"Description: {test_case['description']}")
        print()
        
        # Prepare request
        request = {
            "files": [
                {
                    "filename": f"test_{i}.py",
                    "content": test_case['code']
                }
            ],
            "options": {
                "level": "strict",
                "framework": "pytorch",
                "target": "gpu"
            }
        }
        
        try:
            # Test with semantic analysis (default)
            print("Running with AST-based semantic analysis...")
            start_time = time.time()
            
            response = requests.post(
                f"{base_url}/audit",
                json=request,
                headers={"Content-Type": "application/json"},
                timeout=45
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"Analysis completed in {elapsed:.1f}s")
                print(f"Summary: {data['summary']}")
                
                # Analyze results for this test case
                analyze_test_results(test_case, data['issues'])
                
            else:
                print(f"Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Test failed: {e}")
        
        print()
        print("=" * 60)
        print()
    
    # Summary validation
    print("DEMONSTRATION SUMMARY")
    print("=" * 30)
    print("The AST-based semantic analysis system provides:")
    print("1. Context-aware eval() detection (distinguishes functions vs methods)")
    print("2. Structural analysis of RL environment usage patterns") 
    print("3. Intelligent random seed detection across frameworks")
    print("4. Significant reduction in false positives while maintaining security detection")
    print()
    print("This upgrade moves CodeGuard from simple text pattern matching")
    print("to true semantic understanding of Python code structure.")

def analyze_test_results(test_case, issues):
    """Analyze test results for semantic analysis validation."""
    
    test_name = test_case['name']
    
    if "PyTorch Model" in test_name:
        # Check eval-related issues
        eval_issues = [issue for issue in issues if 'eval' in issue['description'].lower()]
        print(f"Eval-related issues found: {len(eval_issues)}")
        
        dangerous_eval_detected = False
        safe_eval_flagged = False
        
        for issue in eval_issues:
            line_context = get_code_line(test_case['code'], issue['line'])
            print(f"  Line {issue['line']}: {issue['description']}")
            print(f"  Code: {line_context.strip()}")
            
            if 'eval(' in line_context and 'user_input' in line_context:
                dangerous_eval_detected = True
                print("    -> Correctly identified dangerous eval() function")
            elif '.eval()' in line_context:
                safe_eval_flagged = True
                print("    -> WARNING: Safe model.eval() method flagged")
            print()
        
        print(f"Validation: Dangerous eval detected: {dangerous_eval_detected}")
        print(f"Validation: Safe eval methods flagged: {safe_eval_flagged}")
        
    elif "RL Environment" in test_name:
        # Check RL-related issues
        rl_issues = [issue for issue in issues if any(keyword in issue['description'].lower() 
                    for keyword in ['env', 'reset', 'step', 'environment'])]
        print(f"RL environment issues found: {len(rl_issues)}")
        
        for issue in rl_issues:
            line_context = get_code_line(test_case['code'], issue['line'])
            print(f"  Line {issue['line']}: {issue['description']}")
            print(f"  Code: {line_context.strip()}")
            print()
            
    elif "Random Seed" in test_name:
        # Check seed-related issues
        seed_issues = [issue for issue in issues if any(keyword in issue['description'].lower()
                      for keyword in ['seed', 'random', 'reproducibility'])]
        print(f"Seed/reproducibility issues found: {len(seed_issues)}")
        
        for issue in seed_issues:
            line_context = get_code_line(test_case['code'], issue['line'])
            print(f"  Line {issue['line']}: {issue['description']}")
            print(f"  Code: {line_context.strip()}")
            print()

def get_code_line(code, line_number):
    """Get a specific line from code."""
    lines = code.split('\n')
    if 1 <= line_number <= len(lines):
        return lines[line_number - 1]
    return ""

def compare_with_lexical_analysis():
    """Compare AST-based analysis with traditional lexical analysis."""
    
    print("COMPARISON: AST vs Lexical Analysis")
    print("=" * 40)
    print("Traditional lexical analysis (text pattern matching):")
    print("- Flags ANY occurrence of 'eval' substring")
    print("- Cannot distinguish between eval() function and .eval() method")
    print("- Results in false positives on safe PyTorch model.eval()")
    print()
    print("AST-based semantic analysis:")
    print("- Parses code into Abstract Syntax Tree")
    print("- Understands function calls vs method calls")
    print("- Recognizes object types and context")
    print("- Dramatically reduces false positives")
    print()
    print("Example improvement:")
    print("  vae.eval()     -> Lexical: FLAGGED | Semantic: SAFE")
    print("  eval(input)    -> Lexical: FLAGGED | Semantic: FLAGGED")

if __name__ == "__main__":
    run_comprehensive_demo()
    print()
    compare_with_lexical_analysis()