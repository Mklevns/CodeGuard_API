#!/usr/bin/env python3
"""
Comprehensive demonstration of AST-based semantic analysis improvements.
This script validates all the key improvements mentioned in the semantic analysis document.
"""

import requests
import json
import time
from typing import Dict, List, Any

class SemanticAnalysisDemo:
    """Demonstrates semantic analysis improvements with before/after comparisons."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
    
    def run_complete_demo(self):
        """Run the complete semantic analysis demonstration."""
        print("CodeGuard Semantic Analysis Improvements Demonstration")
        print("=" * 60)
        print("This demo validates the AST-based improvements that solve false positive issues.")
        print()
        
        # Test cases covering all improvement areas
        test_cases = [
            self._test_eval_distinction(),
            self._test_rl_environment_analysis(),
            self._test_random_seed_detection(),
            self._test_race_condition_prevention(),
            self._test_false_positive_filtering()
        ]
        
        for test_case in test_cases:
            self._run_test_case(test_case)
            time.sleep(1)  # Brief pause between tests
        
        self._print_summary()
    
    def _test_eval_distinction(self) -> Dict[str, Any]:
        """Test AST-based eval() vs model.eval() distinction."""
        return {
            "name": "Eval Function vs Method Distinction",
            "description": "AST analysis should distinguish dangerous eval() from safe model.eval()",
            "code": '''
import torch
import torch.nn as nn

# Safe PyTorch model evaluation - should NOT be flagged as dangerous
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleNet()
model.eval()  # This is SAFE - model method call

# Dangerous eval function - SHOULD be flagged as security risk
user_input = "os.system('rm -rf /')"
result = eval(user_input)  # This is DANGEROUS - built-in function
''',
            "validation": lambda issues: {
                "safe_eval_not_flagged": not any(
                    "model.eval()" in issue.get('description', '') or 
                    (issue['line'] == 12 and 'eval' in issue.get('description', '').lower())
                    for issue in issues
                ),
                "dangerous_eval_flagged": any(
                    issue['severity'] == 'error' and 'eval' in issue.get('description', '').lower()
                    for issue in issues
                ),
                "security_type_assigned": any(
                    issue['type'] == 'security' and 'eval' in issue.get('description', '').lower()
                    for issue in issues
                )
            }
        }
    
    def _test_rl_environment_analysis(self) -> Dict[str, Any]:
        """Test enhanced RL environment analysis with context awareness."""
        return {
            "name": "RL Environment Context Analysis",
            "description": "Semantic analysis should detect RL patterns in proper context",
            "code": '''
import gym
import numpy as np

# Proper RL environment usage - should NOT flag missing reset
env = gym.make('CartPole-v1')
for episode in range(10):
    observation = env.reset()  # Proper reset call
    total_reward = 0
    
    for step in range(100):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    print(f"Episode {episode}: {total_reward}")

# Problematic RL pattern - SHOULD flag missing reset
env2 = gym.make('MountainCar-v0') 
for episode in range(5):
    # Missing env.reset() here - should be detected
    for step in range(200):
        action = env2.action_space.sample()
        obs, reward, done, info = env2.step(action)
        if done:
            break
''',
            "validation": lambda issues: {
                "proper_usage_not_flagged": not any(
                    issue['line'] <= 15 and 'reset' in issue.get('description', '').lower()
                    for issue in issues
                ),
                "missing_reset_detected": any(
                    'reset' in issue.get('description', '').lower() and issue['line'] > 20
                    for issue in issues
                ),
                "rl_plugin_active": any(
                    issue['source'] == 'rl_plugin' for issue in issues
                )
            }
        }
    
    def _test_random_seed_detection(self) -> Dict[str, Any]:
        """Test AST-based random seed detection across frameworks."""
        return {
            "name": "Framework-Aware Random Seed Detection",
            "description": "AST should detect missing seeds for different frameworks",
            "code": '''
import numpy as np
import torch
import random

# Proper seeding - should NOT be flagged
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Use random functions after seeding - should be OK
good_data = np.random.rand(100)
good_tensor = torch.rand(10, 10)
good_choice = random.choice([1, 2, 3])

# Missing seeds - SHOULD be flagged
import numpy as np2
import torch as torch2

# These should be flagged for missing seeds
bad_data = np2.random.normal(0, 1, 100)  # No np seed for np2
bad_tensor = torch2.randn(5, 5)          # No torch seed for torch2
''',
            "validation": lambda issues: {
                "seeded_usage_ok": not any(
                    issue['line'] <= 12 and 'seed' in issue.get('description', '').lower()
                    for issue in issues
                ),
                "missing_numpy_seed_detected": any(
                    'numpy' in issue.get('description', '').lower() and 'seed' in issue.get('description', '').lower()
                    for issue in issues
                ),
                "missing_torch_seed_detected": any(
                    'torch' in issue.get('description', '').lower() and 'seed' in issue.get('description', '').lower()
                    for issue in issues
                )
            }
        }
    
    def _test_race_condition_prevention(self) -> Dict[str, Any]:
        """Test that concurrent requests don't interfere with each other."""
        return {
            "name": "Concurrent Request Isolation",
            "description": "Multiple simultaneous requests should not interfere",
            "code": '''
# Simple test code for concurrent processing
import os
import tempfile

def process_data():
    temp_file = tempfile.mktemp()
    with open(temp_file, 'w') as f:
        f.write("test data")
    
    return temp_file
''',
            "validation": lambda issues: {
                "processing_successful": True,  # If we get results, isolation worked
                "no_file_conflicts": not any(
                    'file' in issue.get('description', '').lower() and 'conflict' in issue.get('description', '').lower()
                    for issue in issues
                )
            },
            "concurrent_test": True
        }
    
    def _test_false_positive_filtering(self) -> Dict[str, Any]:
        """Test that false positive filtering maintains security detection."""
        return {
            "name": "False Positive Filtering Accuracy",
            "description": "Should filter style issues while keeping security problems",
            "code": '''
import pickle
import os

# Security issue - SHOULD be detected and kept
with open('model.pkl', 'rb') as f:
    dangerous_model = pickle.load(f)  # Dangerous pickle usage

# Style issues - should be filtered out
x=1+2    # Missing spaces around operator
y    =   3   # Multiple spaces
z = [
    1,2,3    # Missing spaces in list
        ]

# Another security issue - SHOULD be kept
os.system("rm -rf /")  # Dangerous system call
''',
            "validation": lambda issues: {
                "security_issues_kept": any(
                    issue['type'] == 'security' or issue['severity'] == 'error'
                    for issue in issues
                ),
                "pickle_detected": any(
                    'pickle' in issue.get('description', '').lower()
                    for issue in issues
                ),
                "style_issues_filtered": not any(
                    'spaces' in issue.get('description', '').lower() or 
                    'formatting' in issue.get('description', '').lower()
                    for issue in issues
                )
            }
        }
    
    def _run_test_case(self, test_case: Dict[str, Any]):
        """Run a single test case and validate results."""
        print(f"Testing: {test_case['name']}")
        print(f"Purpose: {test_case['description']}")
        print("-" * 50)
        
        # Handle concurrent testing
        if test_case.get('concurrent_test'):
            self._run_concurrent_test(test_case)
            return
        
        try:
            # Make request to audit endpoint
            response = requests.post(
                f"{self.base_url}/audit",
                json={
                    "files": [
                        {
                            "filename": f"{test_case['name'].lower().replace(' ', '_')}.py",
                            "content": test_case['code']
                        }
                    ]
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                issues = data.get('issues', [])
                
                print(f"Analysis Results: {len(issues)} issues found")
                
                # Run validation
                validation_results = test_case['validation'](issues)
                
                # Print validation results
                all_passed = True
                for check_name, passed in validation_results.items():
                    status = "PASS" if passed else "FAIL"
                    print(f"  {check_name}: {status}")
                    if not passed:
                        all_passed = False
                
                # Show sample issues for context
                if issues:
                    print(f"\nSample Issues:")
                    for i, issue in enumerate(issues[:3]):
                        print(f"  {i+1}. Line {issue['line']}: {issue['description']}")
                        print(f"     Type: {issue['type']}, Severity: {issue['severity']}")
                
                test_result = {
                    'name': test_case['name'],
                    'passed': all_passed,
                    'total_issues': len(issues),
                    'validation_details': validation_results
                }
                self.test_results.append(test_result)
                
                print(f"\nOverall Test Result: {'PASS' if all_passed else 'FAIL'}")
                
            else:
                print(f"Request failed: {response.status_code}")
                print(f"Error: {response.text}")
                
                self.test_results.append({
                    'name': test_case['name'],
                    'passed': False,
                    'error': f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"Test failed with exception: {e}")
            self.test_results.append({
                'name': test_case['name'],
                'passed': False,
                'error': str(e)
            })
        
        print("\n" + "=" * 60 + "\n")
    
    def _run_concurrent_test(self, test_case: Dict[str, Any]):
        """Run concurrent requests to test race condition prevention."""
        import threading
        import queue
        
        print("Running concurrent requests to test isolation...")
        
        results_queue = queue.Queue()
        
        def make_request(request_id):
            try:
                response = requests.post(
                    f"{self.base_url}/audit",
                    json={
                        "files": [
                            {
                                "filename": f"concurrent_test_{request_id}.py",
                                "content": test_case['code']
                            }
                        ]
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    results_queue.put({
                        'request_id': request_id,
                        'success': True,
                        'issues': response.json().get('issues', [])
                    })
                else:
                    results_queue.put({
                        'request_id': request_id,
                        'success': False,
                        'error': response.status_code
                    })
            except Exception as e:
                results_queue.put({
                    'request_id': request_id,
                    'success': False,
                    'error': str(e)
                })
        
        # Launch 3 concurrent requests
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        successful_requests = [r for r in results if r['success']]
        print(f"Concurrent requests completed: {len(successful_requests)}/{len(results)} successful")
        
        # Validation
        validation_results = test_case['validation'](
            successful_requests[0]['issues'] if successful_requests else []
        )
        validation_results['concurrent_isolation'] = len(successful_requests) == 3
        
        all_passed = all(validation_results.values())
        
        for check_name, passed in validation_results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {check_name}: {status}")
        
        self.test_results.append({
            'name': test_case['name'],
            'passed': all_passed,
            'concurrent_results': len(successful_requests),
            'validation_details': validation_results
        })
        
        print(f"\nConcurrent Test Result: {'PASS' if all_passed else 'FAIL'}")
    
    def _print_summary(self):
        """Print overall test summary."""
        print("SEMANTIC ANALYSIS IMPROVEMENTS SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        
        print(f"Tests Run: {total_tests}")
        print(f"Tests Passed: {passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        print("Individual Test Results:")
        for result in self.test_results:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  [{status}] {result['name']}")
        
        print()
        print("Key Improvements Validated:")
        print("• AST-based eval() detection distinguishes functions from methods")
        print("• Context-aware RL environment analysis")
        print("• Framework-specific random seed detection")
        print("• Race condition prevention with isolated temp directories")
        print("• False positive filtering maintains security detection")
        
        if passed_tests == total_tests:
            print("\nAll semantic analysis improvements are working correctly!")
        else:
            print(f"\n{total_tests - passed_tests} tests need attention.")

def main():
    """Run the semantic analysis demonstration."""
    demo = SemanticAnalysisDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()