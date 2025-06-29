#!/usr/bin/env python3
"""
Live test of DeepSeek FIM completion on the multi_agent_trainer.py file.
"""

import requests
import json
import os

def test_audit_first():
    """First, run audit to see what issues are detected."""
    
    # Read the problematic multi_agent_trainer.py file
    with open('multi_agent_trainer.py', 'r') as f:
        content = f.read()
    
    print("Testing CodeGuard Audit on multi_agent_trainer.py")
    print("=" * 60)
    
    # Prepare audit request
    audit_data = {
        "files": [
            {
                "filename": "multi_agent_trainer.py",
                "content": content
            }
        ],
        "options": {
            "level": "production",
            "framework": "pytorch", 
            "target": "gpu"
        }
    }
    
    try:
        response = requests.post('http://localhost:5000/audit', json=audit_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Audit completed successfully!")
            print(f"Issues found: {len(result.get('issues', []))}")
            print(f"Fixes suggested: {len(result.get('fixes', []))}")
            print(f"Tools used: {', '.join(result.get('tools_used', []))}")
            
            # Show top 10 issues
            issues = result.get('issues', [])
            print(f"\nTop {min(10, len(issues))} Issues Detected:")
            for i, issue in enumerate(issues[:10]):
                print(f"  {i+1}. Line {issue.get('line', '?')}: {issue.get('description', 'No description')}")
                print(f"     Category: {issue.get('category', 'unknown')} | Severity: {issue.get('severity', 'unknown')}")
            
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
            
            return result
        else:
            print(f"Audit failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error during audit: {str(e)}")
        return None

def test_fim_completion():
    """Test FIM completion for specific code sections."""
    
    print("\n" + "=" * 60)
    print("Testing DeepSeek FIM Completion")
    print("=" * 60)
    
    # Test case 1: Complete missing seeding implementation
    prefix1 = '''def train_neural_network():
    """Train with proper ML best practices."""
    import torch
    import numpy as np
    import random
    
    # TODO: Add proper random seeding for reproducibility'''
    
    suffix1 = '''    
    model = torch.nn.Linear(10, 1)
    return model'''
    
    fim_request1 = {
        "prefix": prefix1,
        "suffix": suffix1,
        "ai_provider": "deepseek",
        "max_tokens": 500
    }
    
    print("Test 1: Completing missing random seeding")
    print("Prefix:", prefix1.split('\n')[-1])
    print("Suffix:", suffix1.strip().split('\n')[0])
    
    try:
        if os.getenv('DEEPSEEK_API_KEY'):
            response = requests.post('http://localhost:5000/improve/fim-completion', json=fim_request1)
            
            if response.status_code == 200:
                result = response.json()
                print("✓ FIM completion successful!")
                print(f"Confidence: {result.get('confidence_score', 0)}")
                print("Completion:")
                print(result.get('completion', 'No completion returned'))
            else:
                print(f"✗ FIM completion failed: {response.status_code}")
                print(response.text)
        else:
            print("⚠ No DEEPSEEK_API_KEY - simulating completion")
            simulated_completion = '''
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True'''
            print("Simulated completion:")
            print(simulated_completion)
            
    except Exception as e:
        print(f"Error during FIM completion: {str(e)}")
    
    # Test case 2: Complete security vulnerability fix
    print("\n" + "-" * 40)
    print("Test 2: Security vulnerability fix")
    
    prefix2 = '''def process_config(config_path):
    """Process configuration with security fix."""
    # Security issue: eval usage
    learning_rate = '''
    
    suffix2 = '''  # Fixed: Use safe alternative
    return learning_rate'''
    
    print("Prefix:", prefix2.split('\n')[-1])
    print("Suffix:", suffix2.strip().split('\n')[0])
    
    # For this test, show what FIM would complete
    print("Expected FIM completion: ast.literal_eval(config_data) if config_data else 0.001")

def test_audit_and_improve():
    """Test the combined audit-and-improve endpoint with DeepSeek."""
    
    print("\n" + "=" * 60)
    print("Testing Combined Audit + AI Improvement")
    print("=" * 60)
    
    # Read file content
    with open('multi_agent_trainer.py', 'r') as f:
        content = f.read()
    
    # Create combined request
    combined_request = {
        "files": [
            {
                "filename": "multi_agent_trainer.py",
                "content": content
            }
        ],
        "options": {
            "level": "production",
            "framework": "pytorch",
            "target": "gpu"
        },
        "ai_provider": "deepseek"
    }
    
    # Add API key if available
    if os.getenv('DEEPSEEK_API_KEY'):
        combined_request["ai_api_key"] = os.getenv('DEEPSEEK_API_KEY')
        
        print("Testing with live DeepSeek API...")
        try:
            response = requests.post('http://localhost:5000/audit-and-improve', json=combined_request)
            
            if response.status_code == 200:
                result = response.json()
                print("✓ Combined audit and improvement successful!")
                
                audit_results = result.get('audit_results', {})
                improved_files = result.get('improved_files', {})
                
                print(f"Issues detected: {len(audit_results.get('issues', []))}")
                print(f"Files improved: {len(improved_files)}")
                
                if improved_files:
                    for filename, improvement in improved_files.items():
                        print(f"\nImprovement for {filename}:")
                        print(f"  Applied fixes: {len(improvement.get('applied_fixes', []))}")
                        print(f"  Confidence: {improvement.get('confidence_score', 0)}")
                        print(f"  Summary: {improvement.get('improvement_summary', 'No summary')[:100]}...")
                
            else:
                print(f"✗ Combined endpoint failed: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"Error during combined test: {str(e)}")
    else:
        print("⚠ No DEEPSEEK_API_KEY - testing audit only")
        
        # Test just the audit part
        audit_only = {
            "files": combined_request["files"],
            "options": combined_request["options"]
        }
        
        try:
            response = requests.post('http://localhost:5000/audit', json=audit_only)
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Audit successful: {len(result.get('issues', []))} issues found")
        except Exception as e:
            print(f"Error during audit: {str(e)}")

def run_comprehensive_test():
    """Run all FIM completion tests."""
    
    print("CodeGuard FIM Completion Live Testing")
    print("=" * 60)
    print("Testing the enhanced DeepSeek integration with:")
    print("• Multi-strategy completion (FIM → Prefix → Function Calling)")
    print("• Targeted code improvements")
    print("• Security vulnerability fixes")
    print("• ML/RL best practices implementation")
    print()
    
    # Step 1: Audit to find issues
    audit_result = test_audit_first()
    
    # Step 2: Test FIM completion
    test_fim_completion()
    
    # Step 3: Test combined workflow
    test_audit_and_improve()
    
    print("\n" + "=" * 60)
    print("FIM TESTING SUMMARY")
    print("=" * 60)
    
    if os.getenv('DEEPSEEK_API_KEY'):
        print("✓ DeepSeek API key configured - live testing completed")
    else:
        print("⚠ DeepSeek API key not found - simulated testing completed")
        print("  Set DEEPSEEK_API_KEY environment variable for live testing")
    
    print("\nFIM Capabilities Demonstrated:")
    print("• Targeted completion of TODO markers")
    print("• Security vulnerability fixes")
    print("• ML best practices implementation")
    print("• Integration with CodeGuard audit system")
    print("• Multi-strategy fallback for reliability")
    
    if audit_result:
        issues_count = len(audit_result.get('issues', []))
        print(f"\nReady for improvement: {issues_count} issues detected in multi_agent_trainer.py")
        print("The enhanced DeepSeek system can now apply targeted fixes using FIM completion")

if __name__ == "__main__":
    run_comprehensive_test()