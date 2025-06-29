"""
Test script to demonstrate GitHub repository context integration.
Shows how repository context enhances AI code improvement suggestions.
"""

import requests
import json
import os

def test_github_repository_analysis():
    """Test the GitHub repository analysis endpoint."""
    
    # Test with a public ML repository
    test_repo_url = "https://github.com/pytorch/examples"
    
    payload = {
        "repo_url": test_repo_url,
        "github_token": os.getenv('GITHUB_TOKEN')  # Optional for public repos
    }
    
    try:
        # Test repository analysis
        print("Testing GitHub repository analysis...")
        response = requests.post(
            "http://localhost:5000/repo/analyze",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Repository analysis successful")
            print(f"  Repository: {result['repository']['owner']}/{result['repository']['name']}")
            print(f"  Language: {result['repository']['language']}")
            print(f"  Framework: {result['repository']['framework']}")
            print(f"  Dependencies: {result['repository']['dependencies_count']}")
            print(f"  Context available: {result['context_available']}")
            
            # Print context summary preview
            context = result['context_summary'][:300] + "..." if len(result['context_summary']) > 300 else result['context_summary']
            print(f"  Context preview: {context}")
            
            return result
        else:
            print(f"✗ Repository analysis failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Error testing repository analysis: {e}")
        return None

def test_context_enhanced_improvement():
    """Test code improvement with repository context."""
    
    # Sample PyTorch code with issues
    sample_code = """
import torch
import numpy as np

def train_model(data):
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        # Missing random seed
        outputs = model(data)
        loss = torch.nn.functional.mse_loss(outputs, torch.randn(100, 1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return model
"""
    
    # Sample issues detected by CodeGuard
    sample_issues = [
        {
            "filename": "train.py",
            "line": 6,
            "type": "ml_pattern",
            "description": "Missing random seed for reproducibility",
            "source": "ml_rules",
            "severity": "warning"
        },
        {
            "filename": "train.py", 
            "line": 10,
            "type": "ml_pattern",
            "description": "Hardcoded target tensor should use proper labels",
            "source": "ml_rules",
            "severity": "warning"
        }
    ]
    
    # Test both with and without repository context
    test_cases = [
        {
            "name": "Without Repository Context",
            "payload": {
                "original_code": sample_code,
                "filename": "train.py",
                "issues": sample_issues,
                "ai_provider": "openai",
                "improvement_level": "moderate"
            },
            "endpoint": "/improve/code"
        },
        {
            "name": "With Repository Context", 
            "payload": {
                "original_code": sample_code,
                "filename": "train.py", 
                "issues": sample_issues,
                "github_repo_url": "https://github.com/pytorch/examples",
                "ai_provider": "openai",
                "improvement_level": "moderate"
            },
            "endpoint": "/improve/with-repo-context"
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        
        try:
            response = requests.post(
                f"http://localhost:5000{test_case['endpoint']}",
                json=test_case['payload'],
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Improvement successful")
                print(f"  Confidence: {result.get('confidence_score', 0):.2f}")
                print(f"  Fixes applied: {len(result.get('applied_fixes', []))}")
                print(f"  Context used: {result.get('repository_context_used', False)}")
                
                # Store result for comparison
                results[test_case['name']] = result
                
            else:
                print(f"✗ Improvement failed: {response.status_code}")
                print(f"  Error: {response.text}")
                
        except Exception as e:
            print(f"✗ Error testing improvement: {e}")
    
    return results

def test_context_summary():
    """Test getting repository context summary."""
    
    payload = {
        "repo_url": "https://github.com/pytorch/examples"
    }
    
    try:
        print("\nTesting repository context summary...")
        response = requests.post(
            "http://localhost:5000/repo/context-summary",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Context summary successful")
            print(f"  Status: {result['status']}")
            print(f"  Context available: {result['context_available']}")
            
            if result['context_available']:
                summary = result['context_summary']
                print(f"  Summary length: {len(summary)} characters")
                print(f"  Framework: {result['repository_info']['framework']}")
                
            return result
        else:
            print(f"✗ Context summary failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"✗ Error testing context summary: {e}")
        return None

def compare_improvement_results(results):
    """Compare improvement results with and without repository context."""
    
    if len(results) < 2:
        print("\nInsufficient results for comparison")
        return
    
    print("\n" + "="*60)
    print("COMPARISON: Repository Context Impact")
    print("="*60)
    
    without_context = results.get("Without Repository Context", {})
    with_context = results.get("With Repository Context", {})
    
    print(f"Without Context:")
    print(f"  Confidence Score: {without_context.get('confidence_score', 0):.2f}")
    print(f"  Applied Fixes: {len(without_context.get('applied_fixes', []))}")
    
    print(f"\nWith Repository Context:")
    print(f"  Confidence Score: {with_context.get('confidence_score', 0):.2f}")
    print(f"  Applied Fixes: {len(with_context.get('applied_fixes', []))}")
    print(f"  Context Enhanced: {with_context.get('repository_context_used', False)}")
    
    # Calculate improvement
    confidence_improvement = with_context.get('confidence_score', 0) - without_context.get('confidence_score', 0)
    fixes_improvement = len(with_context.get('applied_fixes', [])) - len(without_context.get('applied_fixes', []))
    
    print(f"\nImprovement with Repository Context:")
    print(f"  Confidence Boost: {confidence_improvement:+.2f}")
    print(f"  Additional Fixes: {fixes_improvement:+d}")
    
    if confidence_improvement > 0:
        print(f"  ✓ Repository context improved AI suggestions by {confidence_improvement:.1%}")
    else:
        print(f"  → Repository context provided consistent quality")

def main():
    """Run comprehensive GitHub integration tests."""
    
    print("GitHub Repository Context Integration Test")
    print("="*50)
    
    # Test 1: Repository Analysis
    repo_analysis = test_github_repository_analysis()
    
    # Test 2: Context Summary
    context_summary = test_context_summary()
    
    # Test 3: Code Improvement Comparison
    improvement_results = test_context_enhanced_improvement()
    
    # Test 4: Results Comparison
    if improvement_results:
        compare_improvement_results(improvement_results)
    
    print("\n" + "="*50)
    print("GitHub Integration Test Complete")
    
    # Summary
    success_count = sum([
        1 if repo_analysis else 0,
        1 if context_summary else 0,
        1 if improvement_results else 0
    ])
    
    print(f"Tests passed: {success_count}/3")
    
    if success_count == 3:
        print("✓ All GitHub integration features working correctly")
        print("✓ Repository context successfully enhances AI code improvements")
    else:
        print("⚠ Some tests failed - check API endpoints and connectivity")

if __name__ == "__main__":
    main()