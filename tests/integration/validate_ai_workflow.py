#!/usr/bin/env python3
"""
Focused validation of AI-powered code improvement workflow.
Tests the complete pipeline from semantic analysis to ChatGPT improvements.
"""

import requests
import json

def validate_ai_workflow():
    """Validate the complete AI-powered improvement workflow."""
    
    # Test code with known issues
    test_code = '''import torch
import pickle

def unsafe_model():
    # Missing seed
    model = torch.nn.Linear(10, 1)
    # Security issue
    rate = eval("0.01")
    # Unsafe loading
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
    return model
'''

    base_url = "https://codeguard.replit.app"
    
    print("Validating AI-Powered Code Improvement Workflow")
    print("=" * 50)
    
    # Test 1: Standard audit with semantic analysis
    print("1. Testing semantic analysis audit...")
    audit_response = requests.post(f"{base_url}/audit", 
        json={"files": [{"filename": "test.py", "content": test_code}]},
        timeout=30
    )
    
    if audit_response.status_code == 200:
        audit_data = audit_response.json()
        print(f"   ✓ Found {len(audit_data['issues'])} issues")
        
        # Test 2: AI-powered improvement
        print("2. Testing ChatGPT code improvement...")
        improve_response = requests.post(f"{base_url}/improve/code",
            json={
                "original_code": test_code,
                "filename": "test.py", 
                "issues": audit_data['issues'][:5],
                "improvement_level": "moderate"
            },
            timeout=60
        )
        
        if improve_response.status_code == 200:
            improve_data = improve_response.json()
            print(f"   ✓ AI improvement completed")
            print(f"   ✓ Confidence: {improve_data['confidence_score']:.1%}")
            print(f"   ✓ Applied {len(improve_data['applied_fixes'])} fixes")
            
            # Test 3: Combined audit + improvement
            print("3. Testing combined workflow...")
            combined_response = requests.post(f"{base_url}/audit-and-improve",
                json={"files": [{"filename": "test.py", "content": test_code}]},
                timeout=90
            )
            
            if combined_response.status_code == 200:
                combined_data = combined_response.json()
                summary = combined_data['combined_summary']
                print(f"   ✓ Combined workflow completed")
                print(f"   ✓ Issues: {summary['total_issues_found']}")
                print(f"   ✓ AI fixes: {summary['ai_fixes_applied']}")
                print(f"   ✓ Framework: {summary['framework_detected']}")
                
                print("\n✅ All AI workflow components are operational!")
                return True
            else:
                print(f"   ❌ Combined workflow failed: {combined_response.status_code}")
        else:
            print(f"   ❌ AI improvement failed: {improve_response.status_code}")
    else:
        print(f"   ❌ Audit failed: {audit_response.status_code}")
    
    return False

if __name__ == "__main__":
    success = validate_ai_workflow()
    if success:
        print("\n🎉 AI-powered code improvement system fully validated!")
    else:
        print("\n⚠️ Some components need attention")