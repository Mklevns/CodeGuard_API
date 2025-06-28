"""
Test demonstration of ChatGPT integration for CodeGuard API.
Shows AI-powered code improvement functionality.
"""

import requests
import json

def test_improve_code_endpoint():
    """Test the /improve/code endpoint with sample ML code."""
    
    # Sample problematic ML code
    sample_code = '''
import torch
import numpy as np
import pickle

def train_model(data):
    # Missing random seeding
    model = torch.nn.Linear(10, 1)
    
    # Hardcoded path
    with open("/tmp/model.pkl", "wb") as f:
        pickle.dump(model, f)  # Security issue
    
    # No error handling
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(100):
        pred = model(data)
        l = loss(pred, torch.randn(32, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        print(f"Loss: {l}")  # Should use logging
    
    return model
'''
    
    # Sample issues and fixes (would come from CodeGuard audit)
    sample_issues = [
        {
            "filename": "train.py",
            "line": 6,
            "type": "security",
            "description": "Use of pickle.dump() is a security risk",
            "source": "custom_rules",
            "severity": "error"
        },
        {
            "filename": "train.py", 
            "line": 5,
            "type": "reproducibility",
            "description": "Missing random seeding for reproducible results",
            "source": "ml_rules",
            "severity": "warning"
        },
        {
            "filename": "train.py",
            "line": 9,
            "type": "portability",
            "description": "Hardcoded file path reduces portability",
            "source": "custom_rules",
            "severity": "warning"
        }
    ]
    
    sample_fixes = [
        {
            "filename": "train.py",
            "line": 6,
            "suggestion": "Replace pickle with torch.save() for model serialization",
            "auto_fixable": True,
            "replacement_code": "        torch.save(model.state_dict(), model_path)"
        },
        {
            "filename": "train.py",
            "line": 5,
            "suggestion": "Add torch.manual_seed() for reproducibility",
            "auto_fixable": True,
            "replacement_code": "    torch.manual_seed(42)  # For reproducible results"
        }
    ]
    
    # Test request
    test_request = {
        "code": sample_code,
        "filename": "train.py",
        "issues": sample_issues,
        "fixes": sample_fixes,
        "improvement_level": "moderate"
    }
    
    try:
        # Test locally first
        url = "http://localhost:5000/improve/code"
        response = requests.post(url, json=test_request, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ ChatGPT Code Improvement Test Successful!")
            print(f"Applied {result['fixes_applied_count']} fixes")
            print(f"Confidence Score: {result['confidence_score']}")
            print(f"Summary: {result['improvement_summary']}")
            
            if result['warnings']:
                print(f"Warnings: {', '.join(result['warnings'])}")
            
            return True
        else:
            print(f"❌ Test failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

def test_audit_and_improve_endpoint():
    """Test the combined /audit-and-improve endpoint."""
    
    sample_files = [
        {
            "filename": "pytorch_training.py",
            "content": '''
import torch
import numpy as np

def train_neural_network():
    # Missing seeding
    data = torch.randn(100, 10)
    target = torch.randn(100, 1)
    
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1)
    )
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(10):
        output = model(data)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return model
'''
        }
    ]
    
    test_request = {
        "files": sample_files,
        "options": {
            "level": "comprehensive",
            "framework": "pytorch",
            "target": "gpu"
        }
    }
    
    try:
        url = "http://localhost:5000/audit-and-improve"
        response = requests.post(url, json=test_request, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Combined Audit & Improve Test Successful!")
            print(f"Session ID: {result['combined_summary']['session_id']}")
            print(f"Issues Found: {result['combined_summary']['total_issues_found']}")
            print(f"CodeGuard Fixes: {result['combined_summary']['codeguard_fixes']}")
            print(f"AI Fixes Applied: {result['combined_summary']['ai_fixes_applied']}")
            print(f"Framework Detected: {result['combined_summary']['framework_detected']}")
            
            return True
        else:
            print(f"❌ Combined test failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Testing ChatGPT Integration for CodeGuard API")
    print("=" * 50)
    
    print("\n1. Testing Code Improvement Endpoint...")
    test1_success = test_improve_code_endpoint()
    
    print("\n2. Testing Combined Audit & Improve Endpoint...")
    test2_success = test_audit_and_improve_endpoint()
    
    print("\n" + "=" * 50)
    if test1_success and test2_success:
        print("🎉 All ChatGPT Integration Tests Passed!")
        print("\nFeatures Available:")
        print("• AI-powered code improvements")
        print("• Automatic fix implementation")
        print("• Security issue resolution")
        print("• ML/RL best practice enforcement")
        print("• Combined audit + improvement workflow")
    else:
        print("⚠️  Some tests failed. Check OpenAI API key configuration.")
        print("Note: Basic auto-fixes still work without OpenAI integration.")