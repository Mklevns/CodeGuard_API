#!/usr/bin/env python3
"""
Quick test of AI-powered code improvements.
"""

import requests
import json

def test_ai_improvements():
    """Test the AI improvement system with a simple example."""
    
    # Simple test code with common issues
    test_code = '''
import torch
import pickle

def unsafe_function():
    # Security issue: using eval
    learning_rate = eval("0.001")
    
    # Missing random seed
    model = torch.nn.Linear(10, 1)
    
    # Safe model loading using torch.load instead of pickle
    with open("model.pth", "rb") as f:
        data = torch.load(f, map_location='cpu')
    
    return model, data
'''

    base_url = "https://codeguard.replit.app"
    
    print("Testing AI-powered code improvement...")
    
    # First, get analysis
    audit_request = {
        "files": [{"filename": "test.py", "content": test_code}],
        "options": {"level": "strict", "framework": "pytorch"}
    }
    
    try:
        response = requests.post(f"{base_url}/audit", json=audit_request, timeout=30)
        
        if response.status_code == 200:
            audit_data = response.json()
            print(f"Audit completed - found {len(audit_data['issues'])} issues")
            
            # Now test AI improvement
            improvement_request = {
                "original_code": test_code,
                "filename": "test.py",
                "issues": audit_data['issues'][:5],  # Top 5 issues
                "improvement_level": "moderate"
            }
            
            improve_response = requests.post(
                f"{base_url}/improve/code", 
                json=improvement_request, 
                timeout=45
            )
            
            if improve_response.status_code == 200:
                improvement_data = improve_response.json()
                print(f"AI improvement completed!")
                print(f"Confidence: {improvement_data['confidence_score']:.1%}")
                print(f"Fixes applied: {len(improvement_data['applied_fixes'])}")
                print(f"Summary: {improvement_data['improvement_summary']}")
                
                # Show improved code snippet
                improved_code = improvement_data['improved_code']
                print("\nImproved code preview:")
                for i, line in enumerate(improved_code.split('\n')[:10], 1):
                    if line.strip():
                        print(f"{i:2d}: {line}")
                
                return True
            else:
                print(f"AI improvement failed: {improve_response.status_code}")
                return False
        else:
            print(f"Audit failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_ai_improvements()
    if success:
        print("\n✅ AI-powered code improvement system is working!")
    else:
        print("\n❌ AI improvement system needs attention")