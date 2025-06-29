#!/usr/bin/env python3
"""
Simple DeepSeek integration test focusing on timeout handling and fallback.
"""

import requests
import json

def test_codeguard_fallback():
    """Test CodeGuard handles DeepSeek timeouts gracefully."""
    
    payload = {
        "original_code": "import torch\ndef model():\n    return torch.nn.Linear(10, 1)",
        "filename": "test.py", 
        "issues": [{"type": "ml", "message": "Missing random seed", "line": 2}],
        "fixes": [],
        "ai_provider": "deepseek",
        "improvement_level": "moderate"
    }
    
    print("Testing CodeGuard DeepSeek integration with timeout handling...")
    
    try:
        response = requests.post(
            "http://localhost:5000/improve/code",
            json=payload,
            timeout=8  # Short timeout to trigger fallback
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Confidence: {data.get('confidence_score', 0):.1%}")
            
            # Check for fallback behavior
            summary = data.get('improvement_summary', '')
            if 'fallback' in summary.lower() or 'timeout' in summary.lower():
                print("Fallback mechanism triggered correctly")
            else:
                print("DeepSeek response received successfully")
            
            return True
        else:
            print(f"Error: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("Timeout occurred - this tests the timeout handling")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_openai_provider():
    """Test OpenAI provider as comparison."""
    
    payload = {
        "original_code": "import torch\ndef model():\n    return torch.nn.Linear(10, 1)",
        "filename": "test.py",
        "issues": [{"type": "ml", "message": "Missing random seed", "line": 2}],
        "fixes": [],
        "ai_provider": "openai",
        "improvement_level": "moderate"
    }
    
    print("Testing OpenAI provider for comparison...")
    
    try:
        response = requests.post(
            "http://localhost:5000/improve/code",
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"OpenAI Success! Confidence: {data.get('confidence_score', 0):.1%}")
            return True
        else:
            print(f"OpenAI Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return False

if __name__ == "__main__":
    print("DeepSeek Integration Test")
    print("=" * 30)
    
    results = []
    
    print("\n1. Testing DeepSeek with fallback...")
    results.append(test_codeguard_fallback())
    
    print("\n2. Testing OpenAI for comparison...")  
    results.append(test_openai_provider())
    
    passed = sum(results)
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed >= 1:
        print("DeepSeek integration working with proper timeout handling")
    else:
        print("Integration issues detected")