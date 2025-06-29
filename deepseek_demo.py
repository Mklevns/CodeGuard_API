#!/usr/bin/env python3
"""
DeepSeek R1 Integration Demo for CodeGuard API.
Demonstrates the new multi-LLM system with DeepSeek R1 support.
"""

import json
import requests
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TestCodeFile:
    filename: str
    content: str

def test_deepseek_integration():
    """Test DeepSeek R1 integration in the multi-LLM system."""
    
    # Test code with security and ML issues
    test_code = '''import torch
import pickle
import os

def unsafe_ml_model():
    # Missing random seed
    model = torch.nn.Linear(10, 1)
    
    # Security vulnerability
    learning_rate = eval("0.01")
    
    # Unsafe data loading
    with open("model.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Missing error handling
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, optimizer, loss
'''

    base_url = "https://codeguard.replit.app"
    
    print("Testing DeepSeek R1 Integration in CodeGuard")
    print("=" * 50)
    
    # Test 1: Standard audit
    print("1. Running standard audit...")
    audit_response = requests.post(f"{base_url}/audit", 
        json={"files": [{"filename": "test_model.py", "content": test_code}]},
        timeout=30
    )
    
    if audit_response.status_code == 200:
        audit_data = audit_response.json()
        print(f"   ✓ Found {len(audit_data['issues'])} issues")
        
        # Test 2: OpenAI improvement (baseline)
        print("2. Testing OpenAI improvement...")
        openai_response = requests.post(f"{base_url}/improve/code",
            json={
                "original_code": test_code,
                "filename": "test_model.py",
                "issues": audit_data['issues'][:3],
                "fixes": audit_data['fixes'][:3],
                "ai_provider": "openai",
                "improvement_level": "moderate"
            },
            timeout=60
        )
        
        if openai_response.status_code == 200:
            openai_data = openai_response.json()
            print(f"   ✓ OpenAI improvement completed")
            print(f"   ✓ Confidence: {openai_data.get('confidence_score', 0):.1%}")
        
        # Test 3: DeepSeek R1 improvement (new feature)
        print("3. Testing DeepSeek R1 improvement...")
        deepseek_response = requests.post(f"{base_url}/improve/code",
            json={
                "original_code": test_code,
                "filename": "test_model.py",
                "issues": audit_data['issues'][:3],
                "fixes": audit_data['fixes'][:3],
                "ai_provider": "deepseek",
                "improvement_level": "moderate"
            },
            timeout=60
        )
        
        if deepseek_response.status_code == 200:
            deepseek_data = deepseek_response.json()
            print(f"   ✓ DeepSeek R1 improvement completed")
            print(f"   ✓ Confidence: {deepseek_data.get('confidence_score', 0):.1%}")
            print(f"   ✓ Applied fixes: {len(deepseek_data.get('applied_fixes', []))}")
            
            # Show comparison
            print("\n📊 Provider Comparison:")
            print(f"OpenAI confidence: {openai_data.get('confidence_score', 0):.1%}")
            print(f"DeepSeek confidence: {deepseek_data.get('confidence_score', 0):.1%}")
            
            print("\n✅ DeepSeek R1 integration successfully added!")
            print("   Available providers: OpenAI, DeepSeek R1, Gemini, Claude")
            return True
        else:
            print(f"   ❌ DeepSeek improvement failed: {deepseek_response.status_code}")
            print(f"   Response: {deepseek_response.text[:200]}...")
    else:
        print(f"   ❌ Audit failed: {audit_response.status_code}")
    
    return False

if __name__ == "__main__":
    success = test_deepseek_integration()
    if success:
        print("\n🎉 DeepSeek R1 API successfully integrated into CodeGuard!")
    else:
        print("\n⚠️ Integration test incomplete - check API configuration")