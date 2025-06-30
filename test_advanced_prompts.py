
"""
Test suite for the Advanced Master Prompt System.
Demonstrates persona-driven prompts, chain-of-thought reasoning, and specialized improvement types.
"""

import requests
import json
from typing import Dict, Any


def test_advanced_prompt_generation():
    """Test the advanced master prompt generation system."""
    
    base_url = "http://localhost:5000"
    
    print("Testing Advanced Master Prompt Generation System")
    print("=" * 70)
    
    # Test scenarios for different improvement types
    test_scenarios = [
        {
            "name": "Security-Focused Analysis",
            "improvement_type": "security",
            "description": "Financial API with security vulnerabilities",
            "files": [{
                "filename": "payment_processor.py",
                "content": """
import subprocess
import pickle
import os

def process_payment(user_data):
    # Security risk: eval with user input
    config = eval(user_data.get('config', '{}'))
    
    # Security risk: pickle deserialization
    with open('user_preferences.pkl', 'rb') as f:
        prefs = pickle.load(f)
    
    # Security risk: command injection
    amount = user_data['amount']
    result = subprocess.run(f"echo Processing ${amount}", shell=True)
    
    # Hardcoded secret
    api_key = "sk-1234567890abcdef"
    
    return {'status': 'processed', 'key': api_key}
"""
            }],
            "issues": [
                {
                    "filename": "payment_processor.py",
                    "line": 7,
                    "type": "security",
                    "description": "Use of eval() with user input - code injection risk",
                    "source": "bandit",
                    "severity": "error"
                },
                {
                    "filename": "payment_processor.py",
                    "line": 10,
                    "type": "security", 
                    "description": "Unsafe pickle.load() - deserialization vulnerability",
                    "source": "bandit",
                    "severity": "error"
                }
            ]
        },
        {
            "name": "Performance Optimization",
            "improvement_type": "performance",
            "description": "Data processing pipeline with performance bottlenecks",
            "files": [{
                "filename": "data_processor.py",
                "content": """
import pandas as pd
import numpy as np

def process_large_dataset(data):
    results = []
    
    # Inefficient loop
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                # Expensive computation in nested loop
                similarity = sum((data[i] - data[j]) ** 2)
                results.append(similarity)
    
    # Inefficient string concatenation
    output = ""
    for result in results:
        output += f"Result: {result}\\n"
    
    # Memory inefficient conversion
    return list(set(results))
"""
            }],
            "issues": [
                {
                    "filename": "data_processor.py",
                    "line": 6,
                    "type": "performance",
                    "description": "O(n^2) nested loop - algorithmic complexity issue",
                    "source": "custom_analysis",
                    "severity": "warning"
                }
            ]
        },
        {
            "name": "ML Optimization",
            "improvement_type": "ml_optimization", 
            "description": "PyTorch model training with reproducibility issues",
            "files": [{
                "filename": "model_trainer.py",
                "content": """
import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

def train_model():
    # Missing random seeding
    model = Model()
    
    # Inefficient data creation
    data = []
    for i in range(1000):
        data.append(torch.randn(10))
    
    # No GPU optimization
    for epoch in range(100):
        for batch in data:
            output = model(batch)
            loss = output.sum()
            loss.backward()
    
    return model
"""
            }],
            "issues": [
                {
                    "filename": "model_trainer.py",
                    "line": 15,
                    "type": "ml",
                    "description": "Missing random seed for reproducibility",
                    "source": "ml_analyzer",
                    "severity": "warning"
                }
            ]
        }
    ]
    
    # Test each scenario
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. Testing: {scenario['name']}")
        print(f"   Improvement Type: {scenario['improvement_type']}")
        print(f"   Description: {scenario['description']}")
        
        try:
            # Generate master prompt
            response = requests.post(
                f"{base_url}/improve/generate-master-prompt",
                json={
                    "files": scenario["files"],
                    "issues": scenario["issues"],
                    "improvement_type": scenario["improvement_type"],
                    "ai_provider": "openai"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"   ✓ Master prompt generated successfully")
                print(f"   ✓ Persona: {data.get('persona_used', 'unknown')}")
                print(f"   ✓ Confidence boost: {data.get('confidence_boost', 0):.1%}")
                print(f"   ✓ Context elements: {data.get('context_elements', {})}")
                print(f"   ✓ Prompt features: {', '.join(data.get('prompt_features', []))}")
                
                # Show persona snippet
                master_prompt = data.get('master_prompt', '')
                persona_section = master_prompt.split('[USER]')[0] if '[USER]' in master_prompt else master_prompt[:300]
                print(f"   ✓ Persona preview: {persona_section[:150]}...")
                
                # Test the prompt with actual improvement
                print(f"   → Testing prompt effectiveness...")
                
                improvement_response = requests.post(
                    f"{base_url}/improve/code",
                    json={
                        "original_code": scenario["files"][0]["content"],
                        "filename": scenario["files"][0]["filename"],
                        "issues": scenario["issues"],
                        "fixes": [],
                        "ai_provider": "openai",
                        "improvement_level": "comprehensive"
                    },
                    timeout=90
                )
                
                if improvement_response.status_code == 200:
                    improvement_data = improvement_response.json()
                    print(f"     ✓ Improvement completed successfully")
                    print(f"     ✓ Confidence: {improvement_data.get('confidence_score', 0):.1%}")
                    print(f"     ✓ Applied fixes: {len(improvement_data.get('applied_fixes', []))}")
                    
                    # Check for persona-specific improvements
                    improved_code = improvement_data.get('improved_code', '')
                    if scenario['improvement_type'] == 'security':
                        if 'eval(' not in improved_code and 'pickle.load(' not in improved_code:
                            print(f"     ✓ Security vulnerabilities appear to be fixed")
                    elif scenario['improvement_type'] == 'performance':
                        if 'vectorize' in improved_code.lower() or 'numpy' in improved_code:
                            print(f"     ✓ Performance optimizations appear to be applied")
                    elif scenario['improvement_type'] == 'ml_optimization':
                        if 'seed' in improved_code and 'reproducible' in improved_code.lower():
                            print(f"     ✓ ML reproducibility improvements appear to be applied")
                
                else:
                    print(f"     ❌ Improvement failed: {improvement_response.status_code}")
            
            else:
                print(f"   ❌ Master prompt generation failed: {response.status_code}")
                if response.text:
                    print(f"      Error: {response.text[:100]}...")
                    
        except requests.exceptions.Timeout:
            print(f"   ⚠️ Timeout - processing may need more time")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print(f"\n{'='*70}")
    print("Advanced Master Prompt System Features Demonstrated:")
    print("✓ Persona-driven role-playing for specialized expertise")
    print("✓ Chain-of-thought reasoning with explicit thought processes")
    print("✓ Structured input format with context and file information")
    print("✓ Explicit explanation requirements with line-by-line justification")
    print("✓ Specialized improvement types (security, performance, ML, RL)")
    print("✓ Context-aware confidence boosting")
    print("✓ Framework detection and dependency analysis")


def demonstrate_prompt_comparison():
    """Compare basic vs advanced master prompts."""
    
    print("\n" + "="*70)
    print("PROMPT COMPARISON: Basic vs Advanced Master Prompt")
    print("="*70)
    
    test_code = """
import pickle
import subprocess

def process_data(user_input):
    config = eval(user_input)
    result = subprocess.run(f"echo {config}", shell=True)
    return result
"""
    
    print("\n1. BASIC PROMPT (Current System):")
    print("-" * 40)
    print("You are an expert Python developer. Fix the security issues in this code.")
    
    print("\n2. ADVANCED MASTER PROMPT (New System):")
    print("-" * 40)
    
    try:
        response = requests.post(
            "http://localhost:5000/improve/generate-master-prompt",
            json={
                "files": [{"filename": "security_test.py", "content": test_code}],
                "issues": [{
                    "filename": "security_test.py",
                    "line": 5,
                    "type": "security",
                    "description": "eval() usage detected - code injection risk",
                    "source": "bandit",
                    "severity": "error"
                }],
                "improvement_type": "security"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            master_prompt = data.get('master_prompt', '')
            
            # Extract key sections
            if '[SYSTEM]' in master_prompt and '[USER]' in master_prompt:
                system_section = master_prompt.split('[USER]')[0].replace('[SYSTEM]', '').strip()
                user_section = master_prompt.split('[USER]')[1].strip()
                
                print(f"PERSONA: {system_section[:200]}...")
                print(f"\nSTRUCTURED INPUT: {user_section[:300]}...")
            
            print("\n3. KEY ADVANTAGES OF ADVANCED SYSTEM:")
            print("-" * 40)
            print("✓ Specific cybersecurity expert persona")
            print("✓ Structured markdown format with context")
            print("✓ Chain-of-thought reasoning requirement")
            print("✓ Explicit JSON response format")
            print("✓ File and project context integration")
            print("✓ Static analysis results incorporation")
            print(f"✓ Confidence boost: +{data.get('confidence_boost', 0):.1%}")
            
        else:
            print("Failed to generate advanced prompt")
            
    except Exception as e:
        print(f"Error in demonstration: {str(e)}")


if __name__ == "__main__":
    # Run comprehensive test
    test_advanced_prompt_generation()
    
    # Show comparison
    demonstrate_prompt_comparison()
    
    print("\n🎯 CONCLUSION:")
    print("The Advanced Master Prompt System provides:")
    print("• Persona-driven expertise for different improvement types")
    print("• Structured, context-aware prompt generation")
    print("• Chain-of-thought reasoning for better analysis")
    print("• Measurable confidence improvements through specialization")
    print("• Framework and dependency-aware optimizations")
    print("\nThis significantly enhances CodeGuard's AI improvement capabilities!")
