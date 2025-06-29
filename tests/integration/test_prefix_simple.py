#!/usr/bin/env python3
"""
Simple test of DeepSeek Chat Prefix Completion integration.
"""

import os
import json

def show_prefix_integration():
    """Show how Chat Prefix Completion enhances CodeGuard."""
    
    print("DeepSeek Chat Prefix Completion for CodeGuard")
    print("=" * 50)
    
    # Example problematic code
    code_sample = '''
import torch
import pickle

def unsafe_function(data_path):
    learning_rate = eval("0.001")  # Security issue
    
    with open(data_path, 'rb') as f:
        model = pickle.load(f)  # Security vulnerability
    
    return model
'''
    
    print("Sample problematic code:")
    print(code_sample)
    
    print("\nPrefix Completion Approaches:")
    print("-" * 30)
    
    # 1. JSON-structured improvement
    print("1. JSON-Structured Response:")
    print("   Prefix: '{'")
    print("   Stop: ['}']")
    print("   Result: Guaranteed JSON format with:")
    print("   - improved_code")
    print("   - applied_fixes")
    print("   - confidence_score")
    print("   - warnings")
    
    # Show example JSON response
    example_json = {
        "improved_code": "# Fixed security vulnerabilities\nimport torch\nimport json\n\ndef safe_function(data_path):\n    learning_rate = 0.001  # Direct assignment\n    \n    with open(data_path, 'r') as f:\n        model = json.load(f)  # Safe JSON loading\n    \n    return model",
        "applied_fixes": [
            {"type": "security", "description": "Replaced eval() with direct assignment"},
            {"type": "security", "description": "Replaced pickle.load() with json.load()"}
        ],
        "confidence_score": 0.9,
        "warnings": ["Validate input file format"]
    }
    
    print("   Example response:")
    print("   " + json.dumps(example_json, indent=2).replace('\n', '\n   '))
    
    # 2. Direct code completion
    print("\n2. Direct Code Completion:")
    print("   Prefix: '```python\\n'")
    print("   Stop: ['```']")
    print("   Result: Complete improved code implementation")
    
    # 3. Security-focused
    print("\n3. Security-Focused Explanation:")
    print("   Prefix: '# Security Fix:'")
    print("   Stop: ['\\n\\n']")
    print("   Result: Detailed security vulnerability explanations")
    
    print("\nIntegration with CodeGuard API:")
    print("✓ /audit endpoint → detect issues")
    print("✓ /improve/code endpoint → DeepSeek prefix completion")
    print("✓ Guaranteed structured responses")
    print("✓ Better parsing reliability")
    print("✓ Multiple output formats available")
    
    # Show API key status
    if os.getenv('DEEPSEEK_API_KEY'):
        print(f"\n✓ DeepSeek API key configured")
        print("  Ready for live prefix completion testing")
    else:
        print(f"\n⚠ DeepSeek API key not found")
        print("  Set DEEPSEEK_API_KEY to test live integration")
    
    print("\nBenefits for CodeGuard users:")
    print("• More reliable AI-powered improvements")
    print("• Structured JSON responses for parsing")
    print("• Direct code completion when needed")  
    print("• Better integration with ChatGPT Actions")
    print("• Reduced API response parsing errors")

if __name__ == "__main__":
    show_prefix_integration()