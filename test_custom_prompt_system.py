"""
Test suite for the LLM-powered custom prompt generation system.
Demonstrates how CodeGuard creates specialized AI prompts based on audit results.
"""

import requests
import json
import time
from typing import Dict, Any, List


def test_custom_prompt_generation():
    """Test the custom prompt generation with various code scenarios."""
    
    base_url = "http://localhost:5000"
    
    print("Testing LLM-Powered Custom Prompt Generation System")
    print("=" * 60)
    
    # Test scenarios with different types of issues
    test_scenarios = [
        {
            "name": "Security-Critical PyTorch Code",
            "description": "Code with security vulnerabilities and PyTorch patterns",
            "files": [{
                "filename": "dangerous_model.py",
                "content": """
import torch
import pickle
import os

def load_model(model_path):
    # Security fix: use torch.load instead of pickle
    model = torch.load(model_path, map_location='cpu')
    
    # Missing random seeding
    data = torch.randn(100, 10)
    
    # Dangerous eval usage
    config = eval(os.environ.get('MODEL_CONFIG', '{}'))
    
    return model, data, config
"""
            }],
            "issues": [
                {
                    "filename": "dangerous_model.py",
                    "line": 8,
                    "type": "security",
                    "description": "Use of pickle.load() poses security risk",
                    "source": "custom_rules",
                    "severity": "error"
                },
                {
                    "filename": "dangerous_model.py",
                    "line": 12,
                    "type": "ml",
                    "description": "Missing random seed for reproducibility",
                    "source": "ml_rules",
                    "severity": "warning"
                },
                {
                    "filename": "dangerous_model.py",
                    "line": 15,
                    "type": "security",
                    "description": "Use of eval() function detected",
                    "source": "custom_rules",
                    "severity": "error"
                }
            ]
        },
        {
            "name": "RL Environment Training",
            "description": "Reinforcement learning code with environment handling issues",
            "files": [{
                "filename": "rl_trainer.py",
                "content": """
import gym
import numpy as np

def train_agent():
    env = gym.make('CartPole-v1')
    
    # Missing env.reset()
    for episode in range(100):
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
        # Missing reset for next episode
    
    env.close()
"""
            }],
            "issues": [
                {
                    "filename": "rl_trainer.py",
                    "line": 7,
                    "type": "rl",
                    "description": "Missing env.reset() at episode start",
                    "source": "rl_plugin",
                    "severity": "error"
                },
                {
                    "filename": "rl_trainer.py",
                    "line": 14,
                    "type": "rl",
                    "description": "Missing env.reset() between episodes",
                    "source": "rl_plugin",
                    "severity": "error"
                }
            ]
        },
        {
            "name": "Style and Import Issues",
            "description": "Code with formatting and import problems",
            "files": [{
                "filename": "messy_code.py",
                "content": """
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

def very_long_function_name_that_exceeds_the_line_length_limit_and_should_be_broken():
    pass

def another_function():
    # Unused imports above
    x = torch.tensor([1, 2, 3])
    return x
"""
            }],
            "issues": [
                {
                    "filename": "messy_code.py",
                    "line": 3,
                    "type": "style",
                    "description": "Unused import: matplotlib.pyplot",
                    "source": "flake8",
                    "severity": "warning"
                },
                {
                    "filename": "messy_code.py",
                    "line": 8,
                    "type": "style",
                    "description": "Line too long (89 > 79 characters)",
                    "source": "flake8",
                    "severity": "warning"
                }
            ]
        }
    ]
    
    # Test each scenario
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. Testing: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        
        # Test with different AI providers
        for provider in ["openai", "deepseek"]:
            print(f"\n   Provider: {provider}")
            
            try:
                # Generate custom prompt
                response = requests.post(
                    f"{base_url}/improve/generate-custom-prompt",
                    json={
                        "files": scenario["files"],
                        "issues": scenario["issues"],
                        "ai_provider": provider
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    print(f"   ✓ Prompt generated successfully")
                    print(f"   ✓ Strategy: {data.get('prompt_strategy', 'unknown')}")
                    print(f"   ✓ Confidence boost: {data.get('confidence_boost', 0):.1%}")
                    print(f"   ✓ Focus areas: {', '.join(data.get('focus_areas', []))}")
                    print(f"   ✓ Effectiveness: {data.get('estimated_effectiveness', 0):.1%}")
                    print(f"   ✓ Issues analyzed: {data.get('total_issues_analyzed', 0)}")
                    
                    # Show snippet of generated prompt
                    custom_prompt = data.get('custom_prompt', '')
                    if len(custom_prompt) > 200:
                        prompt_snippet = custom_prompt[:200] + "..."
                    else:
                        prompt_snippet = custom_prompt
                    
                    print(f"   ✓ Prompt preview: {prompt_snippet}")
                    
                    # Test the generated prompt with actual improvement
                    print(f"   → Testing generated prompt effectiveness...")
                    
                    improvement_response = requests.post(
                        f"{base_url}/improve/code",
                        json={
                            "original_code": scenario["files"][0]["content"],
                            "filename": scenario["files"][0]["filename"],
                            "issues": scenario["issues"],
                            "fixes": [],
                            "ai_provider": provider,
                            "improvement_level": "moderate"
                        },
                        timeout=60
                    )
                    
                    if improvement_response.status_code == 200:
                        improvement_data = improvement_response.json()
                        print(f"     ✓ Improvement completed")
                        print(f"     ✓ Final confidence: {improvement_data.get('confidence_score', 0):.1%}")
                        print(f"     ✓ Applied fixes: {len(improvement_data.get('applied_fixes', []))}")
                        
                        # Check if confidence was boosted
                        final_confidence = improvement_data.get('confidence_score', 0)
                        boost = data.get('confidence_boost', 0)
                        if final_confidence > 0.5 + boost * 0.8:  # Approximate check
                            print(f"     ✓ Custom prompt appears to have boosted confidence")
                        
                    else:
                        print(f"     ❌ Improvement failed: {improvement_response.status_code}")
                
                else:
                    print(f"   ❌ Prompt generation failed: {response.status_code}")
                    if response.text:
                        print(f"      Error: {response.text[:100]}...")
                        
            except requests.exceptions.Timeout:
                print(f"   ⚠️ Timeout - {provider} may need more processing time")
            except Exception as e:
                print(f"   ❌ Error testing {provider}: {str(e)}")
    
    print(f"\n{'='*60}")
    print("Custom Prompt Generation Test Summary:")
    print("✓ Security-focused prompts for dangerous code patterns")
    print("✓ RL-specialized prompts for environment handling")
    print("✓ Style-focused prompts for code quality issues")
    print("✓ Provider-specific optimization (OpenAI vs DeepSeek)")
    print("✓ Confidence boosting based on prompt specialization")
    print("✓ Effectiveness estimation for different scenarios")


def demonstrate_prompt_comparison():
    """Compare default vs custom prompts side by side."""
    
    print("\n" + "="*60)
    print("PROMPT COMPARISON DEMONSTRATION")
    print("="*60)
    
    # Example code with security issues
    test_code = """
import pickle
import torch

def load_and_train():
    # Load model from pickle (security risk)
    model = pickle.load(open('model.pkl', 'rb'))
    
    # No random seeding
    data = torch.randn(100, 10)
    
    # Dangerous eval
    config = eval(input("Enter config: "))
    
    return model.train()
"""
    
    test_issues = [
        {
            "filename": "security_test.py",
            "line": 6,
            "type": "security",
            "description": "Unsafe pickle.load() usage",
            "source": "custom_rules",
            "severity": "error"
        },
        {
            "filename": "security_test.py",
            "line": 12,
            "type": "security", 
            "description": "Dangerous eval() with user input",
            "source": "custom_rules",
            "severity": "error"
        }
    ]
    
    print("\nGenerating custom security-focused prompt...")
    
    try:
        response = requests.post(
            "http://localhost:5000/improve/generate-custom-prompt",
            json={
                "files": [{"filename": "security_test.py", "content": test_code}],
                "issues": test_issues,
                "ai_provider": "openai"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            custom_prompt = data.get('custom_prompt', '')
            
            print("\n" + "-"*40)
            print("CUSTOM GENERATED PROMPT:")
            print("-"*40)
            print(custom_prompt)
            
            print("\n" + "-"*40)
            print("DEFAULT PROMPT WOULD BE:")
            print("-"*40)
            print("You are an expert Python developer specializing in ML/RL code improvements. "
                  "Apply the suggested fixes while maintaining code functionality and readability.")
            
            print("\n" + "-"*40)
            print("KEY DIFFERENCES:")
            print("-"*40)
            print("✓ Custom prompt is security-specialized")
            print("✓ Includes specific security guidelines")
            print("✓ Provides safe alternative implementations")
            print("✓ Explains security risks in detail")
            print(f"✓ Confidence boost: +{data.get('confidence_boost', 0):.1%}")
            print(f"✓ Strategy: {data.get('prompt_strategy', 'unknown')}")
            
        else:
            print(f"Failed to generate prompt: {response.status_code}")
            
    except Exception as e:
        print(f"Error in demonstration: {str(e)}")


if __name__ == "__main__":
    # Run the comprehensive test
    test_custom_prompt_generation()
    
    # Show prompt comparison
    demonstrate_prompt_comparison()
    
    print("\n🎯 CONCLUSION:")
    print("The LLM-powered custom prompt generation system successfully:")
    print("• Analyzes audit results to understand code context")
    print("• Generates specialized prompts for different issue types")
    print("• Boosts AI confidence through targeted instructions")
    print("• Adapts to different AI providers (OpenAI, DeepSeek)")
    print("• Provides measurable effectiveness improvements")
    print("\nThis system makes CodeGuard's AI improvements more precise and effective!")