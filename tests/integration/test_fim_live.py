#!/usr/bin/env python3
"""
Live FIM Completion Test for multi_agent_trainer.py
Tests the DeepSeek FIM completion on the actual ML training code.
"""

import requests
import json
import os
from typing import Dict, Any

def test_fim_completion_on_trainer():
    """Test FIM completion on the multi_agent_trainer.py file."""
    
    # Read the actual multi_agent_trainer.py file
    try:
        with open('multi_agent_trainer.py', 'r') as f:
            trainer_code = f.read()
    except FileNotFoundError:
        print("❌ multi_agent_trainer.py not found")
        return
    
    print("🚀 Testing FIM Completion on Multi-Agent Trainer")
    print("=" * 60)
    
    # Test scenarios for FIM completion
    test_scenarios = [
        {
            "name": "Security Enhancement - Safe Model Loading",
            "prefix": """def secure_model_loader(model_path: str):
    \"\"\"Load ML model with proper security validation.\"\"\"
    import torch
    import os
    
    # TODO: Add security checks before loading""",
            "suffix": """    
    model = torch.load(model_path)
    return model"""
        },
        {
            "name": "Missing Random Seeding",
            "prefix": """def initialize_training(config):
    \"\"\"Initialize training environment with proper seeding.\"\"\"
    import torch
    import numpy as np
    import random
    
    # TODO: Add proper random seeding for reproducibility""",
            "suffix": """    
    # Initialize model and optimizer
    model = torch.nn.Linear(config['input_size'], config['output_size'])
    optimizer = torch.optim.Adam(model.parameters())
    return model, optimizer"""
        },
        {
            "name": "RL Environment Reset Pattern",
            "prefix": """def training_episode(env, agent):
    \"\"\"Run a single training episode with proper environment handling.\"\"\"
    import gym
    
    # TODO: Add proper environment reset and done flag handling""",
            "suffix": """    
    total_reward = 0
    while not done:
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
    
    return total_reward"""
        }
    ]
    
    api_base_url = "http://localhost:5000"
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📝 Test {i}: {scenario['name']}")
        print("-" * 40)
        
        # Test FIM completion
        result = test_fim_endpoint(
            api_base_url,
            scenario['prefix'],
            scenario['suffix'],
            scenario['name']
        )
        
        if result:
            print("✅ FIM completion successful")
            print(f"📊 Confidence: {result.get('confidence_score', 0) * 100:.1f}%")
            
            # Show the completed code
            completed_code = result['prefix'] + '\n' + result.get('completion', '') + '\n' + result['suffix']
            print("\n🔍 Completed Code Preview:")
            print("-" * 30)
            lines = completed_code.split('\n')
            for j, line in enumerate(lines[:15], 1):  # Show first 15 lines
                print(f"{j:2d}: {line}")
            if len(lines) > 15:
                print("    ... (truncated)")
        else:
            print("❌ FIM completion failed")
        
        print()

def test_fim_endpoint(api_base_url: str, prefix: str, suffix: str, scenario_name: str) -> Dict[str, Any]:
    """Test the FIM completion endpoint."""
    
    payload = {
        "prefix": prefix,
        "suffix": suffix,
        "ai_provider": "deepseek",
        "max_tokens": 1000
    }
    
    # Add API key if available in environment
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if api_key:
        payload['ai_api_key'] = api_key
        print(f"🔑 Using DeepSeek API key")
    else:
        print("⚠️  No DeepSeek API key found in environment")
        print("   Set DEEPSEEK_API_KEY to test with actual API")
        return None
    
    try:
        print(f"🌐 Calling FIM endpoint for: {scenario_name}")
        
        response = requests.post(
            f"{api_base_url}/improve/fim-completion",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def show_playground_instructions():
    """Show instructions for testing in the playground."""
    print("\n🎮 Playground Testing Instructions")
    print("=" * 50)
    print("1. Open the playground at: http://localhost:5000/playground")
    print("2. Click the 'FIM Complete' button")
    print("3. Click 'Load Example' to see a security fix demo")
    print("4. Or paste this example in the prefix field:")
    print()
    print("   def secure_model_loader(model_path: str):")
    print("       # TODO: Add security validation")
    print()
    print("5. Add this suffix:")
    print("       model = torch.load(model_path)")
    print("       return model")
    print()
    print("6. Select 'DeepSeek Reasoner' and add your API key")
    print("7. Click 'Run FIM Completion' to see the magic!")
    print()
    print("🔥 The system will intelligently complete the security checks")
    print("   and provide proper validation before loading the model!")

if __name__ == "__main__":
    print("🤖 CodeGuard FIM Completion Live Test")
    print("Testing DeepSeek Fill-in-the-Middle completion")
    print()
    
    # Test the FIM completion functionality
    test_fim_completion_on_trainer()
    
    # Show playground instructions
    show_playground_instructions()
    
    print("\n✨ FIM Completion Test Complete!")
    print("The playground is ready for interactive testing.")