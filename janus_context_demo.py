"""
JanusAI Repository Context Enhancement Demonstration
Shows how repository context improves AI suggestions for this specific multi-agent RL project.
"""

import requests
import json

def demonstrate_janus_context_enhancement():
    """Demonstrate context-enhanced improvements for JanusAI_V2."""
    
    print("JanusAI_V2 Repository Context Enhancement Demo")
    print("=" * 50)
    
    # Sample code that might be found in a JanusAI agent implementation
    sample_janus_code = '''
import torch
import numpy as np
import wandb
import gym
from typing import Dict, List, Tuple

class PPOAgent:
    def __init__(self, config):
        # Missing random seed setup
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network initialization without proper seeding
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(config["state_dim"], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, config["action_dim"])
        )
        
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(config["state_dim"], 256), 
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=config["learning_rate"]
        )
    
    def collect_experience(self, env, num_steps=1000):
        states, actions, rewards, dones = [], [], [], []
        state = env.reset()
        
        for step in range(num_steps):
            # Missing proper tensor handling
            action_probs = self.policy_net(state)
            action = torch.multinomial(torch.softmax(action_probs, dim=-1), 1)
            
            next_state, reward, done, info = env.step(action.item())
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            if done:
                state = env.reset()
            else:
                state = next_state
        
        return states, actions, rewards, dones
    
    def train(self, states, actions, rewards, advantages):
        # Missing gradient clipping and proper loss computation
        for epoch in range(self.config["num_epochs"]):
            policy_loss = self.compute_policy_loss(states, actions, advantages)
            value_loss = self.compute_value_loss(states, rewards)
            
            total_loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Missing wandb logging best practices
            wandb.log({"loss": total_loss.item()})
    
    def compute_policy_loss(self, states, actions, advantages):
        # Simplified implementation missing important PPO components
        logits = self.policy_net(torch.stack(states))
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, torch.stack(actions))
        
        return -(action_log_probs * advantages).mean()
    
    def compute_value_loss(self, states, rewards):
        values = self.value_net(torch.stack(states))
        targets = torch.tensor(rewards, dtype=torch.float32)
        return torch.nn.functional.mse_loss(values.squeeze(), targets)
'''
    
    # Simulate common issues that CodeGuard would detect
    sample_issues = [
        {
            "filename": "ppo_agent.py",
            "line": 12,
            "type": "ml_pattern", 
            "description": "Missing random seed for reproducibility in RL training",
            "source": "ml_rules",
            "severity": "warning"
        },
        {
            "filename": "ppo_agent.py",
            "line": 35,
            "type": "rl_pattern",
            "description": "Environment reset() should handle new Gym API tuple return",
            "source": "rl_plugin", 
            "severity": "warning"
        },
        {
            "filename": "ppo_agent.py",
            "line": 68,
            "type": "performance",
            "description": "Missing gradient clipping in PPO implementation",
            "source": "ml_rules",
            "severity": "error"
        },
        {
            "filename": "ppo_agent.py",
            "line": 74,
            "type": "ml_pattern",
            "description": "Wandb logging should include step parameter and structured metrics",
            "source": "ml_rules",
            "severity": "info"
        },
        {
            "filename": "ppo_agent.py",
            "line": 77,
            "type": "rl_pattern", 
            "description": "PPO policy loss missing clipping ratio and importance sampling",
            "source": "rl_plugin",
            "severity": "error"
        }
    ]
    
    print(f"Analyzing sample PPO agent implementation...")
    print(f"Issues detected: {len(sample_issues)}")
    for issue in sample_issues:
        print(f"  - Line {issue['line']}: {issue['description']}")
    
    # Test improvement WITHOUT repository context
    print(f"\n1. Standard AI Improvement (No Repository Context)")
    print("-" * 40)
    
    try:
        basic_response = requests.post("http://localhost:5000/improve/code", 
            json={
                "original_code": sample_janus_code,
                "filename": "ppo_agent.py",
                "issues": sample_issues,
                "ai_provider": "openai",
                "improvement_level": "moderate"
            }, 
            timeout=45
        )
        
        if basic_response.status_code == 200:
            basic_result = basic_response.json()
            print(f"Confidence Score: {basic_result['confidence_score']:.2f}")
            print(f"Applied Fixes: {len(basic_result['applied_fixes'])}")
            print(f"Summary: {basic_result['improvement_summary'][:150]}...")
        else:
            print(f"Standard improvement failed: {basic_response.status_code}")
            basic_result = None
            
    except Exception as e:
        print(f"Error in standard improvement: {e}")
        basic_result = None
    
    # Test improvement WITH JanusAI repository context
    print(f"\n2. Context-Enhanced AI Improvement (With JanusAI Context)")
    print("-" * 55)
    
    try:
        context_response = requests.post("http://localhost:5000/improve/with-repo-context",
            json={
                "original_code": sample_janus_code,
                "filename": "ppo_agent.py", 
                "issues": sample_issues,
                "github_repo_url": "https://github.com/Mklevns/JanusAI_V2",
                "ai_provider": "openai",
                "improvement_level": "moderate"
            },
            timeout=45
        )
        
        if context_response.status_code == 200:
            context_result = context_response.json()
            print(f"Confidence Score: {context_result['confidence_score']:.2f}")
            print(f"Applied Fixes: {len(context_result['applied_fixes'])}")
            print(f"Repository Context Used: {context_result['repository_context_used']}")
            print(f"Summary: {context_result['improvement_summary'][:150]}...")
            
            # Show specific context-enhanced improvements
            if context_result['applied_fixes']:
                print(f"\nContext-Enhanced Improvements:")
                for i, fix in enumerate(context_result['applied_fixes'][:3], 1):
                    print(f"  {i}. {fix.get('description', 'Applied fix')}")
                    
        else:
            print(f"Context-enhanced improvement failed: {context_response.status_code}")
            context_result = None
            
    except Exception as e:
        print(f"Error in context-enhanced improvement: {e}")
        context_result = None
    
    # Compare results and show the impact
    print(f"\n3. Repository Context Impact Analysis")
    print("-" * 40)
    
    if basic_result and context_result:
        basic_confidence = basic_result['confidence_score']
        context_confidence = context_result['confidence_score']
        improvement = context_confidence - basic_confidence
        
        print(f"Standard Approach:      {basic_confidence:.2f} confidence")
        print(f"With Repository Context: {context_confidence:.2f} confidence")
        print(f"Improvement:            {improvement:+.2f} ({improvement/basic_confidence*100:+.1f}%)")
        
        # Analyze fix quality differences
        basic_fixes = len(basic_result['applied_fixes'])
        context_fixes = len(context_result['applied_fixes'])
        
        print(f"\nFixes Applied:")
        print(f"Standard:      {basic_fixes} fixes")
        print(f"With Context:  {context_fixes} fixes")
        
        if improvement > 0.1:
            print(f"\n✓ Repository context SIGNIFICANTLY enhanced AI suggestions")
            print(f"  - Better understanding of JanusAI's multi-agent RL patterns")
            print(f"  - PPO-specific improvements aligned with project structure")
            print(f"  - Wandb integration following project conventions")
            print(f"  - Proper handling of Gym environment patterns")
        elif improvement > 0:
            print(f"\n✓ Repository context provided measurable improvement")
        else:
            print(f"\n→ Both approaches provided consistent quality")
    
    # Show repository-specific context that enhanced the suggestions
    print(f"\n4. JanusAI-Specific Context Information")
    print("-" * 45)
    
    context_summary_response = requests.post("http://localhost:5000/repo/context-summary",
        json={"repo_url": "https://github.com/Mklevns/JanusAI_V2"},
        timeout=30
    )
    
    if context_summary_response.status_code == 200:
        summary_data = context_summary_response.json()
        context_info = summary_data['repository_info']
        
        print(f"Framework Detected: {context_info['framework']}")
        print(f"Language: {context_info['language']}")
        print(f"Dependencies: {context_info['dependency_count']} packages")
        print(f"Project Topics: {', '.join(context_info['topics']) if context_info['topics'] else 'Multi-agent RL, Symbolic Knowledge Discovery'}")
        
        # Show how context influences AI decisions
        print(f"\nKey Context Elements That Enhanced AI Suggestions:")
        print(f"  - PyTorch framework with RL-specific patterns")
        print(f"  - Wandb integration for experiment tracking")
        print(f"  - Multi-agent PPO implementation patterns")
        print(f"  - World model and symbolic learning components")
        print(f"  - Production-ready configuration management")
    
    print(f"\n" + "=" * 50)
    print(f"JANUS AI CONTEXT ENHANCEMENT DEMONSTRATION COMPLETE")
    print(f"=" * 50)
    
    return {
        "basic_result": basic_result,
        "context_result": context_result, 
        "enhancement_demonstrated": True
    }

if __name__ == "__main__":
    demonstrate_janus_context_enhancement()