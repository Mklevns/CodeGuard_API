"""
GitHub Repository Context Integration - Live Demonstration
Shows how repository context dramatically improves AI code suggestions.
"""

import requests
import json

def demonstrate_repository_context_enhancement():
    """Complete demonstration of GitHub repository context features."""
    
    print("GitHub Repository Context Integration - Live Demo")
    print("=" * 55)
    
    # Sample ML code with common issues
    sample_code = '''
import torch
import numpy as np
import pickle
import random

class MLModel:
    def __init__(self):
        # Missing seed setting
        self.model = torch.nn.Linear(10, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def load_data(self, filename):
        # Security vulnerability - pickle.load
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def train(self, data):
        for i in range(100):
            # Missing proper batch handling
            output = self.model(data)
            loss = torch.nn.functional.mse_loss(output, torch.randn(100, 1))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            print("Loss:", loss.item())
'''
    
    # Step 1: Analyze PyTorch repository for context
    print("\n1. Analyzing PyTorch Examples Repository...")
    repo_analysis = requests.post("http://localhost:5000/repo/analyze", json={
        "repo_url": "https://github.com/pytorch/examples"
    })
    
    if repo_analysis.status_code == 200:
        repo_data = repo_analysis.json()
        print(f"   Repository: {repo_data['repository']['name']}")
        print(f"   Framework: {repo_data['repository']['framework']}")
        print(f"   Language: {repo_data['repository']['language']}")
        print(f"   Dependencies: {repo_data['repository']['dependencies_count']}")
        print("   ✓ Repository context extracted successfully")
    else:
        print("   ✗ Repository analysis failed")
        return
    
    # Step 2: Get audit results first
    print("\n2. Running CodeGuard Analysis...")
    audit_response = requests.post("http://localhost:5000/audit", json={
        "files": [{"filename": "model.py", "content": sample_code}]
    })
    
    if audit_response.status_code == 200:
        audit_data = audit_response.json()
        issues = audit_data['issues']
        print(f"   Issues detected: {len(issues)}")
        for issue in issues[:3]:  # Show first 3 issues
            print(f"   - {issue['type']}: {issue['description']}")
        print("   ✓ Code analysis completed")
    else:
        print("   ✗ Audit failed")
        return
    
    # Step 3: Test improvement without repository context
    print("\n3. AI Improvement WITHOUT Repository Context...")
    basic_improvement = requests.post("http://localhost:5000/improve/code", json={
        "original_code": sample_code,
        "filename": "model.py",
        "issues": issues,
        "ai_provider": "openai"
    })
    
    basic_result = None
    if basic_improvement.status_code == 200:
        basic_result = basic_improvement.json()
        print(f"   Confidence Score: {basic_result['confidence_score']:.2f}")
        print(f"   Fixes Applied: {len(basic_result['applied_fixes'])}")
        print("   ✓ Basic improvement completed")
    else:
        print("   ✗ Basic improvement failed")
    
    # Step 4: Test improvement WITH repository context
    print("\n4. AI Improvement WITH Repository Context...")
    context_improvement = requests.post("http://localhost:5000/improve/with-repo-context", json={
        "original_code": sample_code,
        "filename": "model.py", 
        "issues": issues,
        "github_repo_url": "https://github.com/pytorch/examples",
        "ai_provider": "openai"
    })
    
    context_result = None
    if context_improvement.status_code == 200:
        context_result = context_improvement.json()
        print(f"   Confidence Score: {context_result['confidence_score']:.2f}")
        print(f"   Fixes Applied: {len(context_result['applied_fixes'])}")
        print(f"   Context Used: {context_result['repository_context_used']}")
        print("   ✓ Context-enhanced improvement completed")
    else:
        print("   ✗ Context-enhanced improvement failed")
    
    # Step 5: Compare results
    print("\n5. Results Comparison")
    print("-" * 30)
    
    if basic_result and context_result:
        basic_confidence = basic_result['confidence_score']
        context_confidence = context_result['confidence_score']
        improvement = context_confidence - basic_confidence
        
        print(f"Without Context: {basic_confidence:.2f} confidence")
        print(f"With Context:    {context_confidence:.2f} confidence")
        print(f"Improvement:     {improvement:+.2f} ({improvement/basic_confidence*100:+.1f}%)")
        
        if improvement > 0:
            print("\n✓ Repository context ENHANCED AI suggestions")
            print("  - Better understanding of project patterns")
            print("  - More appropriate fix recommendations")
            print("  - Higher confidence in applied changes")
        else:
            print("\n→ Repository context provided consistent quality")
    
    # Step 6: Show context summary
    print("\n6. Repository Context Summary")
    print("-" * 30)
    context_summary = requests.post("http://localhost:5000/repo/context-summary", json={
        "repo_url": "https://github.com/pytorch/examples"
    })
    
    if context_summary.status_code == 200:
        summary_data = context_summary.json()
        context_text = summary_data['context_summary']
        print(f"Context Length: {len(context_text)} characters")
        print("Context Preview:")
        print(context_text[:200] + "..." if len(context_text) > 200 else context_text)
    
    print("\n" + "=" * 55)
    print("GITHUB INTEGRATION DEMONSTRATION COMPLETE")
    print("=" * 55)
    
    return {
        "repository_analysis": repo_data if 'repo_data' in locals() else None,
        "basic_improvement": basic_result,
        "context_improvement": context_result,
        "improvement_delta": improvement if 'improvement' in locals() else 0
    }

if __name__ == "__main__":
    demonstrate_repository_context_enhancement()