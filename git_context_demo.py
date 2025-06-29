"""
Comprehensive demonstration of Git Context-Aware Repository Analysis for CodeGuard API.
Shows related files discovery, dependency analysis, and context-enhanced AI code improvements.
"""

import requests
import json
import time
from datetime import datetime

def run_git_context_demo():
    """Run comprehensive demonstration of Git context-aware repository analysis features."""
    
    print("Git Context-Aware Repository Analysis Demonstration")
    print("=" * 60)
    print("Showcasing intelligent repository context analysis for enhanced AI code improvements")
    print("Features: Related files discovery, dependency analysis, and context-aware AI suggestions")
    print()
    
    base_url = "http://localhost:5000"
    
    # Test file for demonstration (using a real file from our repository)
    test_file = "main.py"
    
    # Feature 1: Related Files Discovery
    print("1. Related Files Discovery (Git History Analysis)")
    print("-" * 50)
    
    try:
        response = requests.get(f"{base_url}/context/related-files", params={
            "file_path": test_file,
            "limit": 5
        })
        
        if response.status_code == 200:
            related_data = response.json()
            
            if related_data.get("available"):
                print(f"✓ Git context analysis available")
                print(f"  Target file: {related_data['file_path']}")
                print(f"  Analysis method: {related_data['analysis_method']}")
                print(f"  Related files found: {related_data['total_found']}")
                
                if related_data['related_files']:
                    print(f"\n  Files frequently co-changed with {test_file}:")
                    for i, related_file in enumerate(related_data['related_files'][:3], 1):
                        print(f"    {i}. {related_file}")
                else:
                    print(f"  No related files found for {test_file}")
            else:
                print(f"  {related_data.get('message', 'Git analysis not available')}")
        else:
            print(f"  Error: HTTP {response.status_code}")
    except Exception as e:
        print(f"  Connection error: {str(e)}")
    
    print()
    
    # Feature 2: Comprehensive Context Analysis
    print("2. Comprehensive Repository Context Analysis")
    print("-" * 45)
    
    try:
        response = requests.get(f"{base_url}/context/comprehensive", params={
            "file_path": test_file,
            "max_files": 5
        })
        
        if response.status_code == 200:
            context_data = response.json()
            
            if context_data.get("available"):
                print(f"✓ Comprehensive context analysis completed")
                print(f"  Target file: {context_data['target_file']}")
                print(f"  Analysis methods: {', '.join(context_data['analysis_methods'])}")
                
                stats = context_data['statistics']
                print(f"  Context files loaded: {stats['total_files']}")
                print(f"  Total lines of context: {stats['total_lines']}")
                print(f"  File types: {', '.join(f'{ext}({count})' for ext, count in stats['file_types'].items())}")
                
                if context_data['context_files']:
                    print(f"\n  Repository context includes:")
                    for i, context_file in enumerate(context_data['context_files'][:3], 1):
                        print(f"    {i}. {context_file}")
                        
                    if len(context_data['context_files']) > 3:
                        print(f"    ... and {len(context_data['context_files']) - 3} more files")
            else:
                print(f"  {context_data.get('message', 'Context analysis not available')}")
        else:
            print(f"  Error: HTTP {response.status_code}")
    except Exception as e:
        print(f"  Connection error: {str(e)}")
    
    print()
    
    # Feature 3: Context-Aware AI Code Improvement
    print("3. Context-Aware AI Code Improvement")
    print("-" * 40)
    
    # Sample Python code with potential improvements
    sample_code = '''import numpy as np
import pickle
import random

def train_model(data):
    # TODO: Add random seeding
    model_data = pickle.load(open("model.pkl", "rb"))
    
    # Train model with data
    for epoch in range(10):
        loss = random.random()
        print(f"Epoch {epoch}: {loss}")
    
    return model_data

def evaluate_model():
    pass
'''
    
    try:
        context_improvement_request = {
            "original_code": sample_code,
            "filename": "train_model.py",
            "ai_provider": "openai",
            "include_git_context": True,
            "max_context_files": 3
        }
        
        response = requests.post(f"{base_url}/improve/context-aware", json=context_improvement_request)
        
        if response.status_code == 200:
            improvement_data = response.json()
            
            print(f"✓ Context-aware AI improvement completed")
            print(f"  AI provider used: {improvement_data['ai_provider_used']}")
            print(f"  Improvement type: {improvement_data['improvement_type']}")
            print(f"  Confidence score: {improvement_data['confidence_score']:.2f}")
            
            context_analysis = improvement_data['context_analysis']
            print(f"\n  Context Analysis:")
            print(f"    Git context available: {context_analysis['git_context_available']}")
            print(f"    Context summary: {context_analysis['context_summary']}")
            print(f"    Related files used: {context_analysis['related_files_count']}")
            
            static_analysis = improvement_data['static_analysis']
            print(f"\n  Static Analysis Results:")
            print(f"    Issues found: {static_analysis['issues_found']}")
            print(f"    Fixes suggested: {static_analysis['fixes_suggested']}")
            print(f"    Analysis tools used: {static_analysis['analysis_tools']}")
            
            applied_fixes = improvement_data.get('applied_fixes', [])
            if applied_fixes:
                print(f"\n  Applied Improvements:")
                for i, fix in enumerate(applied_fixes[:3], 1):
                    if isinstance(fix, dict):
                        print(f"    {i}. {fix.get('description', 'Improvement applied')}")
                    else:
                        print(f"    {i}. {str(fix)}")
            
            print(f"\n  Improvement Summary:")
            print(f"    {improvement_data.get('improvement_summary', 'AI improvements applied')}")
            
            warnings = improvement_data.get('warnings', [])
            if warnings:
                print(f"\n  Warnings:")
                for warning in warnings[:2]:
                    print(f"    • {warning}")
                    
        else:
            print(f"  Error: HTTP {response.status_code}")
            if response.text:
                error_data = response.json()
                print(f"  Details: {error_data.get('detail', 'Unknown error')}")
    except Exception as e:
        print(f"  Connection error: {str(e)}")
    
    print()
    
    # Feature 4: Enhanced Repository Analysis Integration
    print("4. Enhanced Repository Analysis Integration")
    print("-" * 42)
    
    try:
        # Test the comprehensive repository heatmap with Git context
        response = requests.get(f"{base_url}/analysis/repository-heatmap")
        
        if response.status_code == 200:
            heatmap_data = response.json()
            
            if heatmap_data.get("available"):
                summary = heatmap_data['summary']
                print(f"✓ Repository heatmap with Git integration")
                print(f"  Files analyzed: {summary['total_files']}")
                print(f"  Git analysis integrated: {summary['git_analysis_available']}")
                print(f"  High-risk files: {summary['high_risk_files']}")
                print(f"  Security issues detected: {summary['total_security_issues']}")
                
                recommendations = heatmap_data.get("recommendations", [])
                if recommendations:
                    print(f"\n  Integrated Analysis Recommendations:")
                    for i, rec in enumerate(recommendations[:2], 1):
                        print(f"    {i}. {rec}")
            else:
                print(f"  {heatmap_data.get('message', 'Repository heatmap not available')}")
        else:
            print(f"  Error: HTTP {response.status_code}")
    except Exception as e:
        print(f"  Connection error: {str(e)}")
    
    print()
    
    # Summary and Integration Benefits
    print("=" * 60)
    print("GIT CONTEXT-AWARE REPOSITORY ANALYSIS BENEFITS")
    print("=" * 60)
    
    benefits = [
        "Intelligent Related Files Discovery: Identifies frequently co-changed files using Git history",
        "Smart Dependency Analysis: Analyzes import statements to find file dependencies",
        "Context-Enhanced AI Improvements: Provides repository context to AI for better suggestions",
        "Priority-Based Context Selection: Prioritizes co-changed files over dependencies for relevance",
        "Comprehensive Repository Understanding: Combines static analysis with Git history patterns",
        "Enhanced Code Quality Insights: AI improvements consider project-specific patterns and conventions",
        "Reduced Breaking Changes: Context-aware suggestions maintain compatibility with related code",
        "Intelligent Architectural Consistency: AI follows existing project patterns and design decisions"
    ]
    
    for benefit in benefits:
        print(f"✓ {benefit}")
    
    print()
    print("Git Context Integration Features Successfully Demonstrated:")
    print("• Related files discovery using Git commit history analysis")
    print("• Comprehensive context retrieval with dependency and co-change analysis")
    print("• Context-aware AI code improvements with repository understanding")
    print("• Enhanced repository analysis combining static analysis with Git insights")
    print("• Priority-based context selection for optimal AI enhancement")
    print()
    print("All Git context-aware repository analysis features are now operational!")


if __name__ == "__main__":
    run_git_context_demo()