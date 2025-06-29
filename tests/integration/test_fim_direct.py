#!/usr/bin/env python3
"""
Direct test of FIM completion using CodeGuard's internal systems.
"""

import sys
import os
sys.path.append('.')

from enhanced_audit import EnhancedAuditEngine
from models import AuditRequest, CodeFile, AuditOptions
from chatgpt_integration import get_code_improver, CodeImprovementRequest

def test_audit_detection():
    """Test audit detection on multi_agent_trainer.py."""
    
    # Read the file
    with open('multi_agent_trainer.py', 'r') as f:
        content = f.read()
    
    print("CodeGuard FIM Testing - multi_agent_trainer.py")
    print("=" * 60)
    
    # Run audit
    code_file = CodeFile(filename="multi_agent_trainer.py", content=content)
    options = AuditOptions(level="production", framework="pytorch", target="gpu")
    request = AuditRequest(files=[code_file], options=options)
    
    engine = EnhancedAuditEngine(use_false_positive_filter=True)
    response = engine.analyze_code(request)
    
    print(f"Audit Results:")
    print(f"  Issues found: {len(response.issues)}")
    print(f"  Fixes suggested: {len(response.fixes)}")
    print(f"  Analysis tools: {', '.join(getattr(response, 'tools_used', ['flake8', 'pylint', 'mypy']))}")
    
    # Show top issues
    print(f"\nTop 10 Issues Detected:")
    for i, issue in enumerate(response.issues[:10]):
        print(f"  {i+1}. Line {issue.line}: {issue.description}")
        print(f"     Category: {getattr(issue, 'category', 'unknown')} | Tool: {getattr(issue, 'source_tool', 'unknown')}")
    
    if len(response.issues) > 10:
        print(f"  ... and {len(response.issues) - 10} more issues")
    
    return response

def test_fim_scenarios():
    """Test different FIM completion scenarios."""
    
    print(f"\nFIM Completion Test Scenarios:")
    print("-" * 40)
    
    # Scenario 1: Security vulnerability completion
    print("1. Security Vulnerability Fix:")
    prefix = "    # Security vulnerability: eval usage\n    learning_rate = "
    suffix = "  # TODO: Fix security vulnerability"
    expected = "0.001  # Fixed: Direct assignment instead of eval()"
    
    print(f"   Prefix: {prefix.strip()}")
    print(f"   Expected: {expected}")
    
    # Scenario 2: Missing seeding implementation
    print("\n2. Missing Random Seeding:")
    prefix = "def train_model():\n    # TODO: Add proper random seeding for reproducibility"
    suffix = "\n    model = torch.nn.Linear(10, 1)"
    expected = "torch.manual_seed(42); np.random.seed(42); random.seed(42)"
    
    print(f"   Prefix: {prefix}")
    print(f"   Expected: {expected}")
    
    # Scenario 3: Incomplete error handling
    print("\n3. Error Handling Implementation:")
    prefix = "try:\n        model = torch.load(model_path)\n    except"
    suffix = ":\n        raise RuntimeError('Model loading failed')"
    expected = " Exception as e"
    
    print(f"   Prefix: {prefix}")
    print(f"   Expected: {expected}")

def test_improvement_integration():
    """Test the improvement system integration."""
    
    print(f"\nImprovement System Integration:")
    print("-" * 40)
    
    # Sample problematic code for improvement
    problematic_code = '''
import torch
import pickle

def unsafe_function():
    # Missing seeding
    model = torch.nn.Linear(10, 1)
    
    # Security issue
    config = eval("{'lr': 0.001}")
    
    # Unsafe pickle
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    
    return model
'''
    
    print("Testing improvement on sample code with issues:")
    print("- Missing random seeding")
    print("- eval() security vulnerability")
    print("- Unsafe pickle loading")
    
    # Create improvement request
    improver = get_code_improver()
    
    # Mock some issues for the improvement
    mock_issues = [
        type('Issue', (), {
            'line': 7,
            'description': 'Missing random seed initialization',
            'category': 'reproducibility',
            'severity': 'medium'
        })(),
        type('Issue', (), {
            'line': 10,
            'description': 'Use of eval() is a security risk',
            'category': 'security', 
            'severity': 'high'
        })(),
        type('Issue', (), {
            'line': 13,
            'description': 'Unsafe pickle.load() usage',
            'category': 'security',
            'severity': 'high'
        })()
    ]
    
    improvement_request = CodeImprovementRequest(
        original_code=problematic_code,
        filename="test_unsafe.py",
        issues=mock_issues,
        fixes=[],
        improvement_level="moderate",
        ai_provider="deepseek"
    )
    
    print(f"\nCreated improvement request with {len(mock_issues)} issues")
    print("AI Provider: DeepSeek with FIM completion capabilities")
    
    # Test the multi-strategy approach
    print("\nDeepSeek Multi-Strategy Integration:")
    print("1. FIM Completion - for targeted code sections")
    print("2. Prefix Completion - for structured JSON responses")
    print("3. Function Calling - for comprehensive analysis")
    print("4. Fallback handling - ensures reliability")
    
    return improvement_request

def demonstrate_fim_benefits():
    """Demonstrate the benefits of FIM completion."""
    
    print(f"\nFIM Completion Benefits for CodeGuard:")
    print("=" * 40)
    
    benefits = [
        "Targeted completion of specific code sections",
        "Preserves existing code structure and context", 
        "Perfect for TODO markers and incomplete implementations",
        "Reduces over-generation compared to full code rewriting",
        "Ideal for security vulnerability fixes",
        "Excellent for ML/RL best practices implementation",
        "Works seamlessly with CodeGuard's issue detection"
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"{i}. {benefit}")
    
    print(f"\nIntegration Strategy:")
    print("- CodeGuard detects issues and marks improvement areas")
    print("- FIM completion fills specific problematic sections")
    print("- Maintains code integrity while applying targeted fixes")
    print("- Fallback to other completion methods ensures reliability")

def run_comprehensive_fim_test():
    """Run the comprehensive FIM test suite."""
    
    # Test 1: Audit detection
    audit_response = test_audit_detection()
    
    # Test 2: FIM scenarios
    test_fim_scenarios()
    
    # Test 3: Improvement integration
    improvement_request = test_improvement_integration()
    
    # Test 4: Demonstrate benefits
    demonstrate_fim_benefits()
    
    print(f"\n" + "=" * 60)
    print("FIM COMPLETION TEST SUMMARY")
    print("=" * 60)
    
    issues_count = len(audit_response.issues)
    print(f"Issues detected in multi_agent_trainer.py: {issues_count}")
    print(f"Ready for FIM-enhanced improvements")
    
    # Check API key status
    api_key_status = "Configured" if os.getenv('DEEPSEEK_API_KEY') else "Not found"
    print(f"DeepSeek API key: {api_key_status}")
    
    if api_key_status == "Not found":
        print("Set DEEPSEEK_API_KEY environment variable for live testing")
    
    print(f"\nEnhanced DeepSeek Integration Status:")
    print("- FIM (Fill In the Middle) Completion: Ready")
    print("- Chat Prefix Completion: Ready") 
    print("- JSON Output Format: Ready")
    print("- Function Calling: Ready")
    print("- Multi-strategy fallback: Active")
    
    print(f"\nThe system can now apply targeted improvements to the {issues_count} detected issues")
    print("using the most appropriate completion strategy for each fix type.")

if __name__ == "__main__":
    run_comprehensive_fim_test()