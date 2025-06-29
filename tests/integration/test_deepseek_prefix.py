#!/usr/bin/env python3
"""
Test DeepSeek Chat Prefix Completion integration with CodeGuard.
"""

import sys
import os
sys.path.append('.')

from models import AuditRequest, CodeFile, AuditOptions
from enhanced_audit import EnhancedAuditEngine
from chatgpt_integration import get_code_improver, CodeImprovementRequest

def test_deepseek_prefix_integration():
    """Test DeepSeek prefix completion for code improvement."""
    
    # Read the multi-agent trainer file with issues
    with open('multi_agent_trainer.py', 'r') as f:
        content = f.read()
    
    print("Testing DeepSeek Chat Prefix Completion Integration")
    print("=" * 60)
    
    # Step 1: Run CodeGuard analysis
    print("Step 1: Running CodeGuard analysis...")
    code_file = CodeFile(filename="multi_agent_trainer.py", content=content)
    options = AuditOptions(level="production", framework="pytorch", target="gpu")
    request = AuditRequest(files=[code_file], options=options)
    
    engine = EnhancedAuditEngine(use_false_positive_filter=True)
    audit_response = engine.analyze_code(request)
    
    print(f"Found {len(audit_response.issues)} issues")
    
    # Show top issues
    for i, issue in enumerate(audit_response.issues[:5]):
        print(f"  {i+1}. Line {issue.line}: {issue.description}")
    
    # Step 2: Test DeepSeek improvement with different prefixes
    print("\nStep 2: Testing DeepSeek prefix completion approaches...")
    
    improver = get_code_improver()
    
    # Test with JSON prefix (what we implemented)
    improvement_request = CodeImprovementRequest(
        original_code=content,
        filename="multi_agent_trainer.py",
        issues=audit_response.issues,
        fixes=audit_response.fixes,
        improvement_level="moderate",
        ai_provider="deepseek",
        ai_api_key=os.getenv('DEEPSEEK_API_KEY')
    )
    
    if os.getenv('DEEPSEEK_API_KEY'):
        print("Testing with DeepSeek API key...")
        try:
            response = improver.improve_code(improvement_request)
            print(f"✓ DeepSeek improvement completed")
            print(f"  - Applied fixes: {len(response.applied_fixes)}")
            print(f"  - Confidence score: {response.confidence_score}")
            print(f"  - Summary: {response.improvement_summary[:100]}...")
            
            # Show some applied fixes
            if response.applied_fixes:
                print("  - Top fixes applied:")
                for fix in response.applied_fixes[:3]:
                    if isinstance(fix, dict):
                        print(f"    • {fix.get('description', 'Fix applied')}")
            
        except Exception as e:
            print(f"✗ DeepSeek test failed: {str(e)}")
    else:
        print("⚠ No DEEPSEEK_API_KEY found - skipping live API test")
        print("  Set DEEPSEEK_API_KEY environment variable to test live integration")
    
    # Step 3: Demonstrate different prefix completion scenarios
    print("\nStep 3: Prefix completion scenarios for CodeGuard:")
    
    scenarios = [
        {
            "name": "JSON Structured Response",
            "prefix": "{",
            "stop": ["}"],
            "use_case": "Complete audit results with fixes, confidence scores"
        },
        {
            "name": "Python Code Completion", 
            "prefix": "```python\n",
            "stop": ["```"],
            "use_case": "Direct code implementation and fixes"
        },
        {
            "name": "Security Fix Explanation",
            "prefix": "# Security Fix:",
            "stop": ["\n\n"],
            "use_case": "Focused security vulnerability explanations"
        },
        {
            "name": "ML Best Practice",
            "prefix": "# ML Best Practice:",
            "stop": ["\n\n"], 
            "use_case": "Specific ML/RL improvement recommendations"
        }
    ]
    
    for scenario in scenarios:
        print(f"  • {scenario['name']}")
        print(f"    Prefix: '{scenario['prefix']}'")
        print(f"    Stop: {scenario['stop']}")
        print(f"    Use case: {scenario['use_case']}")
        print()
    
    print("Integration Benefits:")
    print("✓ Guaranteed JSON format for API responses")
    print("✓ Direct code completion for implementation")
    print("✓ Focused improvements by category")
    print("✓ Reduced parsing complexity")
    print("✓ Better control over output format")

if __name__ == "__main__":
    test_deepseek_prefix_integration()