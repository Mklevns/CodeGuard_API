#!/usr/bin/env python3
"""Direct API test for CodeGuard with the multi-agent trainer file."""

import sys
import os
sys.path.append('.')

from models import AuditRequest, CodeFile, AuditOptions
from enhanced_audit import EnhancedAuditEngine

def test_direct_audit():
    """Test CodeGuard analysis directly without HTTP overhead."""
    
    # Read the multi-agent trainer file
    with open('multi_agent_trainer.py', 'r') as f:
        content = f.read()
    
    # Create audit request
    code_file = CodeFile(filename="multi_agent_trainer.py", content=content)
    options = AuditOptions(level="production", framework="pytorch", target="gpu")
    request = AuditRequest(files=[code_file], options=options)
    
    # Run analysis
    engine = EnhancedAuditEngine(use_false_positive_filter=True)
    response = engine.analyze_code(request)
    
    # Display results
    print("CodeGuard Analysis Results for multi_agent_trainer.py")
    print("=" * 60)
    print(f"Files analyzed: {len(response.files)}")
    print(f"Total issues found: {len(response.issues)}")
    print(f"Total fixes suggested: {len(response.fixes)}")
    print(f"Analysis tools used: {', '.join(response.tools_used)}")
    print()
    
    # Show issues by category
    categories = {}
    for issue in response.issues:
        cat = issue.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(issue)
    
    for category, issues in categories.items():
        print(f"{category.upper()} ISSUES ({len(issues)}):")
        for issue in issues[:3]:  # Show first 3 issues per category
            print(f"  Line {issue.line}: {issue.description}")
            if hasattr(issue, 'source_tool'):
                print(f"    Detected by: {issue.source_tool}")
        if len(issues) > 3:
            print(f"  ... and {len(issues) - 3} more {category} issues")
        print()
    
    # Show auto-fixable suggestions
    auto_fixable = [fix for fix in response.fixes if fix.auto_fixable]
    print(f"AUTO-FIXABLE SUGGESTIONS ({len(auto_fixable)}):")
    for fix in auto_fixable[:5]:  # Show first 5 auto-fixable
        print(f"  {fix.description}")
    if len(auto_fixable) > 5:
        print(f"  ... and {len(auto_fixable) - 5} more auto-fixable suggestions")
    
    return response

if __name__ == "__main__":
    test_direct_audit()