#!/usr/bin/env python3
import sys
import os
sys.path.append('.')

from enhanced_audit import EnhancedAuditEngine
from models import AuditRequest, CodeFile, AuditOptions

def quick_test():
    with open('multi_agent_trainer.py', 'r') as f:
        content = f.read()
    
    print("Testing multi_agent_trainer.py")
    print("Issues found:")
    
    code_file = CodeFile(filename="multi_agent_trainer.py", content=content)
    options = AuditOptions(level="production", framework="pytorch", target="gpu")
    request = AuditRequest(files=[code_file], options=options)
    
    engine = EnhancedAuditEngine(use_false_positive_filter=True)
    response = engine.analyze_code(request)
    
    for i, issue in enumerate(response.issues[:10]):
        print(f"{i+1}. Line {issue.line}: {issue.description}")
    
    print(f"\nTotal: {len(response.issues)} issues detected")
    print(f"Ready for FIM completion improvements")

if __name__ == "__main__":
    quick_test()