import subprocess
import tempfile
import os
import re
from typing import List, Dict, Tuple
from models import AuditRequest, AuditResponse, Issue, Fix
from enhanced_audit import EnhancedAuditEngine

def analyze_code(request: AuditRequest) -> AuditResponse:
    """
    Analyzes Python code files using multiple static analysis tools and ML/RL rules.
    
    Args:
        request: AuditRequest containing files to analyze
        
    Returns:
        AuditResponse with analysis results from multiple tools
    """
    # Use the enhanced audit engine for comprehensive analysis
    enhanced_engine = EnhancedAuditEngine()
    return enhanced_engine.analyze_code(request)

# Legacy functions removed - now using enhanced_audit.py for comprehensive analysis
