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

def _analyze_single_file(file_path: str, original_filename: str) -> Tuple[List[Issue], List[Fix]]:
    """
    Analyzes a single Python file using flake8.
    
    Args:
        file_path: Path to the temporary file
        original_filename: Original filename for reporting
        
    Returns:
        Tuple of (issues, fixes) lists
    """
    issues = []
    fixes = []
    
    try:
        # Run flake8 with custom format
        result = subprocess.run(
            [
                "flake8",
                file_path,
                "--format=%(row)d::%(code)s::%(text)s",
                "--max-line-length=88",  # Black-compatible line length
                "--extend-ignore=E203,W503"  # Black-compatible ignores
            ],
            capture_output=True,
            text=True,
            timeout=30  # Prevent hanging
        )
        
        # Parse flake8 output
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
                
            parts = line.split("::")
            if len(parts) >= 3:
                line_num_str, code, description = parts[0], parts[1], "::".join(parts[2:])
                
                try:
                    line_num = int(line_num_str)
                except ValueError:
                    continue
                
                # Categorize issue type
                issue_type = _categorize_issue(code)
                
                issue = Issue(
                    filename=original_filename,
                    line=line_num,
                    type=issue_type,
                    description=f"{code}: {description.strip()}"
                )
                issues.append(issue)
                
                # Generate fix suggestions
                fix_suggestion = _generate_fix_suggestion(code, description.strip())
                if fix_suggestion:
                    fix = Fix(
                        filename=original_filename,
                        line=line_num,
                        suggestion=fix_suggestion
                    )
                    fixes.append(fix)
        
    except subprocess.TimeoutExpired:
        issues.append(Issue(
            filename=original_filename,
            line=1,
            type="error",
            description="Analysis timed out"
        ))
    except FileNotFoundError:
        issues.append(Issue(
            filename=original_filename,
            line=1,
            type="error",
            description="flake8 not found - static analysis tool unavailable"
        ))
    except Exception as e:
        issues.append(Issue(
            filename=original_filename,
            line=1,
            type="error",
            description=f"Analysis failed: {str(e)}"
        ))
    
    return issues, fixes

def _categorize_issue(code: str) -> str:
    """
    Categorizes flake8 error codes into issue types.
    
    Args:
        code: Flake8 error code (e.g., 'F401', 'E302')
        
    Returns:
        Issue type category
    """
    if code.startswith('F'):
        # F codes are logical errors (pyflakes)
        return "error"
    elif code.startswith('E') or code.startswith('W'):
        # E codes are style errors, W codes are warnings (pycodestyle)
        return "style"
    elif code.startswith('C'):
        # C codes are complexity issues
        return "complexity"
    elif code.startswith('N'):
        # N codes are naming convention issues
        return "naming"
    else:
        return "best_practice"

def _generate_fix_suggestion(code: str, description: str) -> str:
    """
    Generates fix suggestions based on flake8 error codes and descriptions.
    
    Args:
        code: Flake8 error code
        description: Error description
        
    Returns:
        Fix suggestion string or empty string if no suggestion available
    """
    # Common fix suggestions based on error codes
    fix_map = {
        'F401': 'Remove unused import',
        'F811': 'Remove or rename duplicate definition',
        'F841': 'Use the variable or remove it',
        'E302': 'Add blank lines before function/class definition',
        'E303': 'Remove extra blank lines',
        'E501': 'Break long line or use shorter variable names',
        'E701': 'Put multiple statements on separate lines',
        'E711': 'Use "is" or "is not" for None comparison',
        'E712': 'Use "is" or "is not" for boolean comparison',
        'W291': 'Remove trailing whitespace',
        'W292': 'Add newline at end of file',
        'W293': 'Remove whitespace on blank line'
    }
    
    if code in fix_map:
        return fix_map[code]
    
    # Pattern-based suggestions
    if 'unused import' in description.lower():
        return 'Remove unused import'
    elif 'undefined name' in description.lower():
        return 'Define the variable or import the module'
    elif 'indentation' in description.lower():
        return 'Fix indentation to use 4 spaces'
    elif 'whitespace' in description.lower():
        return 'Remove trailing whitespace'
    elif 'blank line' in description.lower():
        return 'Adjust blank line spacing according to PEP 8'
    elif 'line too long' in description.lower():
        return 'Break line to stay under character limit'
    
    return ""
