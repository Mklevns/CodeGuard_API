"""
Clean Code Prompt Enhancer for CodeGuard API.
Ensures AI returns complete, clean code replacements instead of appending fixes to original code.
"""

from typing import Dict, Any, List
from models import Issue, Fix, CodeFile


class CleanCodePromptEnhancer:
    """Enhances prompts to ensure clean, complete code output."""
    
    def __init__(self):
        self.output_templates = {
            "before_after": self._get_before_after_template(),
            "complete_replacement": self._get_complete_replacement_template(),
            "targeted_fixes": self._get_targeted_fixes_template()
        }
    
    def enhance_prompt_for_clean_output(self, base_prompt: str, 
                                      original_code: str, 
                                      issues: List[Issue],
                                      output_style: str = "complete_replacement") -> str:
        """
        Enhance the base prompt to ensure clean code output.
        
        Args:
            base_prompt: The base system prompt
            original_code: Original code to be fixed
            issues: List of issues to fix
            output_style: "before_after", "complete_replacement", or "targeted_fixes"
        """
        
        # Count different types of issues
        issue_analysis = self._analyze_issues(issues)
        
        # Choose appropriate template
        if output_style == "before_after" and len(issues) <= 5:
            template = self.output_templates["before_after"]
        elif output_style == "targeted_fixes" and issue_analysis["has_simple_fixes"]:
            template = self.output_templates["targeted_fixes"]
        else:
            template = self.output_templates["complete_replacement"]
        
        # Build enhanced prompt
        enhanced_prompt = f"""{base_prompt}

=== CRITICAL OUTPUT FORMATTING REQUIREMENTS ===

{template}

=== CODE ANALYSIS CONTEXT ===
- Total issues to fix: {len(issues)}
- Issue types: {', '.join(issue_analysis['types'])}
- Complexity level: {issue_analysis['complexity']}
- Original code length: {len(original_code.split())} lines

=== SPECIFIC INSTRUCTIONS FOR THIS CODE ===
"""

        # Add specific instructions based on issue types
        if "security" in issue_analysis['types']:
            enhanced_prompt += """
SECURITY FIXES REQUIRED:
- Replace dangerous functions with safe alternatives
- Provide secure implementation patterns
- Explain security improvements in applied_fixes
"""

        if "style" in issue_analysis['types']:
            enhanced_prompt += """
STYLE FIXES REQUIRED:
- Remove unused imports completely
- Fix line length by breaking long lines properly
- Maintain proper code formatting
- Do NOT add unnecessary comments for style fixes
"""

        if "ml" in issue_analysis['types'] or "rl" in issue_analysis['types']:
            enhanced_prompt += """
ML/RL FIXES REQUIRED:
- Add proper random seeding for reproducibility
- Fix environment handling patterns
- Ensure proper model training setup
"""

        enhanced_prompt += f"""
=== VALIDATION CHECKLIST ===
Before returning your response, verify:
✓ The improved_code contains the COMPLETE fixed file
✓ All {len(issues)} issues have been addressed
✓ No original code appears before or after the fixes
✓ The code is ready to replace the original file entirely
✓ Applied_fixes describes exactly what was changed
✓ JSON format is valid and complete

REMEMBER: Return the ENTIRE corrected code file, not just the changes!
"""

        return enhanced_prompt
    
    def _analyze_issues(self, issues: List[Issue]) -> Dict[str, Any]:
        """Analyze issues to determine output strategy."""
        types = list(set(issue.type for issue in issues))
        severities = [issue.severity for issue in issues]
        
        # Determine complexity
        complexity = "simple"
        if len(issues) > 10:
            complexity = "complex"
        elif any(severity == "error" for severity in severities):
            complexity = "moderate"
        
        # Check for simple fixes (imports, style)
        simple_fix_types = {"style", "import", "formatting"}
        has_simple_fixes = any(issue.type in simple_fix_types for issue in issues)
        
        return {
            "types": types,
            "complexity": complexity,
            "has_simple_fixes": has_simple_fixes,
            "error_count": sum(1 for s in severities if s == "error"),
            "warning_count": sum(1 for s in severities if s == "warning")
        }
    
    def _get_complete_replacement_template(self) -> str:
        """Template for complete code replacement output."""
        return """
**OUTPUT FORMAT: COMPLETE CODE REPLACEMENT**

Your response must be valid JSON with these exact fields:

```json
{
  "improved_code": "THE ENTIRE CORRECTED CODE FILE - NOT ORIGINAL + FIXES",
  "applied_fixes": [
    "Removed unused import: matplotlib.animation.FuncAnimation",
    "Fixed line length on line 27 by breaking into multiple lines",
    "Replaced pickle.load() with safe torch.load() for security"
  ],
  "improvement_summary": "Fixed 3 issues: removed 2 unused imports and 1 security vulnerability",
  "confidence_score": 0.95,
  "warnings": ["Verify torch.load() path is trusted"]
}
```

**CRITICAL: The 'improved_code' field must contain:**
- The complete, corrected code file
- All imports properly organized
- All functions with fixes applied
- Ready to save as the new file
- NO reference to "original code" or "here's the fixed version"
"""

    def _get_before_after_template(self) -> str:
        """Template for before/after comparison output."""
        return """
**OUTPUT FORMAT: BEFORE/AFTER COMPARISON**

Show specific changes with before/after snippets:

```json
{
  "improved_code": "THE COMPLETE CORRECTED CODE FILE",
  "applied_fixes": [
    {
      "description": "Removed unused import",
      "before": "from matplotlib.animation import FuncAnimation",
      "after": "# Import removed - was unused",
      "line": 13
    },
    {
      "description": "Fixed line length",
      "before": "def very_long_function_name_that_exceeds_limit():",
      "after": "def very_long_function_name_that_exceeds_limit(\n        ):",
      "line": 27
    }
  ],
  "improvement_summary": "Applied targeted fixes to specific lines",
  "confidence_score": 0.90,
  "warnings": []
}
```
"""

    def _get_targeted_fixes_template(self) -> str:
        """Template for targeted fixes output."""
        return """
**OUTPUT FORMAT: TARGETED FIXES**

Focus on specific line-by-line improvements:

```json
{
  "improved_code": "THE COMPLETE CORRECTED CODE FILE",
  "applied_fixes": [
    "Line 13: Removed unused import 'from matplotlib.animation import FuncAnimation'",
    "Line 15: Removed unused import 'from pathlib import Path'",
    "Line 27: Broke long line into multiple lines for PEP 8 compliance"
  ],
  "improvement_summary": "Cleaned up imports and fixed line length issues",
  "confidence_score": 0.88,
  "warnings": []
}
```

**IMPORTANT:** The improved_code must be the complete file with all fixes applied, not a diff or partial code.
"""


def enhance_prompt_for_clean_code_output(base_prompt: str, 
                                       original_code: str,
                                       issues: List[Issue],
                                       output_style: str = "complete_replacement") -> str:
    """
    Enhance any prompt to ensure clean code output.
    
    Args:
        base_prompt: The base system prompt to enhance
        original_code: Original code being fixed
        issues: Issues detected by CodeGuard
        output_style: Output format preference
    """
    enhancer = CleanCodePromptEnhancer()
    return enhancer.enhance_prompt_for_clean_output(
        base_prompt, original_code, issues, output_style
    )


# Test the prompt enhancer
if __name__ == "__main__":
    # Test with sample data
    from models import Issue
    
    test_issues = [
        Issue(
            filename="test.py",
            line=13,
            type="style",
            description="Unused import: matplotlib.animation.FuncAnimation",
            source="flake8",
            severity="warning"
        ),
        Issue(
            filename="test.py",
            line=27,
            type="style", 
            description="Line too long (88 > 79 characters)",
            source="flake8",
            severity="warning"
        )
    ]
    
    test_code = """
import torch
from matplotlib.animation import FuncAnimation
import numpy as np

def very_long_function_name_that_definitely_exceeds_the_line_length_limit():
    return torch.tensor([1, 2, 3])
"""

    base_prompt = "You are a Python code improvement expert."
    
    enhanced = enhance_prompt_for_clean_code_output(
        base_prompt, test_code, test_issues, "complete_replacement"
    )
    
    print("Enhanced Prompt Preview:")
    print("=" * 50)
    print(enhanced[:500] + "...")
    print("\nKey features:")
    print("✓ Emphasizes complete code replacement")
    print("✓ Provides clear JSON format requirements") 
    print("✓ Includes validation checklist")
    print("✓ Analyzes issue complexity automatically")