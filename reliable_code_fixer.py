"""
Reliable Code Fixer for CodeGuard API.
Provides immediate automated fixes with optional AI enhancement.
"""

import re
import os
from typing import List, Dict, Any
from models import Issue, Fix, CodeFile


class ReliableCodeFixer:
    """Applies reliable automated fixes with clean code output."""
    
    def __init__(self):
        self.fix_patterns = {
            'unused_import': [
                r'from matplotlib\.animation import FuncAnimation',
                r'from pathlib import Path',
                r'import pickle',
                r'from .*? import .*?FuncAnimation.*?',
                r'from .*? import .*?Path.*?'
            ],
            'security_fixes': {
                'pickle.load(': 'torch.load(',
                'import pickle': 'import torch',
                'eval(': '# SECURITY: eval() removed - use safe parsing',
                'exec(': '# SECURITY: exec() removed - use safe execution'
            },
            'line_length_fixes': True
        }
    
    def fix_code(self, original_code: str, issues: List[Issue], 
                 confidence_boost: float = 0.0) -> Dict[str, Any]:
        """
        Apply reliable automated fixes to code.
        
        Args:
            original_code: Original code to fix
            issues: List of issues to address
            confidence_boost: Confidence boost from custom prompts
            
        Returns:
            Dictionary with improved code and fix details
        """
        improved_code = original_code
        applied_fixes = []
        
        # Process each issue systematically
        for issue in issues:
            fix_result = self._apply_issue_fix(improved_code, issue)
            if fix_result['changed']:
                improved_code = fix_result['code']
                applied_fixes.append(fix_result['description'])
        
        # Final cleanup
        improved_code = self._cleanup_code(improved_code)
        
        # Calculate confidence
        base_confidence = 0.9 if applied_fixes else 0.6
        final_confidence = min(base_confidence + confidence_boost, 1.0)
        
        summary = f"Applied {len(applied_fixes)} reliable fixes" if applied_fixes else "No fixes needed"
        
        return {
            "improved_code": improved_code,
            "applied_fixes": applied_fixes,
            "improvement_summary": summary,
            "confidence_score": final_confidence,
            "warnings": []
        }
    
    def _apply_issue_fix(self, code: str, issue: Issue) -> Dict[str, Any]:
        """Apply fix for a specific issue."""
        original_code = code
        description = ""
        
        issue_desc = issue.description.lower()
        
        # Handle unused imports
        if issue.type == "style" and "unused import" in issue_desc:
            code, description = self._fix_unused_import(code, issue)
        
        # Handle security issues
        elif issue.type == "security":
            code, description = self._fix_security_issue(code, issue)
        
        # Handle line length issues
        elif issue.type == "style" and "line too long" in issue_desc:
            code, description = self._fix_long_line(code, issue)
        
        # Handle ML/RL seeding issues
        elif issue.type == "ml" and "seed" in issue_desc:
            code, description = self._fix_seeding_issue(code, issue)
        
        return {
            'code': code,
            'changed': code != original_code,
            'description': description
        }
    
    def _fix_unused_import(self, code: str, issue: Issue) -> tuple:
        """Fix unused import issues."""
        # Extract import name from issue description
        import_match = self._extract_import_from_description(issue.description)
        
        if not import_match:
            return code, ""
        
        lines = code.split('\n')
        new_lines = []
        removed = False
        
        for line in lines:
            # Skip lines that contain the unused import
            if (line.strip().startswith(('import ', 'from ')) and 
                import_match in line):
                removed = True
                continue
            new_lines.append(line)
        
        if removed:
            return '\n'.join(new_lines), f"Removed unused import: {import_match}"
        
        return code, ""
    
    def _extract_import_from_description(self, description: str) -> str:
        """Extract import name from error description."""
        # Handle different flake8 message formats
        patterns = [
            r"'([^']+)' imported but unused",
            r'"([^"]+)" imported but unused',
            r"unused import: (.+)",
            r"import (.+) but unused"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _fix_security_issue(self, code: str, issue: Issue) -> tuple:
        """Fix security vulnerabilities."""
        if "pickle" in issue.description.lower():
            if "pickle.load(" in code:
                code = code.replace("pickle.load(", "torch.load(")
                # Also ensure torch is imported
                if "import torch" not in code:
                    lines = code.split('\n')
                    # Find first import to insert after
                    for i, line in enumerate(lines):
                        if line.strip().startswith(('import ', 'from ')):
                            lines.insert(i, "import torch")
                            break
                    code = '\n'.join(lines)
                return code, "Replaced pickle.load() with torch.load() for security"
        
        elif "eval" in issue.description.lower():
            if "eval(" in code:
                # Comment out eval usage
                code = re.sub(r'eval\([^)]+\)', '# SECURITY: eval() removed', code)
                return code, "Removed dangerous eval() function call"
        
        return code, ""
    
    def _fix_long_line(self, code: str, issue: Issue) -> tuple:
        """Fix line length issues."""
        lines = code.split('\n')
        
        if issue.line <= len(lines):
            line = lines[issue.line - 1]
            
            if len(line) > 79:
                # Handle function definitions
                if "def " in line and "(" in line and "):" in line:
                    # Break function parameters
                    func_match = re.match(r'(\s*def\s+\w+)\(([^)]+)\):(.*)', line)
                    if func_match:
                        indent, params, end = func_match.groups()
                        if ',' in params:
                            param_list = [p.strip() for p in params.split(',')]
                            formatted_params = ',\n        '.join(param_list)
                            lines[issue.line - 1] = f"{indent}(\n        {formatted_params}\n    ):{end}"
                            return '\n'.join(lines), f"Fixed line length on line {issue.line}"
        
        return code, ""
    
    def _fix_seeding_issue(self, code: str, issue: Issue) -> tuple:
        """Add random seeding for reproducibility."""
        if "torch" in code and "torch.manual_seed" not in code:
            lines = code.split('\n')
            
            # Find the best place to insert seeding (after imports)
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    insert_pos = i + 1
            
            # Insert seeding code
            seed_lines = [
                "",
                "# Set seed for reproducibility",
                "torch.manual_seed(42)"
            ]
            
            for i, seed_line in enumerate(seed_lines):
                lines.insert(insert_pos + i, seed_line)
            
            return '\n'.join(lines), "Added random seed for reproducibility"
        
        return code, ""
    
    def _cleanup_code(self, code: str) -> str:
        """Clean up the final code."""
        # Remove excessive blank lines
        while '\n\n\n' in code:
            code = code.replace('\n\n\n', '\n\n')
        
        # Ensure file ends with single newline
        code = code.rstrip() + '\n'
        
        return code


def create_reliable_fixer() -> ReliableCodeFixer:
    """Create a reliable code fixer instance."""
    return ReliableCodeFixer()


# Test the reliable fixer
if __name__ == "__main__":
    from models import Issue
    
    test_code = """import torch
from matplotlib.animation import FuncAnimation
import pickle

def very_long_function_name_that_exceeds_the_character_limit_significantly():
    model = pickle.load(open('model.pkl', 'rb'))
    return model
"""
    
    test_issues = [
        Issue(
            filename="test.py",
            line=2,
            type="style",
            description="'matplotlib.animation.FuncAnimation' imported but unused",
            source="flake8",
            severity="warning"
        ),
        Issue(
            filename="test.py",
            line=5,
            type="style",
            description="line too long (88 > 79 characters)",
            source="flake8",
            severity="warning"
        ),
        Issue(
            filename="test.py",
            line=6,
            type="security",
            description="Use of pickle.load() poses security risk",
            source="custom_rules",
            severity="error"
        )
    ]
    
    fixer = create_reliable_fixer()
    result = fixer.fix_code(test_code, test_issues)
    
    print("Reliable Code Fixer Test Results:")
    print("=" * 40)
    print(f"Applied fixes: {len(result['applied_fixes'])}")
    print(f"Confidence: {result['confidence_score']:.1%}")
    print("\nFixes applied:")
    for fix in result['applied_fixes']:
        print(f"  - {fix}")
    
    print("\nImproved code:")
    print(result['improved_code'])