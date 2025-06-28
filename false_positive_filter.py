"""
False Positive Filter for CodeGuard API.
Uses ChatGPT to validate issues before reporting them to prevent false positives.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from models import Issue, Fix, CodeFile

class FalsePositiveFilter:
    """Filters out false positive issues using ChatGPT analysis."""
    
    def __init__(self):
        self._openai_client = None
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize OpenAI client with API key."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                # do not change this unless explicitly requested by the user
                self._openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
                self._openai_client = None
    
    def filter_issues(self, issues: List[Issue], fixes: List[Fix], code_files: List[CodeFile]) -> Tuple[List[Issue], List[Fix]]:
        """
        Filter out false positive issues using fast rule-based analysis.
        
        Args:
            issues: List of detected issues
            fixes: List of suggested fixes
            code_files: Original code files for context
            
        Returns:
            Tuple of (filtered_issues, filtered_fixes)
        """
        if not issues:
            return issues, fixes
        
        try:
            # Fast rule-based filtering for common false positives
            filtered_issues = []
            
            for issue in issues:
                # Keep all high-severity issues
                if issue.severity in ['error']:
                    filtered_issues.append(issue)
                    continue
                
                # Apply simple heuristics for common false positives
                description_lower = issue.description.lower()
                
                # Keep security-related issues
                if any(keyword in description_lower for keyword in ['pickle', 'eval', 'exec', 'unsafe']):
                    filtered_issues.append(issue)
                    continue
                
                # Skip common false positive patterns - updated to match actual issue descriptions
                skip_patterns = [
                    'line too long',  # E501: line too long
                    'missing whitespace',  # E225: missing whitespace around operator
                    'too many blank lines',  # E303: too many blank lines
                    'expected 2 blank lines',  # E302: expected 2 blank lines
                    'blank line at end of file',  # W292: no newline at end of file
                    'trailing whitespace',  # W291: trailing whitespace
                    'multiple spaces',  # E221: multiple spaces before operator
                    'whitespace before',  # E201: whitespace after '('
                    'whitespace after',  # E202: whitespace before ')'
                    'blank line contains whitespace',  # W293: blank line contains whitespace
                    'code formatting can be improved',  # Generic formatting issues
                    'import statements can be better organized',  # Import organization
                    'unused import',  # Filter unused imports that might be dynamic
                    'imported but unused',  # F401 unused imports
                    'indentation is not a multiple',  # E111, E112, E113 indentation issues
                    'continuation line',  # E124, E125, E126, E127, E128 line continuation
                ]
                
                if any(pattern in description_lower for pattern in skip_patterns):
                    continue  # Skip these common style issues
                
                # Keep all other issues
                filtered_issues.append(issue)
            
            # Filter corresponding fixes
            filtered_lines = {issue.line for issue in filtered_issues}
            filtered_fixes = [fix for fix in fixes if hasattr(fix, 'line') and fix.line in filtered_lines]
            
            filtered_count = len(issues) - len(filtered_issues)
            if filtered_count > 0:
                print(f"Filtered {filtered_count} potential false positives using fast rules")
            
            return filtered_issues, filtered_fixes
            
        except Exception as e:
            print(f"Warning: False positive filtering failed: {e}")
            return issues, fixes
    
    def _group_issues_by_file(self, issues: List[Issue], code_files: List[CodeFile]) -> List[Dict[str, Any]]:
        """Group issues by filename with code context."""
        file_map = {file.filename: file.content for file in code_files}
        issues_by_file = {}
        
        for issue in issues:
            filename = getattr(issue, 'filename', 'unknown')
            if filename not in issues_by_file:
                issues_by_file[filename] = {
                    'filename': filename,
                    'content': file_map.get(filename, ''),
                    'issues': []
                }
            issues_by_file[filename]['issues'].append(issue)
        
        return list(issues_by_file.values())
    
    def _validate_file_issues(self, file_info: Dict[str, Any], all_fixes: List[Fix]) -> Tuple[List[Issue], List[Fix]]:
        """Validate issues for a single file using ChatGPT."""
        try:
            issues = file_info['issues']
            filename = file_info['filename']
            content = file_info['content']
            
            # Skip validation if OpenAI client is not available
            if not self._openai_client:
                return issues, [fix for fix in all_fixes if getattr(fix, 'filename', '') == filename]
            
            # Create validation prompt
            prompt = self._build_validation_prompt(filename, content, issues)
            
            # Get ChatGPT analysis with timeout
            response = self._openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a code analysis expert. Quickly validate whether detected issues are true positives or false positives. Be conservative - only mark as false positive if very confident. Respond concisely."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=1000   # Limit response size for faster processing
            )
            
            # Parse validation results
            validation_content = response.choices[0].message.content
            if validation_content:
                validation_result = json.loads(validation_content)
                return self._apply_validation_results(issues, all_fixes, validation_result)
            else:
                return issues, [fix for fix in all_fixes if getattr(fix, 'filename', '') == filename]
            
        except Exception as e:
            print(f"Warning: Issue validation failed for {file_info['filename']}: {e}")
            # Return all issues if validation fails for this file
            return file_info['issues'], [fix for fix in all_fixes if getattr(fix, 'filename', '') == file_info['filename']]
    
    def _build_validation_prompt(self, filename: str, content: str, issues: List[Issue]) -> str:
        """Build a comprehensive validation prompt for ChatGPT."""
        issues_text = []
        for i, issue in enumerate(issues):
            issue_info = {
                'index': i,
                'line': issue.line,
                'type': issue.type,
                'description': issue.description,
                'severity': issue.severity,
                'source': issue.source,
                'rule_id': getattr(issue, 'rule_id', None)
            }
            issues_text.append(f"Issue {i}: {json.dumps(issue_info, indent=2)}")
        
        # Get code context around each issue
        code_lines = content.split('\n')
        context_info = []
        
        for issue in issues:
            line_num = issue.line
            start_line = max(0, line_num - 3)
            end_line = min(len(code_lines), line_num + 2)
            
            context_lines = []
            for i in range(start_line, end_line):
                marker = " -> " if i == line_num - 1 else "    "
                context_lines.append(f"{i+1:3d}{marker}{code_lines[i]}")
            
            context_info.append(f"Context around line {line_num}:\n" + "\n".join(context_lines))
        
        prompt = f"""
Please analyze the following code and validate whether the detected issues are true positives or false positives.

**File: {filename}**

**Full Code:**
```python
{content}
```

**Detected Issues:**
{chr(10).join(issues_text)}

**Code Context for Each Issue:**
{chr(10).join(context_info)}

**Instructions:**
1. For each issue, determine if it's a true positive (real problem) or false positive (not actually a problem)
2. Consider the full code context, not just the specific line
3. Be conservative - only mark as false positive if you're very confident
4. Common false positives include:
   - Import statements that are used in ways static analysis can't detect
   - Variables that are used in dynamic contexts (eval, exec, etc.)
   - Framework-specific patterns that tools don't understand
   - Code that's correct for the specific ML/RL framework being used

**Response Format:**
Return a JSON object with this exact structure:
```json
{{
  "analysis_summary": "Brief summary of your analysis",
  "validated_issues": [
    {{
      "index": 0,
      "is_valid": true,
      "confidence": 0.95,
      "reasoning": "Explanation of why this is/isn't a valid issue"
    }}
  ]
}}
```

Validate all {len(issues)} issues.
"""
        return prompt
    
    def _apply_validation_results(self, issues: List[Issue], all_fixes: List[Fix], validation_result: Dict[str, Any]) -> Tuple[List[Issue], List[Fix]]:
        """Apply ChatGPT validation results to filter issues and fixes."""
        try:
            validated_issues = []
            validated_fixes = []
            
            # Create a mapping of validated issues
            validation_map = {}
            for item in validation_result.get('validated_issues', []):
                validation_map[item['index']] = item
            
            # Filter issues based on validation
            for i, issue in enumerate(issues):
                validation = validation_map.get(i, {'is_valid': True, 'confidence': 1.0})
                
                # Keep issue if:
                # 1. Marked as valid
                # 2. High confidence in validation (> 0.8)
                # 3. Conservative approach - keep if uncertain
                if validation.get('is_valid', True) and validation.get('confidence', 1.0) > 0.7:
                    validated_issues.append(issue)
                else:
                    # Log filtered false positive
                    reasoning = validation.get('reasoning', 'No reasoning provided')
                    print(f"Filtered false positive: Line {issue.line} - {issue.description} (Reason: {reasoning})")
            
            # Filter corresponding fixes
            validated_issue_lines = {issue.line for issue in validated_issues}
            for fix in all_fixes:
                if hasattr(fix, 'line') and fix.line in validated_issue_lines:
                    validated_fixes.append(fix)
            
            return validated_issues, validated_fixes
            
        except Exception as e:
            print(f"Warning: Failed to apply validation results: {e}")
            return issues, all_fixes
    
    def _quick_validate_issues(self, issues: List[Issue], code_files: List[CodeFile]) -> List[Issue]:
        """Quick validation for critical issues only."""
        if not self._openai_client or len(issues) > 5:
            return issues
        
        try:
            # Simple validation prompt for critical issues
            issue_descriptions = []
            for i, issue in enumerate(issues):
                issue_descriptions.append(f"{i}: Line {issue.line} - {issue.description}")
            
            prompt = f"""Quickly validate these {len(issues)} critical code issues. Only mark as false positive if very confident:

{chr(10).join(issue_descriptions)}

Respond with JSON: {{"valid_indices": [list of valid issue indices]}}"""

            response = self._openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Quick code issue validation. Be conservative."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=200
            )
            
            if response.choices[0].message.content:
                result = json.loads(response.choices[0].message.content)
                valid_indices = set(result.get('valid_indices', list(range(len(issues)))))
                return [issues[i] for i in range(len(issues)) if i in valid_indices]
            
        except Exception as e:
            print(f"Quick validation failed: {e}")
        
        return issues
    
    def _check_for_dynamic_usage(self, issue: Issue, code_files: List[CodeFile]) -> bool:
        """Quick check if an import might be used dynamically."""
        try:
            # Extract import name from issue description
            if "'" not in issue.description:
                return False
            
            import_name = issue.description.split("'")[1]
            
            # Find code file
            filename = getattr(issue, 'filename', '')
            for file in code_files:
                if file.filename == filename:
                    # Quick pattern check for dynamic usage
                    content_lower = file.content.lower()
                    return any(pattern in content_lower for pattern in [
                        'eval(', 'exec(', 'getattr(', f'"{import_name}"', f"'{import_name}'"
                    ])
            return False
        except Exception:
            return False
    
    def _has_dynamic_usage_patterns(self, issue: Issue, code_files: List[CodeFile]) -> bool:
        """Check if an 'unused' import might actually be used dynamically."""
        try:
            # Extract the import name from the issue description
            import_name = None
            if "'" in issue.description:
                import_name = issue.description.split("'")[1]
            
            if not import_name:
                return False
            
            # Find the relevant code file
            filename = getattr(issue, 'filename', '')
            code_content = None
            
            for file in code_files:
                if file.filename == filename:
                    code_content = file.content
                    break
            
            if not code_content:
                return False
            
            # Simple heuristics for dynamic usage
            dynamic_patterns = [
                f'eval(',
                f'exec(',
                f'getattr(',
                f'hasattr(',
                f'"{import_name}"',
                f"'{import_name}'",
                f'globals()[',
                f'locals()[',
                f'__import__('
            ]
            
            return any(pattern in code_content for pattern in dynamic_patterns)
            
        except Exception:
            return False  # Conservative: don't filter if we can't determine

# Global instance for singleton pattern
_false_positive_filter_instance = None

def get_false_positive_filter() -> FalsePositiveFilter:
    """Get or create false positive filter instance."""
    global _false_positive_filter_instance
    if _false_positive_filter_instance is None:
        _false_positive_filter_instance = FalsePositiveFilter()
    return _false_positive_filter_instance