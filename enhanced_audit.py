"""
Enhanced audit engine that combines multiple static analysis tools with ML/RL-specific rules.
"""

import subprocess
import tempfile
import os
import re
import json
import difflib
import uuid
from typing import List, Dict, Tuple, Optional
from models import AuditRequest, AuditResponse, Issue, Fix
from rule_engine import MLRLRuleEngine
from rule_loader import CustomRuleEngine
from rl_environment_plugin import rl_env_analyzer, rl_config_analyzer
from false_positive_filter import get_false_positive_filter
from semantic_analyzer import analyze_code_semantically, SemanticFalsePositiveFilter


class EnhancedAuditEngine:
    """Enhanced audit engine with multiple static analysis tools."""
    
    def __init__(self, use_false_positive_filter: bool = True):
        self.ml_rl_engine = MLRLRuleEngine()
        self.custom_rule_engine = CustomRuleEngine()
        self.use_false_positive_filter = use_false_positive_filter
        self.tools = {
            'flake8': self._run_flake8,
            'pylint': self._run_pylint,
            'mypy': self._run_mypy,
            'black': self._run_black,
            'isort': self._run_isort,
            'ml_rules': self._run_ml_rules,
            'custom_rules': self._run_custom_rules,
            'rl_plugin': self._run_rl_plugin
        }
    
    def analyze_code(self, request: AuditRequest) -> AuditResponse:
        """
        Analyzes Python code files using multiple tools and returns structured results.
        
        Args:
            request: AuditRequest containing files to analyze
            
        Returns:
            AuditResponse with analysis results
        """
        all_issues = []
        all_fixes = []
        
        # Create a unique temporary directory for each request to prevent race conditions
        with tempfile.TemporaryDirectory(prefix=f"codeguard_audit_{os.getpid()}_") as temp_dir:
            # Create a unique subdirectory for this specific audit request
            import uuid
            request_id = str(uuid.uuid4())[:8]
            unique_dir = os.path.join(temp_dir, f"audit_{request_id}")
            os.makedirs(unique_dir, exist_ok=True)
            
            file_paths = []
            
            # Write files to unique temporary directory
            for file in request.files:
                # Sanitize filename to prevent path traversal
                safe_filename = os.path.basename(file.filename)
                if not safe_filename.endswith('.py'):
                    safe_filename += '.py'
                    
                file_path = os.path.join(unique_dir, safe_filename)
                file_paths.append((file_path, file.filename, file.content))
                
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(file.content)
                except Exception as e:
                    all_issues.append(Issue(
                        filename=file.filename,
                        line=1,
                        type="error",
                        description=f"Failed to write file for analysis: {str(e)}",
                        source="system",
                        severity="error"
                    ))
                    continue
            
            # Run all analysis tools
            for tool_name, tool_func in self.tools.items():
                try:
                    for file_path, original_filename, content in file_paths:
                        issues, fixes = tool_func(file_path, original_filename, content, unique_dir)
                        all_issues.extend(issues)
                        all_fixes.extend(fixes)
                except Exception as e:
                    all_issues.append(Issue(
                        filename="system",
                        line=1,
                        type="error",
                        description=f"Analysis tool {tool_name} failed: {str(e)}",
                        source=tool_name,
                        severity="warning"
                    ))
        
        # Apply false positive filtering using ChatGPT if enabled
        if self.use_false_positive_filter:
            false_positive_filter = get_false_positive_filter()
            validated_issues, validated_fixes = false_positive_filter.filter_issues(
                all_issues, all_fixes, request.files
            )
            
            # Add false positive filter status to summary
            filtered_count = len(all_issues) - len(validated_issues)
            filter_status = f" - {filtered_count} potential false positives filtered" if filtered_count > 0 else ""
        else:
            validated_issues, validated_fixes = all_issues, all_fixes
            filter_status = ""
        
        # Generate summary with validated results
        issue_count = len(validated_issues)
        file_count = len(request.files)
        
        # Count issues by severity
        error_count = len([i for i in validated_issues if i.severity == "error"])
        warning_count = len([i for i in validated_issues if i.severity == "warning"])
        
        if error_count > 0:
            summary = f"{issue_count} issues found across {file_count} files ({error_count} errors, {warning_count} warnings)"
        else:
            summary = f"{issue_count} issues found across {file_count} files"
        
        summary += filter_status
        
        return AuditResponse(
            summary=summary,
            issues=validated_issues,
            fixes=validated_fixes
        )
    
    def _run_flake8(self, file_path: str, original_filename: str, content: str, temp_dir: str) -> Tuple[List[Issue], List[Fix]]:
        """Run flake8 analysis on a file."""
        issues = []
        fixes = []
        
        try:
            # Run flake8 with extended configuration
            result = subprocess.run(
                ["flake8", "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s", file_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=temp_dir
            )
            
            # Parse flake8 output
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                # Parse format: filename:line:col: code message
                match = re.match(r'.*:(\d+):(\d+): (\w+) (.+)', line)
                if match:
                    line_num = int(match.group(1))
                    col_num = int(match.group(2))
                    code = match.group(3)
                    description = match.group(4)
                    
                    issue_type = self._categorize_flake8_issue(code)
                    severity = "error" if code.startswith('F') else "warning"
                    
                    issue = Issue(
                        filename=original_filename,
                        line=line_num,
                        type=issue_type,
                        description=f"{code}: {description.strip()}",
                        source="flake8",
                        severity=severity
                    )
                    issues.append(issue)
                    
                    # Generate fix suggestions
                    fix_suggestion, diff, auto_fixable = self._generate_flake8_fix(
                        code, description.strip(), content, line_num
                    )
                    if fix_suggestion:
                        fix = Fix(
                            filename=original_filename,
                            line=line_num,
                            suggestion=fix_suggestion,
                            diff=diff,
                            auto_fixable=auto_fixable
                        )
                        fixes.append(fix)
                        
        except subprocess.TimeoutExpired:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="flake8 analysis timed out",
                source="flake8",
                severity="warning"
            ))
        except FileNotFoundError:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="flake8 not found - tool unavailable",
                source="flake8",
                severity="warning"
            ))
        except Exception as e:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description=f"flake8 analysis failed: {str(e)}",
                source="flake8",
                severity="warning"
            ))
        
        return issues, fixes
    
    def _run_pylint(self, file_path: str, original_filename: str, content: str, temp_dir: str) -> Tuple[List[Issue], List[Fix]]:
        """Run pylint analysis on a file."""
        issues = []
        fixes = []
        
        try:
            result = subprocess.run(
                ["pylint", "--output-format=json", "--disable=C0114,C0115,C0116", file_path],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir
            )
            
            # Parse pylint JSON output
            if result.stdout:
                try:
                    pylint_issues = json.loads(result.stdout)
                    for item in pylint_issues:
                        issue_type = self._categorize_pylint_issue(item.get('type', 'warning'))
                        severity = self._map_pylint_severity(item.get('type', 'warning'))
                        
                        issue = Issue(
                            filename=original_filename,
                            line=item.get('line', 1),
                            type=issue_type,
                            description=f"{item.get('symbol', '')}: {item.get('message', '')}",
                            source="pylint",
                            severity=severity
                        )
                        issues.append(issue)
                        
                        # Generate fix suggestions for common pylint issues
                        fix_suggestion, auto_fixable = self._generate_pylint_fix(
                            item.get('symbol', ''), item.get('message', ''), content, item.get('line', 1)
                        )
                        if fix_suggestion:
                            fix = Fix(
                                filename=original_filename,
                                line=item.get('line', 1),
                                suggestion=fix_suggestion,
                                auto_fixable=auto_fixable
                            )
                            fixes.append(fix)
                            
                except json.JSONDecodeError:
                    # Fallback to text parsing if JSON fails
                    pass
                    
        except subprocess.TimeoutExpired:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="pylint analysis timed out",
                source="pylint",
                severity="warning"
            ))
        except FileNotFoundError:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="pylint not found - tool unavailable",
                source="pylint",
                severity="warning"
            ))
        except Exception as e:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description=f"pylint analysis failed: {str(e)}",
                source="pylint",
                severity="warning"
            ))
        
        return issues, fixes
    
    def _run_mypy(self, file_path: str, original_filename: str, content: str, temp_dir: str) -> Tuple[List[Issue], List[Fix]]:
        """Run mypy type checking on a file."""
        issues = []
        fixes = []
        
        try:
            result = subprocess.run(
                ["mypy", "--ignore-missing-imports", "--no-error-summary", file_path],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir
            )
            
            # Parse mypy output
            for line in result.stdout.strip().split('\n'):
                if not line or 'Success' in line:
                    continue
                    
                # Parse format: filename:line: error: message
                match = re.match(r'.*:(\d+): (error|warning|note): (.+)', line)
                if match:
                    line_num = int(match.group(1))
                    severity = match.group(2)
                    message = match.group(3)
                    
                    issue = Issue(
                        filename=original_filename,
                        line=line_num,
                        type="type_error",
                        description=f"Type check: {message}",
                        source="mypy",
                        severity="error" if severity == "error" else "warning"
                    )
                    issues.append(issue)
                    
                    # Generate type-related fixes
                    fix_suggestion = self._generate_mypy_fix(message, content, line_num)
                    if fix_suggestion:
                        fix = Fix(
                            filename=original_filename,
                            line=line_num,
                            suggestion=fix_suggestion,
                            auto_fixable=False
                        )
                        fixes.append(fix)
                        
        except subprocess.TimeoutExpired:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="mypy analysis timed out",
                source="mypy",
                severity="warning"
            ))
        except FileNotFoundError:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="mypy not found - tool unavailable",
                source="mypy",
                severity="warning"
            ))
        except Exception as e:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description=f"mypy analysis failed: {str(e)}",
                source="mypy",
                severity="warning"
            ))
        
        return issues, fixes
    
    def _run_black(self, file_path: str, original_filename: str, content: str, temp_dir: str) -> Tuple[List[Issue], List[Fix]]:
        """Run black formatting check and generate diffs."""
        issues = []
        fixes = []
        
        try:
            # Check if file needs formatting
            result = subprocess.run(
                ["black", "--check", "--diff", file_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=temp_dir
            )
            
            if result.returncode != 0 and result.stdout:
                # File needs formatting
                issue = Issue(
                    filename=original_filename,
                    line=1,
                    type="style",
                    description="Code formatting can be improved",
                    source="black",
                    severity="info"
                )
                issues.append(issue)
                
                # Get formatted version
                format_result = subprocess.run(
                    ["black", "--code", content],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    input=content
                )
                
                if format_result.returncode == 0:
                    formatted_content = format_result.stdout
                    diff = '\n'.join(difflib.unified_diff(
                        content.splitlines(keepends=True),
                        formatted_content.splitlines(keepends=True),
                        fromfile=f"a/{original_filename}",
                        tofile=f"b/{original_filename}",
                        lineterm=''
                    ))
                    
                    fix = Fix(
                        filename=original_filename,
                        line=1,
                        suggestion="Apply black formatting",
                        diff=diff,
                        replacement_code=formatted_content,
                        auto_fixable=True
                    )
                    fixes.append(fix)
                    
        except subprocess.TimeoutExpired:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="black formatting check timed out",
                source="black",
                severity="warning"
            ))
        except FileNotFoundError:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="black not found - tool unavailable",
                source="black",
                severity="warning"
            ))
        except Exception as e:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description=f"black formatting check failed: {str(e)}",
                source="black",
                severity="warning"
            ))
        
        return issues, fixes
    
    def _run_isort(self, file_path: str, original_filename: str, content: str, temp_dir: str) -> Tuple[List[Issue], List[Fix]]:
        """Run isort import sorting check."""
        issues = []
        fixes = []
        
        try:
            # Check if imports need sorting
            result = subprocess.run(
                ["isort", "--check-only", "--diff", file_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=temp_dir
            )
            
            if result.returncode != 0 and result.stdout:
                # Imports need sorting
                issue = Issue(
                    filename=original_filename,
                    line=1,
                    type="style",
                    description="Import statements can be better organized",
                    source="isort",
                    severity="info"
                )
                issues.append(issue)
                
                # Get sorted version
                sort_result = subprocess.run(
                    ["isort", "--stdout", file_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=temp_dir
                )
                
                if sort_result.returncode == 0:
                    sorted_content = sort_result.stdout
                    diff = '\n'.join(difflib.unified_diff(
                        content.splitlines(keepends=True),
                        sorted_content.splitlines(keepends=True),
                        fromfile=f"a/{original_filename}",
                        tofile=f"b/{original_filename}",
                        lineterm=''
                    ))
                    
                    fix = Fix(
                        filename=original_filename,
                        line=1,
                        suggestion="Sort and organize imports",
                        diff=diff,
                        replacement_code=sorted_content,
                        auto_fixable=True
                    )
                    fixes.append(fix)
                    
        except subprocess.TimeoutExpired:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="isort check timed out",
                source="isort",
                severity="warning"
            ))
        except FileNotFoundError:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="isort not found - tool unavailable",
                source="isort",
                severity="warning"
            ))
        except Exception as e:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description=f"isort check failed: {str(e)}",
                source="isort",
                severity="warning"
            ))
        
        return issues, fixes
    
    def _run_ml_rules(self, file_path: str, original_filename: str, content: str, temp_dir: str) -> Tuple[List[Issue], List[Fix]]:
        """Run ML/RL-specific rule analysis."""
        return self.ml_rl_engine.analyze_file(original_filename, content)
    
    def _run_custom_rules(self, file_path: str, original_filename: str, content: str, temp_dir: str) -> Tuple[List[Issue], List[Fix]]:
        """Run custom rules loaded from external rule files."""
        return self.custom_rule_engine.analyze_file(original_filename, content)
    
    def _run_rl_plugin(self, file_path: str, original_filename: str, content: str, temp_dir: str) -> Tuple[List[Issue], List[Fix]]:
        """Run RL environment plugin analysis."""
        if original_filename.endswith('.yaml') or original_filename.endswith('.yml'):
            return rl_config_analyzer.analyze_config(original_filename, content)
        else:
            return rl_env_analyzer.analyze_environment_code(original_filename, content)
    
    def _categorize_flake8_issue(self, code: str) -> str:
        """Categorizes flake8 error codes into issue types."""
        if code.startswith('F'):
            return "error"
        elif code.startswith('E') or code.startswith('W'):
            return "style"
        elif code.startswith('C'):
            return "complexity"
        elif code.startswith('N'):
            return "naming"
        else:
            return "best_practice"
    
    def _categorize_pylint_issue(self, pylint_type: str) -> str:
        """Categorize pylint issue types."""
        mapping = {
            'error': 'error',
            'warning': 'best_practice',
            'refactor': 'refactor',
            'convention': 'style',
            'info': 'info'
        }
        return mapping.get(pylint_type, 'best_practice')
    
    def _map_pylint_severity(self, pylint_type: str) -> str:
        """Map pylint types to severity levels."""
        mapping = {
            'error': 'error',
            'warning': 'warning',
            'refactor': 'info',
            'convention': 'info',
            'info': 'info'
        }
        return mapping.get(pylint_type, 'warning')
    
    def _generate_flake8_fix(self, code: str, description: str, content: str, line: int) -> Tuple[Optional[str], Optional[str], bool]:
        """Generate fix suggestions for flake8 issues."""
        lines = content.split('\n')
        
        if code == 'F401':  # Unused import
            if line <= len(lines):
                current_line = lines[line - 1]
                suggestion = f"Remove unused import: {current_line.strip()}"
                diff = f"- {current_line}\n+"
                return suggestion, diff, True
                
        elif code in ['E302', 'E303']:  # Expected blank lines
            suggestion = "Add appropriate blank lines"
            return suggestion, None, False
            
        elif code == 'E501':  # Line too long
            suggestion = "Break long line into multiple lines"
            return suggestion, None, False
            
        elif code in ['E225', 'E226']:  # Missing whitespace around operator
            suggestion = "Add whitespace around operators"
            return suggestion, None, True
            
        return None, None, False
    
    def _generate_pylint_fix(self, symbol: str, message: str, content: str, line: int) -> Tuple[Optional[str], bool]:
        """Generate fix suggestions for pylint issues."""
        if symbol == 'unused-variable':
            return "Remove or prefix with underscore if intentionally unused", False
        elif symbol == 'line-too-long':
            return "Break long line into multiple lines", False
        elif symbol == 'missing-docstring':
            return "Add docstring to document the function/class", False
        elif symbol == 'invalid-name':
            return "Use descriptive variable names following naming conventions", False
        elif symbol == 'too-many-locals':
            return "Consider breaking function into smaller functions", False
        elif symbol == 'broad-except':
            return "Catch specific exceptions instead of broad Exception", False
            
        return None, False
    
    def _generate_mypy_fix(self, message: str, content: str, line: int) -> Optional[str]:
        """Generate fix suggestions for mypy type errors."""
        if "has no attribute" in message:
            return "Check object type or add type annotations"
        elif "incompatible types" in message:
            return "Ensure type compatibility or add type casting"
        elif "Cannot determine type" in message:
            return "Add explicit type annotations"
        elif "Missing return statement" in message:
            return "Add return statement or specify return type as None"
            
        return "Review type annotations and ensure type safety"