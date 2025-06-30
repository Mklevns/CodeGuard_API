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
from typing import List, Dict, Tuple, Optional, Any
from models import AuditRequest, AuditResponse, Issue, Fix
from rule_engine import MLRLRuleEngine
from rule_loader import CustomRuleEngine
from rl_environment_plugin import rl_env_analyzer, rl_config_analyzer
from false_positive_filter import get_false_positive_filter
from semantic_analyzer import analyze_code_semantically, SemanticFalsePositiveFilter
from graph_analyzer import analyze_repository_structure
from git_analyzer import GitContextRetriever
from analysis_cache import get_file_cache, get_project_cache
from pathlib import Path
import io
from contextlib import contextmanager


class EnhancedAuditEngine:
    """Enhanced audit engine with multiple static analysis tools."""

    def __init__(self, use_false_positive_filter: bool = True):
        self.ml_rl_engine = MLRLRuleEngine()
        self.custom_rule_engine = CustomRuleEngine()
        self.use_false_positive_filter = use_false_positive_filter
        self.git_context_retriever = GitContextRetriever()
        self.tools = {
            'flake8': self._run_flake8,
            'pylint': self._run_pylint,
            'mypy': self._run_mypy,
            'black': self._run_black,
            'isort': self._run_isort,
            'ml_rules': self._run_ml_rules,
            'custom_rules': self._run_custom_rules,
            'rl_plugin': self._run_rl_plugin,
            'dependency_audit': self._run_dependency_audit,
            'complexity_analysis': self._run_complexity_analysis,
            'cross_file_analysis': self._run_cross_file_analysis
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

                except json.JSONDecodeError as e:
                    # Log JSON parsing error and skip pylint results
                    issues.append(Issue(
                        filename=original_filename,
                        line=1,
                        type="error",
                        description=f"pylint JSON parsing failed: {str(e)}",
                        source="pylint",
                        severity="warning"
                    ))
                    return issues, fixes

        except subprocess.TimeoutExpired:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="pylint analysis timed out after 60 seconds",
                source="pylint",
                severity="warning"
            ))
        except subprocess.CalledProcessError as e:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description=f"pylint exited with error code {e.returncode}: {e.stderr if e.stderr else 'Unknown error'}",
                source="pylint",
                severity="warning"
            ))
        except PermissionError:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="Permission denied when accessing file for pylint analysis",
                source="pylint",
                severity="warning"
            ))
        except OSError as e:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description=f"OS error during pylint execution: {str(e)}",
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
        except UnicodeDecodeError as e:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description=f"Unicode decode error in pylint output: {str(e)}",
                source="pylint",
                severity="warning"
            ))
        except Exception as e:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description=f"Unexpected pylint error: {type(e).__name__}: {str(e)}",
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

    def _run_dependency_audit(self, file_path: str, original_filename: str, content: str, temp_dir: str) -> Tuple[List[Issue], List[Fix]]:
        """Run dependency and license audit using pip-audit and pip-licenses."""
        issues = []
        fixes = []

        if original_filename not in ["pyproject.toml", "requirements.txt", "setup.py", "Pipfile"]:
            return issues, fixes

        # Vulnerability scanning with pip-audit
        try:
            result = subprocess.run(
                ["pip-audit", "--format=json", "-r", file_path],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir
            )
            if result.stdout:
                try:
                    vulnerabilities = json.loads(result.stdout)
                    for vuln in vulnerabilities.get("vulnerabilities", []):
                        issues.append(Issue(
                            filename=original_filename,
                            line=1,
                            type="security",
                            description=f"Vulnerability found in {vuln['name']} ({vuln['id']}): {vuln['summary']}",
                            source="pip-audit",
                            severity="error"
                        ))

                        # Generate fix suggestion
                        fix_version = vuln.get('fix_versions', [])
                        if fix_version:
                            fix = Fix(
                                filename=original_filename,
                                line=1,
                                suggestion=f"Update {vuln['name']} to version {fix_version[0]} or later",
                                diff=f"- {vuln['name']}=={vuln['installed_version']}\n+ {vuln['name']}>={fix_version[0]}",
                                replacement_code=None,
                                auto_fixable=False
                            )
                            fixes.append(fix)
                except json.JSONDecodeError:
                    pass
        except subprocess.TimeoutExpired:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="Dependency vulnerability scan timed out",
                source="pip-audit",
                severity="warning"
            ))
        except FileNotFoundError:
            # pip-audit not available, skip silently
            pass
        except Exception as e:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description=f"Dependency audit failed: {str(e)}",
                source="pip-audit",
                severity="warning"
            ))

        # License auditing with pip-licenses
        try:
            result = subprocess.run(
                ["pip-licenses", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir
            )
            if result.stdout:
                try:
                    licenses = json.loads(result.stdout)
                    problematic_licenses = ['GPL', 'AGPL', 'UNKNOWN', 'COMMERCIAL']

                    for lib in licenses:
                        license_name = lib.get('License', 'UNKNOWN')
                        if any(prob in license_name.upper() for prob in problematic_licenses):
                            severity = "error" if license_name == 'UNKNOWN' else "warning"
                            issues.append(Issue(
                                filename=original_filename,
                                line=1,
                                type="license",
                                description=f"Problematic license '{license_name}' for {lib['Name']}",
                                source="pip-licenses",
                                severity=severity
                            ))

                            if license_name == 'UNKNOWN':
                                fix = Fix(
                                    filename=original_filename,
                                    line=1,
                                    suggestion=f"Review license for {lib['Name']} - unknown license may cause compliance issues",
                                    diff=None,
                                    replacement_code=None,
                                    auto_fixable=False
                                )
                                fixes.append(fix)
                except json.JSONDecodeError:
                    pass
        except subprocess.TimeoutExpired:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="License audit timed out",
                source="pip-licenses",
                severity="warning"
            ))
        except FileNotFoundError:
            # pip-licenses not available, skip silently
            pass
        except Exception as e:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description=f"License audit failed: {str(e)}",
                source="pip-licenses",
                severity="warning"
            ))

        return issues, fixes

    def _run_complexity_analysis(self, file_path: str, original_filename: str, content: str, temp_dir: str) -> Tuple[List[Issue], List[Fix]]:
        """Analyze code complexity with radon."""
        issues = []
        fixes = []

        if not original_filename.endswith('.py'):
            return issues, fixes

        try:
            import radon.complexity as complexity_mod
            import radon.metrics as metrics_mod

            # Get cyclomatic complexity using direct API
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # Calculate complexity
            complexity_results = complexity_mod.cc_visit(source_code)
            for result in complexity_results:
                if result.complexity > 10:  # High complexity threshold
                    issues.append(Issue(
                        filename=original_filename,
                        line=result.lineno,
                        type="complexity",
                        description=f"{result.name} has high cyclomatic complexity ({result.complexity}). Consider refactoring.",
                        source="radon",
                        severity="warning" if result.complexity <= 15 else "error"
                    ))

                    fix = Fix(
                        filename=original_filename,
                        line=result.lineno,
                        suggestion=f"Refactor {result.name} to reduce complexity from {result.complexity} to below 10",
                        diff=None,
                        replacement_code=None,
                        auto_fixable=False
                    )
                    fixes.append(fix)

            # Get maintainability index
            try:
                mi_result = metrics_mod.mi_visit(source_code, multi=True)
                if mi_result < 20:  # Low maintainability
                    issues.append(Issue(
                        filename=original_filename,
                        line=1,
                        type="maintainability",
                        description=f"Low maintainability index ({mi_result:.1f}). Consider refactoring for better code quality.",
                        source="radon",
                        severity="warning"
                    ))

                    fix = Fix(
                        filename=original_filename,
                        line=1,
                        suggestion="Improve maintainability by reducing complexity, adding documentation, and following best practices",
                        diff=None,
                        replacement_code=None,
                        auto_fixable=False
                    )
                    fixes.append(fix)
            except Exception:
                # Maintainability index calculation failed, skip
                pass

        except ImportError:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description="radon not available for complexity analysis",
                source="radon",
                severity="info"
            ))
        except Exception as e:
            issues.append(Issue(
                filename=original_filename,
                line=1,
                type="error",
                description=f"Complexity analysis failed: {str(e)}",
                source="radon",
                severity="warning"
            ))

        return issues, fixes

    def _run_cross_file_analysis(self, file_path: str, original_filename: str, content: str, temp_dir: str) -> Tuple[List[Issue], List[Fix]]:
        """Run cross-file analysis to detect unused code and circular dependencies."""
        # This method will be called once per file, but we only want to run the analysis once per request
        # We'll use a flag file to ensure this runs only once
        flag_file = os.path.join(temp_dir, ".cross_file_analysis_done")
        if os.path.exists(flag_file):
            return [], []

        try:
            # Create flag file to prevent multiple runs
            with open(flag_file, 'w') as f:
                f.write("done")

            # Collect all Python files in the temp directory
            from models import CodeFile
            code_files = []

            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.py') and file != os.path.basename(flag_file):
                        file_path_inner = os.path.join(root, file)
                        try:
                            with open(file_path_inner, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                            code_files.append(CodeFile(filename=file, content=file_content))
                        except Exception:
                            continue

            if len(code_files) > 1:  # Only run if we have multiple files
                return analyze_repository_structure(code_files)

        except Exception as e:
            return [Issue(
                filename="cross_file_analysis",
                line=1,
                type="error",
                description=f"Cross-file analysis failed: {str(e)}",
                source="graph_analyzer",
                severity="warning"
            )], []

        return [], []

@contextmanager
def in_memory_file_context(filename: str, content: str):
    """Context manager for in-memory file operations."""
    try:
        # Create a temporary file only when absolutely necessary
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            yield temp_file.name
    finally:
        try:
            os.unlink(temp_file.name)
        except (OSError, UnboundLocalError):
            pass

def run_flake8_analysis_in_memory(filename: str, content: str, filter_false_positives: bool = True) -> Dict[str, Any]:
    """Run flake8 analysis on code content without file I/O overhead."""
    import subprocess
    import json

    try:
        # Use stdin for flake8 to avoid file I/O
        process = subprocess.Popen(
            ['flake8', '--format=json', '--stdin-display-name', filename, '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=content)

        if process.returncode == 0:
            return {"issues": [], "summary": "No issues found"}

        # Parse flake8 JSON output
        try:
            flake8_results = json.loads(stdout) if stdout.strip() else []
        except json.JSONDecodeError:
            # Fallback to parsing line-based output
            #flake8_results = parse_flake8_output(stdout, filename)
            return {"issues": [], "summary": "JSONDecodeError"}

        issues = []
        for result in flake8_results:
            issue = {
                "filename": result.get("filename", filename),
                "line": result.get("line_number", 1),
                "column": result.get("column_number", 1),
                "type": categorize_flake8_code(result.get("code", "")),
                "description": f"{result.get('code', 'Unknown')}: {result.get('text', 'No description')}",
                "severity": get_severity_from_code(result.get("code", "")),
                "source": "flake8"
            }
            issues.append(issue)

        # Apply false positive filtering if requested
        #if filter_false_positives:
        #    from false_positive_filter import FalsePositiveFilter
        #    filter_engine = FalsePositiveFilter()
        #    issues = filter_engine.filter_issues(issues)

        return {
            "issues": issues,
            "summary": f"Found {len(issues)} issues in {filename}"
        }

    except Exception as e:
        return {
            "issues": [],
            "summary": f"Analysis failed: {str(e)}",
            "error": str(e)
        }

def run_flake8_analysis(file_path: str, filter_false_positives: bool = True) -> Dict[str, Any]:
    """Run flake8 analysis on a file."""
    import subprocess
    import json

    try:
        result = subprocess.run(
            ["flake8", "--format=json", file_path],
            capture_output=True,
            text=True
        )

        # Parse flake8 JSON output
        try:
            flake8_results = json.loads(result.stdout) if result.stdout.strip() else []
        except json.JSONDecodeError:
            return {"issues": [], "summary": "JSONDecodeError"}

        issues = []
        for result in flake8_results:
            issue = {
                "filename": result.get("filename", file_path),
                "line": result.get("line_number", 1),
                "column": result.get("column_number", 1),
                "type": categorize_flake8_code(result.get("code", "")),
                "description": f"{result.get('code', 'Unknown')}: {result.get('text', 'No description')}",
                "severity": get_severity_from_code(result.get("code", "")),
                "source": "flake8"
            }
            issues.append(issue)

        # Apply false positive filtering if requested
        #if filter_false_positives:
        #    from false_positive_filter import FalsePositiveFilter
        #    filter_engine = FalsePositiveFilter()
        #    issues = filter_engine.filter_issues(issues)

        return {
            "issues": issues,
            "summary": f"Found {len(issues)} issues in {file_path}"
        }

    except Exception as e:
        return {
            "issues": [],
            "summary": f"Analysis failed: {str(e)}",
            "error": str(e)
        }

def categorize_flake8_code(code: str) -> str:
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

def get_severity_from_code(code: str) -> str:
    """Determines issue severity based on flake8 error code."""
    if code.startswith('F'):
        return "error"
    else:
        return "warning"

def audit_code_comprehensive(files: List[Dict[str, str]], options: Dict[str, Any] = None, use_in_memory: bool = True) -> Dict[str, Any]:
    """
    Comprehensive code auditing using multiple static analysis tools.

    Args:
        files (List[Dict[str, str]]): List of files to analyze (filename, content).
        options (Dict[str, Any], optional): Configuration options. Defaults to None.
        use_in_memory (bool): Use in-memory analysis for better performance.

    Returns:
        Dict[str, Any]: Auditing results.
    """
    # Configuration
    filter_false_positives = options.get("filter_false_positives", True) if options else True

    # Results aggregation
    all_issues = []
    summary = ""
    error_count = 0
    warning_count = 0

    # Process each file
    for file_data in files:
        filename = file_data['filename']
        content = file_data['content']

        print(f"Analyzing {filename}...")

        if use_in_memory:
            # Use in-memory analysis for better performance
            file_results = run_flake8_analysis_in_memory(filename, content, filter_false_positives)
        else:
            # Fallback to file-based analysis
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(content)
                temp_file.flush()
                temp_path = temp_file.name

                file_results = run_flake8_analysis(temp_path, filter_false_positives)

                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        if file_results:
            all_issues.extend(file_results["issues"])
            summary += f"{file_results['summary']} | "

            # Count errors and warnings
            for issue in file_results["issues"]:
                if issue["severity"] == "error":
                    error_count += 1
                elif issue["severity"] == "warning":
                    warning_count += 1

    # Overall summary
    issue_count = len(all_issues)
    summary = f"Found {issue_count} issues across {len(files)} files - {summary}"
    if error_count > 0:
        summary = f"{summary} ({error_count} errors, {warning_count} warnings)"

    return {
        "summary": summary,
        "issues": all_issues,
    }