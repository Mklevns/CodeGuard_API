"""
AST-based Semantic Analyzer for CodeGuard API.
Provides sophisticated code analysis using Abstract Syntax Trees to reduce false positives
and understand code context semantically rather than through pattern matching.
"""

import ast
import os
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from models import Issue, Fix, CodeFile


@dataclass
class SemanticContext:
    """Context information extracted from AST analysis."""
    function_calls: List[Dict[str, Any]]
    method_calls: List[Dict[str, Any]]
    imports: List[Dict[str, Any]]
    class_definitions: List[Dict[str, Any]]
    variable_assignments: List[Dict[str, Any]]
    loop_structures: List[Dict[str, Any]]
    security_patterns: List[Dict[str, Any]]


class SemanticAnalyzer(ast.NodeVisitor):
    """AST-based semantic analyzer for Python code."""
    
    def __init__(self, source_code: str, filename: str = ""):
        self.source_code = source_code
        self.filename = filename
        self.lines = source_code.splitlines()
        
        # Analysis results
        self.function_calls = []
        self.method_calls = []
        self.imports = []
        self.class_definitions = []
        self.variable_assignments = []
        self.loop_structures = []
        self.security_patterns = []
        
        # Context tracking
        self.current_function = None
        self.current_class = None
        self.loop_depth = 0
        self.in_loop_stack = []
        
        # Parse the AST
        try:
            self.tree = ast.parse(source_code, filename=filename)
        except SyntaxError as e:
            self.tree = None
            self.syntax_error = e
        else:
            self.syntax_error = None
    
    def analyze(self) -> SemanticContext:
        """Perform complete semantic analysis of the code."""
        if self.tree is None:
            return SemanticContext([], [], [], [], [], [], [])
        
        self.visit(self.tree)
        
        return SemanticContext(
            function_calls=self.function_calls,
            method_calls=self.method_calls,
            imports=self.imports,
            class_definitions=self.class_definitions,
            variable_assignments=self.variable_assignments,
            loop_structures=self.loop_structures,
            security_patterns=self.security_patterns
        )
    
    def visit_Import(self, node):
        """Visit import statements."""
        for alias in node.names:
            self.imports.append({
                'type': 'import',
                'module': alias.name,
                'alias': alias.asname,
                'line': node.lineno,
                'col': node.col_offset
            })
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        for alias in node.names:
            self.imports.append({
                'type': 'from_import',
                'module': node.module,
                'name': alias.name,
                'alias': alias.asname,
                'line': node.lineno,
                'col': node.col_offset
            })
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Visit function/method calls."""
        call_info = {
            'line': node.lineno,
            'col': node.col_offset,
            'args_count': len(node.args),
            'kwargs_count': len(node.keywords),
            'in_function': self.current_function,
            'in_class': self.current_class,
            'in_loop': self.loop_depth > 0,
            'loop_depth': self.loop_depth
        }
        
        # Determine if this is a function call or method call
        if isinstance(node.func, ast.Name):
            # Function call: func_name()
            call_info.update({
                'type': 'function',
                'name': node.func.id,
                'full_name': node.func.id
            })
            self.function_calls.append(call_info)
            
            # Check for security patterns
            self._check_security_function_call(node.func.id, node, call_info)
            
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.method()
            call_info.update({
                'type': 'method',
                'method': node.func.attr,
                'object': self._get_attribute_chain(node.func.value),
                'full_name': self._get_full_call_name(node.func)
            })
            self.method_calls.append(call_info)
            
            # Check for RL environment patterns
            self._check_rl_patterns(node.func, node, call_info)
            
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Visit function definitions."""
        old_function = self.current_function
        self.current_function = node.name
        
        # Analyze function
        func_info = {
            'name': node.name,
            'line': node.lineno,
            'args': [arg.arg for arg in node.args.args],
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
            'in_class': self.current_class
        }
        
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_ClassDef(self, node):
        """Visit class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        
        class_info = {
            'name': node.name,
            'line': node.lineno,
            'bases': [self._get_base_name(base) for base in node.bases],
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list]
        }
        self.class_definitions.append(class_info)
        
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_For(self, node):
        """Visit for loops."""
        self._enter_loop('for', node)
        self.generic_visit(node)
        self._exit_loop()
    
    def visit_While(self, node):
        """Visit while loops."""
        self._enter_loop('while', node)
        self.generic_visit(node)
        self._exit_loop()
    
    def visit_Assign(self, node):
        """Visit variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                assign_info = {
                    'variable': target.id,
                    'line': node.lineno,
                    'value_type': type(node.value).__name__,
                    'in_function': self.current_function,
                    'in_class': self.current_class
                }
                self.variable_assignments.append(assign_info)
        
        self.generic_visit(node)
    
    def _enter_loop(self, loop_type: str, node):
        """Enter a loop context."""
        self.loop_depth += 1
        loop_info = {
            'type': loop_type,
            'line': node.lineno,
            'depth': self.loop_depth,
            'in_function': self.current_function,
            'in_class': self.current_class
        }
        self.loop_structures.append(loop_info)
        self.in_loop_stack.append(loop_info)
    
    def _exit_loop(self):
        """Exit a loop context."""
        self.loop_depth -= 1
        if self.in_loop_stack:
            self.in_loop_stack.pop()
    
    def _check_security_function_call(self, func_name: str, node, call_info: Dict):
        """Check for dangerous function calls."""
        dangerous_functions = {
            'eval': 'Code execution vulnerability - avoid eval()',
            'exec': 'Code execution vulnerability - avoid exec()',
            'compile': 'Code compilation - ensure safe usage',
            '__import__': 'Dynamic import - potential security risk'
        }
        
        if func_name in dangerous_functions:
            self.security_patterns.append({
                'type': 'dangerous_function',
                'function': func_name,
                'line': node.lineno,
                'description': dangerous_functions[func_name],
                'severity': 'high',
                'context': call_info
            })
    
    def _check_rl_patterns(self, func, node, call_info: Dict):
        """Check for RL environment usage patterns."""
        if isinstance(func.value, ast.Name):
            obj_name = func.value.id
            method_name = func.attr
            
            # Check for env.step() without env.reset()
            if method_name == 'step' and 'env' in obj_name.lower():
                self._check_env_reset_pattern(obj_name, node, call_info)
            
            # Check for model.eval() vs eval() - this is SAFE
            if method_name == 'eval' and any(keyword in obj_name.lower() 
                                           for keyword in ['model', 'net', 'vae', 'gan']):
                # This is a model.eval() call - safe pattern
                pass
    
    def _check_env_reset_pattern(self, env_name: str, node, call_info: Dict):
        """Check if env.step() is used without proper env.reset()."""
        # Look for env.reset() calls in the same scope or parent scopes
        reset_found = False
        
        # Check if we're in a loop and look for reset patterns
        if call_info['in_loop']:
            # This is a more complex analysis that would require 
            # tracking the control flow - simplified for now
            pass
    
    def _get_attribute_chain(self, node) -> str:
        """Get the full attribute chain for an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attribute_chain(node.value)}.{node.attr}"
        else:
            return "unknown"
    
    def _get_full_call_name(self, node) -> str:
        """Get the full name of a function/method call."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attribute_chain(node.value)}.{node.attr}"
        else:
            return "unknown"
    
    def _get_decorator_name(self, decorator) -> str:
        """Get decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_attribute_chain(decorator.value)}.{decorator.attr}"
        else:
            return "unknown"
    
    def _get_base_name(self, base) -> str:
        """Get base class name."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return self._get_full_call_name(base)
        else:
            return "unknown"


class SemanticFalsePositiveFilter:
    """Filter false positives using semantic analysis."""
    
    def __init__(self):
        self.false_positive_patterns = {
            'eval_method_calls': [
                'model.eval', 'net.eval', 'vae.eval', 'gan.eval', 
                'discriminator.eval', 'generator.eval', 'encoder.eval'
            ],
            'safe_pickle_contexts': [
                'torch.save', 'torch.load', 'joblib.dump', 'joblib.load'
            ]
        }
    
    def filter_issues(self, issues: List[Issue], semantic_context: SemanticContext) -> List[Issue]:
        """Filter false positives using semantic analysis."""
        filtered_issues = []
        
        for issue in issues:
            if not self._is_false_positive(issue, semantic_context):
                filtered_issues.append(issue)
        
        return filtered_issues
    
    def _is_false_positive(self, issue: Issue, context: SemanticContext) -> bool:
        """Check if an issue is a false positive based on semantic context."""
        
        # Check for model.eval() false positives
        if 'eval' in issue.description.lower() and hasattr(issue, 'type') and issue.type in ['F821', 'W292']:
            return self._is_safe_eval_call(issue, context)
        
        # Check for pickle false positives in safe contexts
        if 'pickle' in issue.description.lower():
            return self._is_safe_pickle_usage(issue, context)
        
        # Check for import false positives
        if 'imported but unused' in issue.description.lower():
            return self._is_legitimate_import(issue, context)
        
        return False
    
    def _is_safe_eval_call(self, issue: Issue, context: SemanticContext) -> bool:
        """Check if eval() call is actually a safe model.eval() call."""
        for method_call in context.method_calls:
            if (method_call['line'] == issue.line and 
                method_call['method'] == 'eval' and
                any(keyword in method_call['object'].lower() 
                    for keyword in ['model', 'net', 'vae', 'gan'])):
                return True  # This is a safe model.eval() call
        return False
    
    def _is_safe_pickle_usage(self, issue: Issue, context: SemanticContext) -> bool:
        """Check if pickle usage is in a safe context."""
        # Look for torch.save/torch.load patterns near the pickle usage
        for func_call in context.function_calls:
            if (abs(func_call['line'] - issue.line) <= 2 and
                func_call['name'] in ['save', 'load'] and
                any('torch' in imp['module'] or '' for imp in context.imports
                    if imp['module'] and 'torch' in imp['module'])):
                return True
        return False
    
    def _is_legitimate_import(self, issue: Issue, context: SemanticContext) -> bool:
        """Check if an unused import is actually used in the code."""
        # This would require more sophisticated analysis
        # For now, keep all import issues
        return False


def analyze_code_semantically(code_file: CodeFile) -> Tuple[SemanticContext, List[Issue]]:
    """
    Perform semantic analysis on a code file and generate enhanced issues.
    
    Args:
        code_file: CodeFile object containing filename and content
        
    Returns:
        Tuple of (semantic_context, semantic_issues)
    """
    analyzer = SemanticAnalyzer(code_file.content, code_file.filename)
    context = analyzer.analyze()
    
    # Generate semantic-based issues
    semantic_issues = []
    
    # Check for security issues
    for pattern in context.security_patterns:
        if pattern['type'] == 'dangerous_function':
            issue = Issue(
                filename=code_file.filename,
                line=pattern['line'],
                type='security',
                description=f"Security: {pattern['description']}",
                severity='error',
                source='semantic_analyzer'
            )
            semantic_issues.append(issue)
    
    # Check for missing random seeds in ML code
    if _has_ml_imports(context) and not _has_random_seeding(context):
        issue = Issue(
            filename=code_file.filename,
            line=1,
            type='reproducibility',
            description='Missing random seed for reproducibility in ML code',
            severity='warning',
            source='semantic_analyzer'
        )
        semantic_issues.append(issue)
    
    return context, semantic_issues


def _has_ml_imports(context: SemanticContext) -> bool:
    """Check if code has ML/RL framework imports."""
    ml_modules = ['torch', 'tensorflow', 'sklearn', 'gym', 'numpy']
    return any(any(ml_mod in imp['module'] or '' for ml_mod in ml_modules)
               for imp in context.imports if imp['module'])


def _has_random_seeding(context: SemanticContext) -> bool:
    """Check if code has random seeding calls."""
    seed_functions = ['manual_seed', 'seed', 'set_seed']
    return any(call['name'] in seed_functions for call in context.function_calls)


def get_semantic_analyzer() -> SemanticFalsePositiveFilter:
    """Get semantic analyzer instance."""
    return SemanticFalsePositiveFilter()