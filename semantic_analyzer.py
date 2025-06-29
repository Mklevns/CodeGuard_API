"""
Semantic Analyzer for CodeGuard API.
Uses Abstract Syntax Tree (AST) analysis to reduce false positives by understanding code context.
"""

import ast
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from models import Issue, Fix, CodeFile


class SemanticIssueDetector:
    """Detects issues using semantic analysis instead of simple pattern matching."""
    
    def __init__(self):
        # Common ML/DL framework attributes that use 'eval' as method names
        self.safe_eval_contexts = {
            'torch.nn.Module',  # PyTorch models
            'tensorflow.keras.Model',  # TensorFlow models
            'jax.nn.Module',  # JAX models
        }
        
        # Dangerous built-in functions to flag
        self.dangerous_builtins = {
            'eval', 'exec', 'compile', '__import__'
        }
    
    def analyze_file(self, filename: str, content: str) -> Tuple[List[Issue], List[Fix]]:
        """
        Analyze a Python file using AST-based semantic analysis.
        
        Args:
            filename: Name of the file being analyzed
            content: File content as string
            
        Returns:
            Tuple of (issues, fixes) found through semantic analysis
        """
        issues = []
        fixes = []
        
        try:
            # Parse code into AST
            tree = ast.parse(content, filename=filename)
            
            # Create visitor for semantic analysis
            visitor = SemanticVisitor(filename, content)
            visitor.visit(tree)
            
            # Collect results
            issues.extend(visitor.get_issues())
            fixes.extend(visitor.get_fixes())
            
        except SyntaxError as e:
            # Handle syntax errors gracefully
            issues.append(Issue(
                filename=filename,
                line=e.lineno or 1,
                type="syntax_error",
                description=f"Syntax error: {e.msg}",
                severity="error",
                source="semantic_analyzer"
            ))
        except Exception as e:
            print(f"Warning: Semantic analysis failed for {filename}: {e}")
        
        return issues, fixes


class SemanticVisitor(ast.NodeVisitor):
    """AST visitor that performs semantic analysis to detect issues."""
    
    def __init__(self, filename: str, content: str):
        self.filename = filename
        self.content = content
        self.lines = content.split('\n')
        self.issues = []
        self.fixes = []
        
        # Track variable assignments and imports
        self.variable_types = {}  # var_name -> inferred_type
        self.imports = {}  # alias -> module_name
        self.function_scopes = []  # Stack of function names
        self.loop_depth = 0  # Track nested loops for RL environment analysis
        self.has_env_reset = False  # Track if env.reset() is called in current scope
    
    def get_issues(self) -> List[Issue]:
        """Get all detected issues."""
        return self.issues
    
    def get_fixes(self) -> List[Fix]:
        """Get all suggested fixes."""
        return self.fixes
    
    def visit_Call(self, node: ast.Call):
        """Visit function calls to detect dangerous or problematic patterns."""
        # Analyze the type of call
        if isinstance(node.func, ast.Name):
            # Direct function call: func()
            self._analyze_direct_call(node)
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.method()
            self._analyze_method_call(node)
        
        # Continue traversing
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import):
        """Track import statements."""
        for alias in node.names:
            import_name = alias.asname if alias.asname else alias.name
            self.imports[import_name] = alias.name
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track from-import statements."""
        if node.module:
            for alias in node.names:
                import_name = alias.asname if alias.asname else alias.name
                full_name = f"{node.module}.{alias.name}"
                self.imports[import_name] = full_name
        
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        """Track variable assignments to infer types."""
        # Try to infer variable types from assignments
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            inferred_type = self._infer_type_from_value(node.value)
            if inferred_type:
                self.variable_types[var_name] = inferred_type
        
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For):
        """Track for loops for RL environment analysis."""
        self.loop_depth += 1
        
        # Check if this is a training loop pattern
        if self.loop_depth >= 2:  # Nested loops might indicate episode/step structure
            self._check_rl_environment_usage(node)
        
        self.generic_visit(node)
        self.loop_depth -= 1
    
    def visit_While(self, node: ast.While):
        """Track while loops for RL environment analysis."""
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1
    
    def _analyze_direct_call(self, node: ast.Call):
        """Analyze direct function calls like eval(), exec()."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Check for dangerous built-in functions
            if func_name in ['eval', 'exec']:
                self._flag_dangerous_builtin(node, func_name)
            
            # Check for random functions without seeding
            elif func_name in ['randint', 'choice', 'random', 'uniform', 'normal'] and not self._has_seed_set():
                self._flag_missing_seed(node, 'random')
        
        # Check for module-level random calls  
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                method_name = node.func.attr
                
                # Check numpy random without seeding
                if module_name == 'np' and method_name in ['random', 'rand', 'randn', 'randint']:
                    numpy_seed_patterns = ['np.random.seed(', 'numpy.random.seed(']
                    if not any(pattern in self.content for pattern in numpy_seed_patterns):
                        self._flag_missing_seed(node, 'numpy')
                
                # Check torch random without seeding
                elif module_name == 'torch' and method_name in ['rand', 'randn', 'randint']:
                    torch_seed_patterns = ['torch.manual_seed(', 'torch.cuda.manual_seed(']
                    if not any(pattern in self.content for pattern in torch_seed_patterns):
                        self._flag_missing_seed(node, 'torch')
    
    def _analyze_method_call(self, node: ast.Call):
        """Analyze method calls like obj.method()."""
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            
            # Get the object being called
            obj_info = self._get_object_info(node.func.value)
            
            if method_name == 'eval':
                # Check if this is a safe .eval() method call or dangerous eval()
                if self._is_safe_eval_method(obj_info):
                    # This is safe - obj.eval() on a model
                    pass
                else:
                    # Potentially dangerous if obj is unknown
                    self._flag_potentially_dangerous_eval(node, obj_info)
            
            elif method_name == 'reset' and obj_info.get('name') == 'env':
                self.has_env_reset = True
            
            elif method_name == 'step' and obj_info.get('name') == 'env':
                self._check_env_step_usage(node)
    
    def _get_object_info(self, node: ast.AST) -> Dict[str, Any]:
        """Get information about an object from AST node."""
        if isinstance(node, ast.Name):
            var_name = node.id
            return {
                'name': var_name,
                'type': self.variable_types.get(var_name),
                'is_known': var_name in self.variable_types
            }
        elif isinstance(node, ast.Attribute):
            # Handle chained attributes like torch.nn.Module
            return {
                'name': ast.unparse(node) if hasattr(ast, 'unparse') else str(node),
                'type': None,
                'is_known': False
            }
        else:
            return {
                'name': 'unknown',
                'type': None,
                'is_known': False
            }
    
    def _is_safe_eval_method(self, obj_info: Dict[str, Any]) -> bool:
        """Check if .eval() is being called on a safe object (like a PyTorch model)."""
        obj_type = obj_info.get('type', '')
        obj_name = obj_info.get('name', '')
        
        # Check if object type indicates it's a neural network model
        safe_patterns = [
            'torch.nn.Module',
            'nn.Module',
            'keras.Model',
            'tf.keras.Model',
            'Model',
            'Sequential',  # torch.nn.Sequential
            'nn.Sequential'
        ]
        
        if any(pattern in obj_type for pattern in safe_patterns):
            return True
        
        # Check common variable naming patterns for models
        model_patterns = ['model', 'net', 'network', 'vae', 'gan', 'cnn', 'rnn', 'lstm', 'encoder', 'decoder']
        if any(pattern in obj_name.lower() for pattern in model_patterns):
            return True
        
        # Check if the variable was assigned from a known ML framework constructor
        if obj_name in self.variable_types:
            var_type = self.variable_types[obj_name].lower()
            ml_constructors = ['module', 'sequential', 'linear', 'conv', 'lstm', 'gru', 'transformer']
            if any(constructor in var_type for constructor in ml_constructors):
                return True
        
        return False
    
    def _infer_type_from_value(self, value_node: ast.AST) -> Optional[str]:
        """Infer variable type from assignment value."""
        if isinstance(value_node, ast.Call):
            if isinstance(value_node.func, ast.Attribute):
                # Handle torch.nn.Linear(), etc.
                func_name = ast.unparse(value_node.func) if hasattr(ast, 'unparse') else str(value_node.func)
                return func_name
            elif isinstance(value_node.func, ast.Name):
                return value_node.func.id
        
        return None
    
    def _flag_dangerous_builtin(self, node: ast.Call, func_name: str):
        """Flag dangerous built-in function calls."""
        line_num = node.lineno
        
        self.issues.append(Issue(
            filename=self.filename,
            line=line_num,
            type="security",
            description=f"Dangerous use of {func_name}() - can execute arbitrary code",
            severity="error",
            source="semantic_analyzer"
        ))
        
        # Suggest safer alternatives
        if func_name == 'eval':
            fix_suggestion = "Replace eval() with safer alternatives like ast.literal_eval() for literals or specific parsing"
        else:
            fix_suggestion = f"Avoid using {func_name}() - consider safer alternatives"
        
        self.fixes.append(Fix(
            filename=self.filename,
            line=line_num,
            suggestion=fix_suggestion,
            replacement_code=f"# TODO: Replace {func_name}() with safer alternative",
            auto_fixable=False
        ))
    
    def _flag_potentially_dangerous_eval(self, node: ast.Call, obj_info: Dict[str, Any]):
        """Flag potentially dangerous eval calls that aren't clearly safe."""
        if not obj_info.get('is_known'):
            # Object type is unknown - could be dangerous
            line_num = node.lineno
            obj_name = obj_info.get('name', 'unknown')
            
            self.issues.append(Issue(
                filename=self.filename,
                line=line_num,
                type="security",
                description=f"Potentially dangerous {obj_name}.eval() - verify this is a model method, not eval() function",
                severity="warning",
                source="semantic_analyzer"
            ))
    
    def _flag_missing_seed(self, node: ast.Call, seed_type: str):
        """Flag missing random seed setting."""
        line_num = node.lineno
        
        self.issues.append(Issue(
            filename=self.filename,
            line=line_num,
            type="reproducibility",
            description=f"Random {seed_type} operation without seed - may cause non-reproducible results",
            severity="warning",
            source="semantic_analyzer"
        ))
    
    def _has_seed_set(self) -> bool:
        """Check if random seed has been set in the code."""
        seed_patterns = [
            'random.seed(',
            'seed('
        ]
        return any(pattern in self.content for pattern in seed_patterns)
    
    def _has_numpy_seed_set(self) -> bool:
        """Check if numpy random seed has been set."""
        seed_patterns = [
            'np.random.seed(',
            'numpy.random.seed('
        ]
        return any(pattern in self.content for pattern in seed_patterns)
    
    def _has_torch_seed_set(self) -> bool:
        """Check if torch random seed has been set."""
        seed_patterns = [
            'torch.manual_seed(',
            'torch.cuda.manual_seed(',
            'torch.cuda.manual_seed_all('
        ]
        return any(pattern in self.content for pattern in seed_patterns)
    
    def _check_rl_environment_usage(self, node: ast.For):
        """Check for proper RL environment usage patterns."""
        # Look for env.step() calls in nested loops without env.reset()
        step_calls = []
        reset_calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if child.func.attr == 'step':
                    step_calls.append(child)
                elif child.func.attr == 'reset':
                    reset_calls.append(child)
        
        if step_calls and not reset_calls:
            # Found env.step() without env.reset() in this loop structure
            for step_call in step_calls:
                self.issues.append(Issue(
                    filename=self.filename,
                    line=step_call.lineno,
                    type="rl_error",
                    description="env.step() called in loop without corresponding env.reset() - may cause environment state issues",
                    severity="warning",
                    source="semantic_analyzer"
                ))
    
    def _check_env_step_usage(self, node: ast.Call):
        """Check for proper env.step() usage patterns."""
        # This is a placeholder for more sophisticated env.step() analysis
        # Could check for proper unpacking of return values, etc.
        pass


class SemanticFalsePositiveFilter:
    """Filters false positives using semantic analysis results."""
    
    def __init__(self):
        self.analyzer = SemanticIssueDetector()
    
    def filter_with_semantics(self, issues: List[Issue], fixes: List[Fix], code_files: List[CodeFile]) -> Tuple[List[Issue], List[Fix]]:
        """
        Filter false positives using semantic analysis.
        
        Args:
            issues: Issues from lexical analysis
            fixes: Fixes from lexical analysis
            code_files: Original code files
            
        Returns:
            Tuple of (filtered_issues, filtered_fixes)
        """
        # Run semantic analysis on each file
        semantic_results = {}
        
        for file in code_files:
            try:
                semantic_issues, semantic_fixes = self.analyzer.analyze_file(file.filename, file.content)
                semantic_results[file.filename] = {
                    'issues': semantic_issues,
                    'fixes': semantic_fixes
                }
            except Exception as e:
                print(f"Warning: Semantic analysis failed for {file.filename}: {e}")
                semantic_results[file.filename] = {'issues': [], 'fixes': []}
        
        # Filter original issues using semantic context
        filtered_issues = []
        filtered_fixes = []
        
        for issue in issues:
            filename = getattr(issue, 'filename', '')
            
            # Check if this issue should be filtered based on semantic analysis
            if self._should_keep_issue(issue, semantic_results.get(filename, {})):
                filtered_issues.append(issue)
        
        # Add semantic issues
        for file_results in semantic_results.values():
            filtered_issues.extend(file_results['issues'])
            filtered_fixes.extend(file_results['fixes'])
        
        # Filter fixes to match remaining issues
        issue_lines = {(getattr(issue, 'filename', ''), issue.line) for issue in filtered_issues}
        final_fixes = []
        
        for fix in fixes + filtered_fixes:
            fix_key = (getattr(fix, 'filename', ''), fix.line)
            if fix_key in issue_lines:
                final_fixes.append(fix)
        
        return filtered_issues, final_fixes
    
    def _should_keep_issue(self, issue: Issue, semantic_results: Dict) -> bool:
        """Determine if an issue should be kept based on semantic analysis."""
        # If semantic analysis found no issues for this line, it might be a false positive
        semantic_issues = semantic_results.get('issues', [])
        
        # Check if semantic analysis confirmed this as a real issue
        for semantic_issue in semantic_issues:
            if (semantic_issue.line == issue.line and 
                semantic_issue.type == issue.type):
                return True
        
        # Apply specific filtering rules
        description_lower = issue.description.lower()
        
        # Filter eval() false positives - if semantic analysis didn't flag it, it's likely safe
        if 'eval' in description_lower:
            # Check if semantic analysis found a dangerous eval at this line
            for semantic_issue in semantic_issues:
                if (semantic_issue.line == issue.line and 
                    'eval' in semantic_issue.description.lower() and
                    semantic_issue.severity == 'error'):
                    return True
            # No semantic confirmation - likely a false positive
            return False
        
        # Keep other issues by default
        return True


# Global instance for easy access
semantic_filter = SemanticFalsePositiveFilter()


def get_semantic_filter() -> SemanticFalsePositiveFilter:
    """Get the global semantic filter instance."""
    return semantic_filter