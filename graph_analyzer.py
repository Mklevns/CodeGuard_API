"""
Cross-file analysis and call graph generator for CodeGuard API.
Analyzes relationships between files and functions to detect unused code.
"""

import ast
import os
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from models import Issue, Fix, CodeFile


@dataclass
class CodeNode:
    """Represents a code element (function, class, variable) in the call graph."""
    name: str
    node_type: str  # 'function', 'class', 'variable', 'import'
    filename: str
    line: int
    defined_in: str
    called_by: Set[str]
    calls: Set[str]
    imported_by: Set[str]
    imports: Set[str]


class CallGraphBuilder(ast.NodeVisitor):
    """AST visitor to build call graphs from Python code."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.nodes: Dict[str, CodeNode] = {}
        self.current_scope = []
        self.imports = set()
        self.function_calls = set()
        self.class_definitions = set()
        self.function_definitions = set()
        
    def visit_Import(self, node: ast.Import):
        """Handle import statements."""
        for alias in node.names:
            import_name = alias.asname if alias.asname else alias.name
            self.imports.add(import_name)
            self._add_node(import_name, 'import', node.lineno)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Handle from ... import statements."""
        module = node.module or ''
        for alias in node.names:
            import_name = alias.asname if alias.asname else alias.name
            full_name = f"{module}.{import_name}" if module else import_name
            self.imports.add(import_name)
            self._add_node(import_name, 'import', node.lineno)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Handle function definitions."""
        func_name = self._get_qualified_name(node.name)
        self.function_definitions.add(func_name)
        self._add_node(func_name, 'function', node.lineno)
        
        # Enter function scope
        self.current_scope.append(node.name)
        self.generic_visit(node)
        self.current_scope.pop()
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Handle async function definitions."""
        func_name = self._get_qualified_name(node.name)
        self.function_definitions.add(func_name)
        self._add_node(func_name, 'function', node.lineno)
        
        # Enter function scope
        self.current_scope.append(node.name)
        self.generic_visit(node)
        self.current_scope.pop()
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Handle class definitions."""
        class_name = self._get_qualified_name(node.name)
        self.class_definitions.add(class_name)
        self._add_node(class_name, 'class', node.lineno)
        
        # Enter class scope
        self.current_scope.append(node.name)
        self.generic_visit(node)
        self.current_scope.pop()
    
    def visit_Call(self, node: ast.Call):
        """Handle function calls."""
        func_name = self._get_call_name(node.func)
        if func_name:
            self.function_calls.add(func_name)
            current_context = self._get_current_context()
            if current_context:
                self._add_call_relationship(current_context, func_name)
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute):
        """Handle attribute access (method calls)."""
        if isinstance(node.ctx, ast.Load):
            attr_name = self._get_attribute_name(node)
            if attr_name:
                current_context = self._get_current_context()
                if current_context:
                    self._add_call_relationship(current_context, attr_name)
        self.generic_visit(node)
    
    def _get_qualified_name(self, name: str) -> str:
        """Get qualified name based on current scope."""
        if self.current_scope:
            return f"{'.'.join(self.current_scope)}.{name}"
        return name
    
    def _get_current_context(self) -> Optional[str]:
        """Get current function/class context."""
        if self.current_scope:
            return '.'.join(self.current_scope)
        return None
    
    def _get_call_name(self, node: ast.expr) -> Optional[str]:
        """Extract function name from call node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        return None
    
    def _get_attribute_name(self, node: ast.Attribute) -> Optional[str]:
        """Extract attribute name from attribute node."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            base = self._get_attribute_name(node.value)
            return f"{base}.{node.attr}" if base else None
        return node.attr
    
    def _add_node(self, name: str, node_type: str, line: int):
        """Add a node to the call graph."""
        if name not in self.nodes:
            self.nodes[name] = CodeNode(
                name=name,
                node_type=node_type,
                filename=self.filename,
                line=line,
                defined_in=self.filename,
                called_by=set(),
                calls=set(),
                imported_by=set(),
                imports=set()
            )
    
    def _add_call_relationship(self, caller: str, callee: str):
        """Add a call relationship between two nodes."""
        if caller not in self.nodes:
            self._add_node(caller, 'function', 0)
        if callee not in self.nodes:
            self._add_node(callee, 'function', 0)
        
        self.nodes[caller].calls.add(callee)
        self.nodes[callee].called_by.add(caller)


class RepoAnalyzer:
    """Analyzes repository structure and detects unused code."""
    
    def __init__(self, files: List[CodeFile]):
        self.files = files
        self.nodes: Dict[str, CodeNode] = {}
        self.file_dependencies: Dict[str, Set[str]] = {}
        self._build_graph()
    
    def _build_graph(self):
        """Build the complete call graph for all files."""
        # First pass: collect all definitions
        for file in self.files:
            try:
                tree = ast.parse(file.content, filename=file.filename)
                builder = CallGraphBuilder(file.filename)
                builder.visit(tree)
                
                # Merge nodes from this file
                for name, node in builder.nodes.items():
                    full_name = f"{file.filename}:{name}"
                    self.nodes[full_name] = node
                
                # Track file dependencies
                self.file_dependencies[file.filename] = builder.imports
                
            except SyntaxError:
                # Skip files with syntax errors
                continue
        
        # Second pass: resolve cross-file references
        self._resolve_cross_file_references()
    
    def _resolve_cross_file_references(self):
        """Resolve function calls and imports across files."""
        for file in self.files:
            filename = file.filename
            
            # Find all nodes defined in this file
            file_nodes = {k: v for k, v in self.nodes.items() if v.filename == filename}
            
            for node_key, node in file_nodes.items():
                # Resolve calls to other files
                for call in node.calls.copy():
                    # Look for this function in other files
                    for other_key, other_node in self.nodes.items():
                        if (other_node.name == call and 
                            other_node.filename != filename and
                            other_node.node_type in ['function', 'class']):
                            node.calls.add(other_key)
                            other_node.called_by.add(node_key)
    
    def find_unused_code(self) -> List[Dict[str, Any]]:
        """Find unused functions, classes, and variables."""
        unused = []
        
        for node_key, node in self.nodes.items():
            # Skip imports and built-in functions
            if node.node_type == 'import':
                continue
            
            # Check if this node is never called
            if (len(node.called_by) == 0 and 
                node.node_type in ['function', 'class'] and
                not self._is_special_function(node.name)):
                
                unused.append({
                    'name': node.name,
                    'type': node.node_type,
                    'file': node.filename,
                    'line': node.line,
                    'reason': 'Never called or referenced'
                })
        
        return unused
    
    def find_circular_dependencies(self) -> List[Dict[str, Any]]:
        """Find circular dependencies between files."""
        circular_deps = []
        visited = set()
        
        def dfs(filename: str, path: List[str]) -> None:
            if filename in path:
                # Found circular dependency
                cycle_start = path.index(filename)
                cycle = path[cycle_start:] + [filename]
                circular_deps.append({
                    'cycle': cycle,
                    'description': f"Circular dependency: {' -> '.join(cycle)}"
                })
                return
            
            if filename in visited:
                return
            
            visited.add(filename)
            path.append(filename)
            
            # Follow dependencies
            for dep in self.file_dependencies.get(filename, set()):
                # Try to find the dependency file
                dep_file = self._find_dependency_file(dep)
                if dep_file and dep_file != filename:
                    dfs(dep_file, path.copy())
            
            path.pop()
        
        for filename in self.file_dependencies.keys():
            dfs(filename, [])
        
        return circular_deps
    
    def get_complexity_metrics(self) -> Dict[str, Any]:
        """Get complexity metrics for the repository."""
        metrics = {
            'total_files': len(self.files),
            'total_functions': len([n for n in self.nodes.values() if n.node_type == 'function']),
            'total_classes': len([n for n in self.nodes.values() if n.node_type == 'class']),
            'unused_code_count': len(self.find_unused_code()),
            'circular_dependencies': len(self.find_circular_dependencies()),
            'average_calls_per_function': 0,
            'most_called_functions': [],
            'most_complex_files': []
        }
        
        # Calculate average calls per function
        function_nodes = [n for n in self.nodes.values() if n.node_type == 'function']
        if function_nodes:
            total_calls = sum(len(n.called_by) for n in function_nodes)
            metrics['average_calls_per_function'] = total_calls / len(function_nodes)
        
        # Find most called functions
        most_called = sorted(function_nodes, key=lambda n: len(n.called_by), reverse=True)[:5]
        metrics['most_called_functions'] = [
            {'name': n.name, 'file': n.filename, 'call_count': len(n.called_by)}
            for n in most_called
        ]
        
        # Find files with most definitions
        file_complexity = {}
        for node in self.nodes.values():
            if node.filename not in file_complexity:
                file_complexity[node.filename] = 0
            if node.node_type in ['function', 'class']:
                file_complexity[node.filename] += 1
        
        most_complex = sorted(file_complexity.items(), key=lambda x: x[1], reverse=True)[:5]
        metrics['most_complex_files'] = [
            {'file': filename, 'definition_count': count}
            for filename, count in most_complex
        ]
        
        return metrics
    
    def _is_special_function(self, name: str) -> bool:
        """Check if a function is special (main, init, etc.)."""
        special_names = ['main', '__init__', '__main__', 'setup', 'teardown', 'test_']
        return any(special in name.lower() for special in special_names)
    
    def _find_dependency_file(self, dep_name: str) -> Optional[str]:
        """Find the file that provides a dependency."""
        for filename in self.file_dependencies.keys():
            # Simple heuristic: check if module name matches filename
            base_name = os.path.splitext(os.path.basename(filename))[0]
            if base_name == dep_name or dep_name in base_name:
                return filename
        return None


def analyze_repository_structure(files: List[CodeFile]) -> Tuple[List[Issue], List[Fix]]:
    """
    Analyze repository structure and generate issues for unused code and circular dependencies.
    
    Args:
        files: List of code files to analyze
        
    Returns:
        Tuple of issues and fixes found
    """
    issues = []
    fixes = []
    
    try:
        analyzer = RepoAnalyzer(files)
        
        # Find unused code
        unused_code = analyzer.find_unused_code()
        for unused in unused_code:
            issues.append(Issue(
                filename=unused['file'],
                line=unused['line'],
                type="unused_code",
                description=f"Unused {unused['type']} '{unused['name']}': {unused['reason']}",
                source="graph_analyzer",
                severity="warning"
            ))
            
            fixes.append(Fix(
                filename=unused['file'],
                line=unused['line'],
                suggestion=f"Consider removing unused {unused['type']} '{unused['name']}' or add it to __all__ if intentionally public",
                diff=None,
                replacement_code=None,
                auto_fixable=False
            ))
        
        # Find circular dependencies
        circular_deps = analyzer.find_circular_dependencies()
        for circ_dep in circular_deps:
            # Report on the first file in the cycle
            first_file = circ_dep['cycle'][0] if circ_dep['cycle'] else 'unknown'
            issues.append(Issue(
                filename=first_file,
                line=1,
                type="circular_dependency",
                description=circ_dep['description'],
                source="graph_analyzer",
                severity="error"
            ))
            
            fixes.append(Fix(
                filename=first_file,
                line=1,
                suggestion="Refactor to break circular dependency by moving shared code to a common module",
                diff=None,
                replacement_code=None,
                auto_fixable=False
            ))
        
        # Generate repository metrics as informational issues
        metrics = analyzer.get_complexity_metrics()
        if metrics['unused_code_count'] > 0:
            issues.append(Issue(
                filename="repository_summary",
                line=1,
                type="metrics",
                description=f"Repository has {metrics['unused_code_count']} unused code elements across {metrics['total_files']} files",
                source="graph_analyzer",
                severity="info"
            ))
        
    except Exception as e:
        issues.append(Issue(
            filename="repository_analysis",
            line=1,
            type="error",
            description=f"Repository analysis failed: {str(e)}",
            source="graph_analyzer",
            severity="warning"
        ))
    
    return issues, fixes