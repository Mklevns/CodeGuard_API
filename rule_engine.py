"""
ML/RL-specific static analysis rules for detecting common patterns and issues.
"""

import re
import ast
from typing import List, Tuple, Optional
from models import Issue, Fix
import difflib


class MLRLRuleEngine:
    """Engine for detecting ML/RL-specific code patterns and issues."""
    
    def __init__(self):
        self.rules = [
            self._check_missing_seeding,
            self._check_rl_environment_reset,
            self._check_training_loop_issues,
            self._check_hardcoded_paths,
            self._check_print_instead_of_logging,
            self._check_gpu_memory_management,
            self._check_data_leakage_patterns,
            self._check_model_saving_loading,
        ]
    
    def analyze_file(self, filename: str, content: str) -> Tuple[List[Issue], List[Fix]]:
        """Analyze a file for ML/RL-specific issues."""
        issues = []
        fixes = []
        
        try:
            # Parse the AST for more sophisticated analysis
            tree = ast.parse(content)
            
            # Run all rules
            for rule in self.rules:
                rule_issues, rule_fixes = rule(filename, content, tree)
                issues.extend(rule_issues)
                fixes.extend(rule_fixes)
                
        except SyntaxError:
            # Skip ML/RL analysis if file has syntax errors
            pass
        except Exception:
            # Skip analysis on parsing errors
            pass
            
        return issues, fixes
    
    def _check_missing_seeding(self, filename: str, content: str, tree: ast.AST) -> Tuple[List[Issue], List[Fix]]:
        """Check for missing random seeding in ML/RL code."""
        issues = []
        fixes = []
        
        lines = content.split('\n')
        
        # Check if imports suggest ML/RL usage
        has_torch = 'import torch' in content or 'from torch' in content
        has_numpy = 'import numpy' in content or 'from numpy' in content
        has_random = 'import random' in content or 'from random' in content
        has_tf = 'tensorflow' in content or 'import tf' in content
        
        if not (has_torch or has_numpy or has_tf):
            return issues, fixes
        
        # Check for seeding patterns
        seed_patterns = [
            r'torch\.manual_seed\(',
            r'np\.random\.seed\(',
            r'random\.seed\(',
            r'tf\.random\.set_seed\(',
            r'torch\.cuda\.manual_seed',
        ]
        
        has_seeding = any(re.search(pattern, content) for pattern in seed_patterns)
        
        if not has_seeding:
            issues.append(Issue(
                filename=filename,
                line=1,
                type="best_practice",
                description="Missing random seeding for reproducibility",
                source="ml_rules",
                severity="warning"
            ))
            
            # Generate appropriate seeding code based on imports
            seed_code = []
            if has_torch:
                seed_code.append("torch.manual_seed(42)")
                if "cuda" in content:
                    seed_code.append("torch.cuda.manual_seed(42)")
            if has_numpy:
                seed_code.append("np.random.seed(42)")
            if has_random:
                seed_code.append("random.seed(42)")
            if has_tf:
                seed_code.append("tf.random.set_seed(42)")
            
            if seed_code:
                suggestion = f"Add seeding for reproducibility:\n" + "\n".join(seed_code)
                fixes.append(Fix(
                    filename=filename,
                    line=1,
                    suggestion=suggestion,
                    replacement_code="\n".join(seed_code) + "\n",
                    auto_fixable=True
                ))
        
        return issues, fixes
    
    def _check_rl_environment_reset(self, filename: str, content: str, tree: ast.AST) -> Tuple[List[Issue], List[Fix]]:
        """Check for proper environment reset patterns in RL code."""
        issues = []
        fixes = []
        
        if 'env.reset()' not in content and '.reset()' not in content:
            return issues, fixes
        
        lines = content.split('\n')
        
        # Look for training loops
        for i, line in enumerate(lines):
            if re.search(r'for.*episode|while.*episode|for.*step', line, re.IGNORECASE):
                # Check if env.reset() is called at the beginning of the loop
                loop_start = i
                reset_found = False
                
                # Look ahead a few lines for reset
                for j in range(i + 1, min(i + 5, len(lines))):
                    if 'env.reset()' in lines[j] or '.reset()' in lines[j]:
                        reset_found = True
                        break
                
                if not reset_found:
                    issues.append(Issue(
                        filename=filename,
                        line=i + 1,
                        type="error",
                        description="Missing environment reset at beginning of episode loop",
                        source="ml_rules",
                        severity="error"
                    ))
                    
                    fixes.append(Fix(
                        filename=filename,
                        line=i + 1,
                        suggestion="Add env.reset() at the beginning of the episode loop",
                        replacement_code="    obs = env.reset()",
                        auto_fixable=False
                    ))
        
        return issues, fixes
    
    def _check_training_loop_issues(self, filename: str, content: str, tree: ast.AST) -> Tuple[List[Issue], List[Fix]]:
        """Check for common training loop issues."""
        issues = []
        fixes = []
        
        lines = content.split('\n')
        
        # Check for missing model.eval() when using torch
        if 'import torch' in content or 'from torch' in content:
            has_model_eval = 'model.eval()' in content or '.eval()' in content
            has_model_train = 'model.train()' in content or '.train()' in content
            has_torch_no_grad = 'torch.no_grad()' in content or 'with torch.no_grad()' in content
            
            # Look for validation/test patterns
            validation_patterns = [
                r'valid',
                r'test',
                r'eval',
                r'inference'
            ]
            
            has_validation = any(re.search(pattern, content, re.IGNORECASE) for pattern in validation_patterns)
            
            if has_validation and not (has_model_eval or has_torch_no_grad):
                issues.append(Issue(
                    filename=filename,
                    line=1,
                    type="best_practice",
                    description="Missing model.eval() or torch.no_grad() during validation/testing",
                    source="ml_rules",
                    severity="warning"
                ))
                
                fixes.append(Fix(
                    filename=filename,
                    line=1,
                    suggestion="Add model.eval() and torch.no_grad() for validation/testing",
                    replacement_code="model.eval()\nwith torch.no_grad():",
                    auto_fixable=False
                ))
        
        # Check for potential exploding loss patterns
        for i, line in enumerate(lines):
            if re.search(r'loss.*\*.*1e[0-9]+|loss.*\*.*10\*\*', line):
                issues.append(Issue(
                    filename=filename,
                    line=i + 1,
                    type="best_practice",
                    description="Potential loss scaling issue - check for exploding gradients",
                    source="ml_rules",
                    severity="info"
                ))
        
        return issues, fixes
    
    def _check_hardcoded_paths(self, filename: str, content: str, tree: ast.AST) -> Tuple[List[Issue], List[Fix]]:
        """Check for hardcoded file paths."""
        issues = []
        fixes = []
        
        lines = content.split('\n')
        
        # Patterns for hardcoded paths
        path_patterns = [
            r'["\'][/\\].*[/\\].*["\']',  # Absolute paths
            r'["\']C:\\.*["\']',          # Windows absolute paths
            r'["\'][A-Za-z]:[/\\].*["\']', # Drive letter paths
        ]
        
        for i, line in enumerate(lines):
            for pattern in path_patterns:
                if re.search(pattern, line) and not line.strip().startswith('#'):
                    # Skip common exceptions
                    if any(skip in line for skip in ['/dev/', '/tmp/', '/proc/', 'http://', 'https://']):
                        continue
                        
                    issues.append(Issue(
                        filename=filename,
                        line=i + 1,
                        type="best_practice",
                        description="Hardcoded file path detected - consider using environment variables or config",
                        source="ml_rules",
                        severity="warning"
                    ))
                    
                    fixes.append(Fix(
                        filename=filename,
                        line=i + 1,
                        suggestion="Replace hardcoded path with environment variable or config parameter",
                        auto_fixable=False
                    ))
        
        return issues, fixes
    
    def _check_print_instead_of_logging(self, filename: str, content: str, tree: ast.AST) -> Tuple[List[Issue], List[Fix]]:
        """Check for print statements that should use logging."""
        issues = []
        fixes = []
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if re.search(r'\bprint\s*\(', line) and not line.strip().startswith('#'):
                # Skip debug prints or simple outputs
                if any(skip in line.lower() for skip in ['debug', 'hello', 'test']):
                    continue
                
                issues.append(Issue(
                    filename=filename,
                    line=i + 1,
                    type="best_practice",
                    description="Consider using logging instead of print for production code",
                    source="ml_rules",
                    severity="info"
                ))
                
                # Generate logging replacement
                print_match = re.search(r'print\s*\((.*)\)', line)
                if print_match:
                    print_content = print_match.group(1)
                    new_line = line.replace(f"print({print_content})", f"logger.info({print_content})")
                    
                    fixes.append(Fix(
                        filename=filename,
                        line=i + 1,
                        suggestion="Replace print with logging",
                        replacement_code=new_line,
                        auto_fixable=True
                    ))
        
        return issues, fixes
    
    def _check_gpu_memory_management(self, filename: str, content: str, tree: ast.AST) -> Tuple[List[Issue], List[Fix]]:
        """Check for GPU memory management issues."""
        issues = []
        fixes = []
        
        if 'torch' not in content and 'cuda' not in content:
            return issues, fixes
        
        lines = content.split('\n')
        
        # Check for missing torch.cuda.empty_cache()
        has_cuda_usage = any('cuda' in line for line in lines)
        has_empty_cache = 'empty_cache()' in content
        
        if has_cuda_usage and not has_empty_cache:
            issues.append(Issue(
                filename=filename,
                line=1,
                type="best_practice",
                description="Consider adding torch.cuda.empty_cache() for GPU memory management",
                source="ml_rules",
                severity="info"
            ))
            
            fixes.append(Fix(
                filename=filename,
                line=1,
                suggestion="Add GPU memory cleanup",
                replacement_code="torch.cuda.empty_cache()",
                auto_fixable=True
            ))
        
        return issues, fixes
    
    def _check_data_leakage_patterns(self, filename: str, content: str, tree: ast.AST) -> Tuple[List[Issue], List[Fix]]:
        """Check for potential data leakage patterns."""
        issues = []
        fixes = []
        
        lines = content.split('\n')
        
        # Check for fitting on test data
        for i, line in enumerate(lines):
            if re.search(r'\.fit\(.*test|\.fit\(.*Test', line):
                issues.append(Issue(
                    filename=filename,
                    line=i + 1,
                    type="error",
                    description="Potential data leakage - fitting on test data",
                    source="ml_rules",
                    severity="error"
                ))
        
        return issues, fixes
    
    def _check_model_saving_loading(self, filename: str, content: str, tree: ast.AST) -> Tuple[List[Issue], List[Fix]]:
        """Check for proper model saving/loading patterns."""
        issues = []
        fixes = []
        
        if 'torch.save' not in content and 'save_model' not in content:
            return issues, fixes
        
        lines = content.split('\n')
        
        # Check for saving entire model vs state_dict
        for i, line in enumerate(lines):
            if 'torch.save(model,' in line:
                issues.append(Issue(
                    filename=filename,
                    line=i + 1,
                    type="best_practice",
                    description="Consider saving model.state_dict() instead of entire model for better compatibility",
                    source="ml_rules",
                    severity="info"
                ))
                
                fixes.append(Fix(
                    filename=filename,
                    line=i + 1,
                    suggestion="Save state_dict instead of entire model",
                    replacement_code=line.replace("torch.save(model,", "torch.save(model.state_dict(),"),
                    auto_fixable=True
                ))
        
        return issues, fixes