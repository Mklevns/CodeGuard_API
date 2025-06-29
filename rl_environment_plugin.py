"""
Enhanced RL Environment Plugin Support for CodeGuard API.
Analyzes RL environments and detects issues with gym environments,
observation/action space mismatches, reward saturation, and reset loops.
Implements the expanded analysis capabilities identified in the code review.
"""

import ast
import re
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from models import Issue, Fix
import yaml


class RLEnvironmentAnalyzer:
    """Analyzes RL environments for common issues and patterns with enhanced detection."""
    
    def __init__(self):
        self.gym_patterns = {
            'make_env': r'gym\.make\s*\(\s*["\']([^"\']+)["\']',
            'step_call': r'\.step\s*\(',
            'reset_call': r'\.reset\s*\(',
            'action_space': r'\.action_space',
            'observation_space': r'\.observation_space',
            'render': r'\.render\s*\(',
            'close': r'\.close\s*\(',
            'seed': r'\.seed\s*\('
        }
    
    def analyze_environment_code(self, filename: str, content: str) -> Tuple[List[Issue], List[Fix]]:
        """Analyze RL environment code for common issues."""
        issues = []
        fixes = []
        
        # Check for basic RL patterns
        if not self._contains_rl_patterns(content):
            return issues, fixes
        
        lines = content.splitlines()
        
        # Enhanced analysis based on code review feedback
        issues.extend(self._check_missing_reset(filename, content, lines))
        issues.extend(self._check_reward_saturation(filename, content, lines))
        issues.extend(self._check_action_space_mismatch(filename, content, lines))
        issues.extend(self._check_observation_handling(filename, content, lines))
        issues.extend(self._check_environment_lifecycle(filename, content, lines))
        issues.extend(self._check_seed_handling(filename, content, lines))
        issues.extend(self._check_episode_termination(filename, content, lines))
        issues.extend(self._check_done_flag_handling(filename, content, lines))
        
        # Generate corresponding fixes
        fixes.extend(self._generate_rl_fixes(filename, issues))
        
        return issues, fixes
    
    def _contains_rl_patterns(self, content: str) -> bool:
        """Check if content contains RL-related patterns."""
        rl_indicators = [
            'gym.make', 'env.step', 'env.reset', 'action_space', 'observation_space',
            'reward', 'done', 'info', 'episode', 'agent', 'policy'
        ]
        return any(indicator in content for indicator in rl_indicators)
    
    def _check_missing_reset(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Check for missing environment reset in episode loops."""
        issues = []
        
        for i, line in enumerate(lines):
            # Look for episode loops
            if re.search(r'for.*episode|while.*episode', line, re.IGNORECASE):
                # Check next few lines for reset call
                reset_found = False
                for j in range(i + 1, min(i + 10, len(lines))):
                    if re.search(r'\.reset\s*\(', lines[j]):
                        reset_found = True
                        break
                
                if not reset_found:
                    issues.append(Issue(
                        filename=filename,
                        line=i + 1,
                        type="error",
                        description="RL environment loop missing env.reset() call - can cause state pollution between episodes",
                        source="rl_plugin",
                        severity="error"
                    ))
        
        return issues
    
    def _check_reward_saturation(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Enhanced reward saturation detection."""
        issues = []
        
        for i, line in enumerate(lines):
            # Look for reward clipping or saturation patterns
            if re.search(r'reward\s*=.*np\.clip|reward.*min.*max', line):
                issues.append(Issue(
                    filename=filename,
                    line=i + 1,
                    type="warning",
                    description="Reward clipping detected - ensure reward scaling is appropriate for your RL algorithm",
                    source="rl_plugin",
                    severity="warning"
                ))
            
            # Check for potential reward explosion
            if re.search(r'reward\s*\*=|reward\s*=.*reward\s*\*', line):
                issues.append(Issue(
                    filename=filename,
                    line=i + 1,
                    type="warning", 
                    description="Multiplicative reward modification detected - may cause reward explosion",
                    source="rl_plugin",
                    severity="warning"
                ))
            
            # Check for hardcoded reward values
            if re.search(r'reward\s*=\s*[+-]?\d+\.?\d*\s*$', line.strip()):
                issues.append(Issue(
                    filename=filename,
                    line=i + 1,
                    type="best_practice",
                    description="Hardcoded reward value - consider making rewards configurable",
                    source="rl_plugin",
                    severity="info"
                ))
        
        return issues
    
    def _check_action_space_mismatch(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Enhanced action space mismatch detection."""
        issues = []
        
        for i, line in enumerate(lines):
            # Look for manual action construction without space validation
            if re.search(r'action\s*=\s*\[.*\]|action\s*=.*np\.array', line):
                # Check if action space is referenced nearby
                space_check = False
                for j in range(max(0, i-5), min(len(lines), i+5)):
                    if 'action_space' in lines[j]:
                        space_check = True
                        break
                
                if not space_check:
                    issues.append(Issue(
                        filename=filename,
                        line=i + 1,
                        type="warning",
                        description="Manual action construction without action space validation - may cause environment errors",
                        source="rl_plugin",
                        severity="warning"
                    ))
            
            # Check for action sampling without space validation
            if re.search(r'action\s*=.*random|action\s*=.*choice', line):
                context_lines = lines[max(0, i-5):min(len(lines), i+5)]
                context = '\n'.join(context_lines)
                
                if 'action_space' not in context:
                    issues.append(Issue(
                        filename=filename,
                        line=i + 1,
                        type="warning",
                        description="Random action generation without checking env.action_space - may cause invalid actions",
                        source="rl_plugin",
                        severity="warning"
                    ))
        
        return issues
    
    def _check_observation_handling(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Enhanced observation handling detection."""
        issues = []
        
        for i, line in enumerate(lines):
            # Check for direct observation indexing without shape validation
            if re.search(r'obs\[.*\]|observation\[.*\]', line):
                issues.append(Issue(
                    filename=filename,
                    line=i + 1,
                    type="info",
                    description="Direct observation indexing - ensure observation space shape is validated",
                    source="rl_plugin",
                    severity="info"
                ))
            
            # Check for observation normalization
            if re.search(r'obs\s*/=|observation\s*/=', line):
                issues.append(Issue(
                    filename=filename,
                    line=i + 1,
                    type="info",
                    description="Observation normalization detected - ensure consistent preprocessing",
                    source="rl_plugin",
                    severity="info"
                ))
            
            # Check if step() return values are properly handled
            if re.search(r'\.step\s*\(', line):
                if not re.search(r'obs|observation|reward|done|info', line):
                    issues.append(Issue(
                        filename=filename,
                        line=i + 1,
                        type="warning",
                        description="env.step() call should handle all return values (obs, reward, done, info)",
                        source="rl_plugin",
                        severity="warning"
                    ))
        
        return issues
    
    def _check_environment_lifecycle(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Enhanced environment lifecycle management."""
        issues = []
        has_close = any(re.search(r'\.close\s*\(', line) for line in lines)
        has_env_creation = any(re.search(r'gym\.make|env\s*=', line) for line in lines)
        
        if has_env_creation and not has_close:
            issues.append(Issue(
                filename=filename,
                line=1,
                type="warning",
                description="Environment created but never closed - may cause resource leaks",
                source="rl_plugin",
                severity="warning"
            ))
        
        # Check for proper context manager usage
        has_with_statement = any(re.search(r'with.*env|with.*gym\.make', line) for line in lines)
        if has_env_creation and not has_with_statement and not has_close:
            issues.append(Issue(
                filename=filename,
                line=1,
                type="best_practice",
                description="Consider using context manager (with statement) for automatic environment cleanup",
                source="rl_plugin",
                severity="info"
            ))
        
        return issues
    
    def _check_seed_handling(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Enhanced seed handling detection."""
        issues = []
        
        has_env_seed = any(re.search(r'\.seed\s*\(', line) for line in lines)
        has_env_creation = any(re.search(r'gym\.make', line) for line in lines)
        
        if has_env_creation and not has_env_seed:
            issues.append(Issue(
                filename=filename,
                line=1,
                type="warning",
                description="Environment created without seeding - may affect reproducibility",
                source="rl_plugin",
                severity="warning"
            ))
        
        return issues
    
    def _check_episode_termination(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Enhanced episode termination handling for Gym 0.26+ compatibility."""
        issues = []
        
        for i, line in enumerate(lines):
            # Look for done flag usage
            if re.search(r'if\s+done|while.*not.*done', line):
                # Check if truncated flag is also handled (Gym 0.26+)
                truncated_handled = False
                for j in range(max(0, i-3), min(len(lines), i+3)):
                    if 'truncated' in lines[j]:
                        truncated_handled = True
                        break
                
                if not truncated_handled:
                    issues.append(Issue(
                        filename=filename,
                        line=i + 1,
                        type="compatibility",
                        description="Episode termination only checks 'done' flag - consider handling 'truncated' flag for Gym 0.26+ compatibility",
                        source="rl_plugin",
                        severity="info"
                    ))
        
        return issues
    
    def _check_done_flag_handling(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Check for proper done flag handling patterns."""
        issues = []
        
        for i, line in enumerate(lines):
            # Look for step calls that don't handle done flag
            if re.search(r'\.step\s*\(', line):
                # Check if done is handled in subsequent lines
                done_handled = False
                for j in range(i, min(i + 5, len(lines))):
                    if re.search(r'done|terminated', lines[j], re.IGNORECASE):
                        done_handled = True
                        break
                
                if not done_handled:
                    issues.append(Issue(
                        filename=filename,
                        line=i + 1,
                        type="warning",
                        description="env.step() call should check done/terminated flag for proper episode handling",
                        source="rl_plugin",
                        severity="warning"
                    ))
        
        return issues
    
    def _generate_rl_fixes(self, filename: str, issues: List[Issue]) -> List[Fix]:
        """Generate fixes for RL-specific issues."""
        fixes = []
        
        for issue in issues:
            fix = None
            
            if "missing env.reset()" in issue.description:
                fix = Fix(
                    filename=filename,
                    line=issue.line,
                    suggestion="Add environment reset at episode start",
                    replacement_code="obs = env.reset()  # Reset environment at episode start"
                )
            
            elif "Environment created but never closed" in issue.description:
                fix = Fix(
                    filename=filename,
                    line=issue.line,
                    suggestion="Add environment cleanup",
                    replacement_code="env.close()  # Clean up environment resources"
                )
            
            elif "without seeding" in issue.description:
                fix = Fix(
                    filename=filename,
                    line=issue.line,
                    suggestion="Add environment seeding for reproducibility",
                    replacement_code="env.seed(42)  # Set seed for reproducibility"
                )
            
            elif "should handle all return values" in issue.description:
                fix = Fix(
                    filename=filename,
                    line=issue.line,
                    suggestion="Handle all step() return values",
                    replacement_code="obs, reward, done, info = env.step(action)"
                )
            
            if fix:
                fixes.append(fix)
        
        return fixes


def rl_env_analyzer(filename: str, content: str) -> Tuple[List[Issue], List[Fix]]:
    """Main function to analyze RL environment code."""
    analyzer = RLEnvironmentAnalyzer()
    return analyzer.analyze_environment_code(filename, content)


def rl_config_analyzer(config_path: str) -> Tuple[List[Issue], List[Fix]]:
    """Analyze RL configuration files (YAML, JSON)."""
    issues = []
    fixes = []
    
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config = json.load(f)
            else:
                return issues, fixes
        
        # Check for common RL hyperparameter issues
        if isinstance(config, dict):
            # Check for missing critical hyperparameters
            rl_params = ['learning_rate', 'batch_size', 'gamma', 'epsilon']
            missing_params = [param for param in rl_params if param not in config]
            
            if missing_params:
                issues.append(Issue(
                    filename=config_path,
                    line=1,
                    type="configuration",
                    description=f"Missing RL hyperparameters: {', '.join(missing_params)}",
                    source="rl_plugin",
                    severity="warning"
                ))
            
            # Check for potentially problematic values
            if 'learning_rate' in config and config['learning_rate'] > 0.1:
                issues.append(Issue(
                    filename=config_path,
                    line=1,
                    type="configuration",
                    description="Learning rate > 0.1 may be too high - consider reducing",
                    source="rl_plugin", 
                    severity="warning"
                ))
            
            # Check for missing discount factor
            if 'gamma' in config and (config['gamma'] <= 0 or config['gamma'] > 1):
                issues.append(Issue(
                    filename=config_path,
                    line=1,
                    type="configuration",
                    description="Gamma (discount factor) should be between 0 and 1",
                    source="rl_plugin",
                    severity="error"
                ))
    
    except Exception as e:
        issues.append(Issue(
            filename=config_path,
            line=1,
            type="error",
            description=f"Failed to analyze configuration file: {str(e)}",
            source="rl_plugin",
            severity="warning"
        ))
    
    return issues, fixes