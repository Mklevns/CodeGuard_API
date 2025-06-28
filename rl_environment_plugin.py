"""
RL Environment Plugin Support for CodeGuard API.
Analyzes RL environments and detects issues with gym environments,
observation/action space mismatches, reward saturation, and reset loops.
"""

import ast
import re
import json
import tempfile
import os
import importlib.util
from typing import Dict, List, Any, Optional, Tuple
from models import Issue, Fix
import yaml


class RLEnvironmentAnalyzer:
    """Analyzes RL environments for common issues and patterns."""
    
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
        
        # Check for various RL-specific issues
        issues.extend(self._check_missing_reset(filename, content, lines))
        issues.extend(self._check_reward_saturation(filename, content, lines))
        issues.extend(self._check_action_space_mismatch(filename, content, lines))
        issues.extend(self._check_observation_handling(filename, content, lines))
        issues.extend(self._check_environment_lifecycle(filename, content, lines))
        issues.extend(self._check_seed_handling(filename, content, lines))
        issues.extend(self._check_episode_termination(filename, content, lines))
        
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
        """Check for potential reward saturation issues."""
        issues = []
        
        for i, line in enumerate(lines):
            # Look for reward clipping or saturation patterns
            if re.search(r'reward\s*=.*np\.clip|reward.*min.*max', line):
                issues.append(Issue(
                    filename=filename,
                    line=i + 1,
                    type="best_practice",
                    description="Reward clipping detected - ensure this doesn't mask important reward signals",
                    source="rl_plugin",
                    severity="info"
                ))
            
            # Check for hardcoded reward values
            if re.search(r'reward\s*=\s*[0-9]+\.?[0-9]*$', line.strip()):
                issues.append(Issue(
                    filename=filename,
                    line=i + 1,
                    type="best_practice",
                    description="Hardcoded reward value - consider making rewards configurable",
                    source="rl_plugin",
                    severity="warning"
                ))
        
        return issues
    
    def _check_action_space_mismatch(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Check for action space mismatches."""
        issues = []
        
        for i, line in enumerate(lines):
            # Look for action sampling without checking action space
            if re.search(r'action\s*=.*random|action\s*=.*choice', line):
                # Check if action_space is referenced nearby
                context_lines = lines[max(0, i-5):min(len(lines), i+5)]
                context = '\n'.join(context_lines)
                
                if 'action_space' not in context:
                    issues.append(Issue(
                        filename=filename,
                        line=i + 1,
                        type="best_practice",
                        description="Random action generation without checking env.action_space - may cause invalid actions",
                        source="rl_plugin",
                        severity="warning"
                    ))
        
        return issues
    
    def _check_observation_handling(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Check for observation handling issues."""
        issues = []
        
        for i, line in enumerate(lines):
            # Look for step calls
            if re.search(r'\.step\s*\(', line):
                # Check if all return values are handled
                if 'obs' not in line and 'observation' not in line:
                    issues.append(Issue(
                        filename=filename,
                        line=i + 1,
                        type="best_practice",
                        description="env.step() call should handle observation return value",
                        source="rl_plugin",
                        severity="info"
                    ))
                
                # Check for proper unpacking
                if not re.search(r'=.*\.step.*,.*,.*,', line):
                    issues.append(Issue(
                        filename=filename,
                        line=i + 1,
                        type="best_practice",
                        description="env.step() should unpack all return values (obs, reward, done, info)",
                        source="rl_plugin",
                        severity="warning"
                    ))
        
        return issues
    
    def _check_environment_lifecycle(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Check for proper environment lifecycle management."""
        issues = []
        
        has_make = 'gym.make' in content or 'env =' in content
        has_close = '.close()' in content
        
        if has_make and not has_close:
            issues.append(Issue(
                filename=filename,
                line=1,
                type="best_practice",
                description="Environment created but never closed - add env.close() for proper cleanup",
                source="rl_plugin",
                severity="info"
            ))
        
        return issues
    
    def _check_seed_handling(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Check for proper seed handling in RL environments."""
        issues = []
        
        has_env = any(re.search(r'gym\.make|env\s*=', line) for line in lines)
        has_env_seed = '.seed(' in content
        has_random_seed = any('seed' in line and ('random' in line or 'np.' in line or 'torch.' in line) 
                             for line in lines)
        
        if has_env and not has_env_seed and not has_random_seed:
            issues.append(Issue(
                filename=filename,
                line=1,
                type="best_practice",
                description="RL environment without seeding - add env.seed() for reproducible experiments",
                source="rl_plugin",
                severity="warning"
            ))
        
        return issues
    
    def _check_episode_termination(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Check for proper episode termination handling."""
        issues = []
        
        for i, line in enumerate(lines):
            # Look for while loops with done condition
            if re.search(r'while.*not\s+done', line):
                # Check if done is properly updated
                loop_end = self._find_loop_end(lines, i)
                loop_body = lines[i+1:loop_end]
                
                step_found = any('.step(' in loop_line for loop_line in loop_body)
                done_update = any('done' in loop_line and '=' in loop_line for loop_line in loop_body)
                
                if step_found and not done_update:
                    issues.append(Issue(
                        filename=filename,
                        line=i + 1,
                        type="error",
                        description="while not done loop without updating done variable - will cause infinite loop",
                        source="rl_plugin",
                        severity="error"
                    ))
        
        return issues
    
    def _find_loop_end(self, lines: List[str], start_line: int) -> int:
        """Find the end of a loop block (simplified)."""
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level:
                return i
        
        return len(lines)
    
    def _generate_rl_fixes(self, filename: str, issues: List[Issue]) -> List[Fix]:
        """Generate fix suggestions for RL-specific issues."""
        fixes = []
        
        for issue in issues:
            if "missing env.reset()" in issue.description:
                fixes.append(Fix(
                    filename=filename,
                    line=issue.line,
                    suggestion="Add environment reset at the beginning of episode",
                    replacement_code="    obs = env.reset()",
                    auto_fixable=False
                ))
            
            elif "env.close()" in issue.description:
                fixes.append(Fix(
                    filename=filename,
                    line=issue.line,
                    suggestion="Add environment cleanup",
                    replacement_code="env.close()",
                    auto_fixable=True
                ))
            
            elif "env.seed()" in issue.description:
                fixes.append(Fix(
                    filename=filename,
                    line=issue.line,
                    suggestion="Add environment seeding for reproducibility",
                    replacement_code="env.seed(42)",
                    auto_fixable=True
                ))
            
            elif "action_space" in issue.description:
                fixes.append(Fix(
                    filename=filename,
                    line=issue.line,
                    suggestion="Sample actions from environment action space",
                    replacement_code="action = env.action_space.sample()",
                    auto_fixable=False
                ))
            
            elif "step() should unpack" in issue.description:
                fixes.append(Fix(
                    filename=filename,
                    line=issue.line,
                    suggestion="Properly unpack env.step() return values",
                    replacement_code="obs, reward, done, info = env.step(action)",
                    auto_fixable=False
                ))
        
        return fixes


class EnvironmentConfigAnalyzer:
    """Analyzes environment configuration files (YAML) for issues."""
    
    def analyze_config(self, filename: str, content: str) -> Tuple[List[Issue], List[Fix]]:
        """Analyze environment configuration for issues."""
        issues = []
        fixes = []
        
        try:
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                return issues, fixes
            
            # Check environment configuration
            issues.extend(self._check_env_config(filename, config))
            issues.extend(self._check_hyperparameters(filename, config))
            issues.extend(self._check_missing_keys(filename, config))
            
        except yaml.YAMLError as e:
            issues.append(Issue(
                filename=filename,
                line=1,
                type="error",
                description=f"Invalid YAML configuration: {str(e)}",
                source="rl_plugin",
                severity="error"
            ))
        
        return issues, fixes
    
    def _check_env_config(self, filename: str, config: Dict) -> List[Issue]:
        """Check environment-specific configuration."""
        issues = []
        
        if 'env_id' in config:
            env_id = config['env_id']
            if not isinstance(env_id, str) or not env_id:
                issues.append(Issue(
                    filename=filename,
                    line=1,
                    type="error",
                    description="env_id must be a non-empty string",
                    source="rl_plugin",
                    severity="error"
                ))
        
        return issues
    
    def _check_hyperparameters(self, filename: str, config: Dict) -> List[Issue]:
        """Check hyperparameter values for common issues."""
        issues = []
        
        # Check learning rate
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if isinstance(lr, (int, float)):
                if lr <= 0:
                    issues.append(Issue(
                        filename=filename,
                        line=1,
                        type="error",
                        description="Learning rate must be positive",
                        source="rl_plugin",
                        severity="error"
                    ))
                elif lr > 1.0:
                    issues.append(Issue(
                        filename=filename,
                        line=1,
                        type="warning",
                        description="Learning rate > 1.0 may cause unstable training",
                        source="rl_plugin",
                        severity="warning"
                    ))
        
        # Check discount factor
        if 'gamma' in config:
            gamma = config['gamma']
            if isinstance(gamma, (int, float)):
                if not (0 <= gamma <= 1):
                    issues.append(Issue(
                        filename=filename,
                        line=1,
                        type="error",
                        description="Discount factor (gamma) must be between 0 and 1",
                        source="rl_plugin",
                        severity="error"
                    ))
        
        return issues
    
    def _check_missing_keys(self, filename: str, config: Dict) -> List[Issue]:
        """Check for missing important configuration keys."""
        issues = []
        
        important_keys = ['env_id', 'total_timesteps', 'algorithm']
        missing_keys = [key for key in important_keys if key not in config]
        
        if missing_keys:
            issues.append(Issue(
                filename=filename,
                line=1,
                type="warning",
                description=f"Missing important configuration keys: {', '.join(missing_keys)}",
                source="rl_plugin",
                severity="warning"
            ))
        
        return issues


# Global RL analyzer instances
rl_env_analyzer = RLEnvironmentAnalyzer()
rl_config_analyzer = EnvironmentConfigAnalyzer()