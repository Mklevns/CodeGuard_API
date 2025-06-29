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
        
        return issues
    
    def _check_action_space_mismatch(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Check for action space mismatches."""
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
        
        return issues
    
    def _check_observation_handling(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Check for observation handling issues."""
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
        
        return issues
    
    def _check_environment_lifecycle(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Check for proper environment lifecycle management."""
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
        
        return issues
    
    def _check_seed_handling(self, filename: str, content: str, lines: List[str]) -> List[Issue]:
        """Check for proper seed handling in RL environments."""
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
        """Check for proper episode termination handling."""
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
    
    def _generate_rl_fixes(self, filename: str, issues: List[Issue]) -> List[Fix]:
        """Generate fixes for RL-specific issues."""
        fixes = []
        
        for issue in issues:
            fix = None
            
            if "missing env.reset()" in issue.description:
                fix = Fix(
                    filename=filename,
                    line_number=issue.line,
                    description="Add environment reset at episode start",
                    original_code="# Missing reset",
                    fixed_code="obs = env.reset()  # Reset environment at episode start",
                    diff_preview="+ obs = env.reset()  # Reset environment at episode start",
                    auto_fixable=True
                )
            
            elif "Environment created but never closed" in issue.description:
                fix = Fix(
                    filename=filename,
                    line_number=issue.line,
                    description="Add environment cleanup",
                    original_code="# Missing cleanup",
                    fixed_code="env.close()  # Clean up environment resources",
                    diff_preview="+ env.close()  # Clean up environment resources",
                    auto_fixable=True
                )
            
            elif "without seeding" in issue.description:
                fix = Fix(
                    filename=filename,
                    line_number=issue.line,
                    description="Add environment seeding for reproducibility",
                    original_code="# Missing seed",
                    fixed_code="env.seed(42)  # Set seed for reproducibility",
                    diff_preview="+ env.seed(42)  # Set seed for reproducibility",
                    auto_fixable=True
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
                import yaml
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                import json
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