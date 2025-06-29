"""
Granular Rule Configuration System for CodeGuard API.
Allows users to configure rules with custom severity levels, enable/disable specific rules,
and set file/directory-specific rule overrides.
"""

import json
import os
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum


class RuleSeverity(Enum):
    """Rule severity levels."""
    IGNORE = "ignore"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class RuleConfig:
    """Configuration for a specific rule."""
    rule_id: str
    enabled: bool = True
    severity: RuleSeverity = RuleSeverity.WARNING
    description: Optional[str] = None
    tags: List[str] = None
    file_patterns: List[str] = None  # Files/patterns where this rule applies
    exclude_patterns: List[str] = None  # Files/patterns to exclude

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.file_patterns is None:
            self.file_patterns = ["*.py"]
        if self.exclude_patterns is None:
            self.exclude_patterns = []


@dataclass
class ProjectRuleConfig:
    """Project-level rule configuration."""
    name: str
    version: str = "1.0"
    default_severity: RuleSeverity = RuleSeverity.WARNING
    global_excludes: List[str] = None
    rules: Dict[str, RuleConfig] = None
    rule_sets: Dict[str, List[str]] = None  # Named rule sets (e.g., "security", "ml_best_practices")

    def __post_init__(self):
        if self.global_excludes is None:
            self.global_excludes = ["__pycache__", "*.pyc", ".git", "node_modules"]
        if self.rules is None:
            self.rules = {}
        if self.rule_sets is None:
            self.rule_sets = {
                "security": ["S001", "S002", "S003"],
                "ml_best_practices": ["ML001", "ML002", "ML003"],
                "rl_patterns": ["RL001", "RL002", "RL003"],
                "performance": ["P001", "P002", "P003"],
                "style": ["E501", "W292", "F401"]
            }


class GranularRuleManager:
    """Manages granular rule configurations."""
    
    def __init__(self, config_file: str = ".codeguard_rules.json"):
        self.config_file = config_file
        self.default_rules = self._get_default_rules()
        self.project_config = self._load_project_config()
    
    def _get_default_rules(self) -> Dict[str, RuleConfig]:
        """Get default rule configurations."""
        return {
            # Security rules
            "S001": RuleConfig(
                rule_id="S001",
                severity=RuleSeverity.CRITICAL,
                description="Dangerous function usage (eval, exec)",
                tags=["security", "dangerous"],
                enabled=True
            ),
            "S002": RuleConfig(
                rule_id="S002", 
                severity=RuleSeverity.ERROR,
                description="Unsafe pickle usage",
                tags=["security", "serialization"],
                enabled=True
            ),
            "S003": RuleConfig(
                rule_id="S003",
                severity=RuleSeverity.WARNING,
                description="Hardcoded credentials",
                tags=["security", "credentials"],
                enabled=True
            ),
            
            # ML/RL rules
            "ML001": RuleConfig(
                rule_id="ML001",
                severity=RuleSeverity.WARNING,
                description="Missing random seed for reproducibility",
                tags=["ml", "reproducibility"],
                enabled=True
            ),
            "ML002": RuleConfig(
                rule_id="ML002",
                severity=RuleSeverity.INFO,
                description="GPU memory management suggestions",
                tags=["ml", "performance", "gpu"],
                enabled=True
            ),
            "RL001": RuleConfig(
                rule_id="RL001",
                severity=RuleSeverity.ERROR,
                description="Missing environment reset in RL loops",
                tags=["rl", "environment"],
                enabled=True
            ),
            "RL002": RuleConfig(
                rule_id="RL002",
                severity=RuleSeverity.WARNING,
                description="Action space mismatch",
                tags=["rl", "environment", "validation"],
                enabled=True
            ),
            "RL003": RuleConfig(
                rule_id="RL003",
                severity=RuleSeverity.INFO,
                description="Episode termination handling",
                tags=["rl", "compatibility"],
                enabled=True
            ),
            
            # Performance rules
            "P001": RuleConfig(
                rule_id="P001",
                severity=RuleSeverity.INFO,
                description="Inefficient loop patterns",
                tags=["performance", "optimization"],
                enabled=True
            ),
            "P002": RuleConfig(
                rule_id="P002",
                severity=RuleSeverity.WARNING,
                description="Memory leak potential",
                tags=["performance", "memory"],
                enabled=True
            ),
            
            # Style rules (can be disabled for specific projects)
            "E501": RuleConfig(
                rule_id="E501",
                severity=RuleSeverity.INFO,
                description="Line too long",
                tags=["style", "formatting"],
                enabled=True
            ),
            "W292": RuleConfig(
                rule_id="W292",
                severity=RuleSeverity.INFO,
                description="No newline at end of file",
                tags=["style", "formatting"],
                enabled=True
            ),
            "F401": RuleConfig(
                rule_id="F401",
                severity=RuleSeverity.INFO,
                description="Imported but unused",
                tags=["style", "imports"],
                enabled=True
            )
        }
    
    def _load_project_config(self) -> ProjectRuleConfig:
        """Load project-specific rule configuration."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Convert rule configs
                rules = {}
                for rule_id, rule_data in config_data.get('rules', {}).items():
                    rule_data['severity'] = RuleSeverity(rule_data.get('severity', 'warning'))
                    rules[rule_id] = RuleConfig(**rule_data)
                
                config_data['rules'] = rules
                config_data['default_severity'] = RuleSeverity(config_data.get('default_severity', 'warning'))
                
                return ProjectRuleConfig(**config_data)
            
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: Invalid rule config file: {e}")
                return self._get_default_project_config()
        
        return self._get_default_project_config()
    
    def _get_default_project_config(self) -> ProjectRuleConfig:
        """Get default project configuration."""
        return ProjectRuleConfig(
            name="default",
            rules=self.default_rules.copy()
        )
    
    def save_config(self):
        """Save current configuration to file."""
        config_dict = asdict(self.project_config)
        
        # Convert enums to strings
        config_dict['default_severity'] = self.project_config.default_severity.value
        for rule_id, rule_config in config_dict['rules'].items():
            rule_config['severity'] = rule_config['severity'].value
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_rule_config(self, rule_id: str) -> Optional[RuleConfig]:
        """Get configuration for a specific rule."""
        return self.project_config.rules.get(rule_id) or self.default_rules.get(rule_id)
    
    def is_rule_enabled(self, rule_id: str, filename: str = "") -> bool:
        """Check if a rule is enabled for a specific file."""
        rule_config = self.get_rule_config(rule_id)
        if not rule_config or not rule_config.enabled:
            return False
        
        # Check global excludes
        for pattern in self.project_config.global_excludes:
            if self._matches_pattern(filename, pattern):
                return False
        
        # Check rule-specific patterns
        if rule_config.exclude_patterns:
            for pattern in rule_config.exclude_patterns:
                if self._matches_pattern(filename, pattern):
                    return False
        
        if rule_config.file_patterns:
            matches_pattern = any(
                self._matches_pattern(filename, pattern) 
                for pattern in rule_config.file_patterns
            )
            if not matches_pattern:
                return False
        
        return True
    
    def get_rule_severity(self, rule_id: str) -> RuleSeverity:
        """Get severity level for a rule."""
        rule_config = self.get_rule_config(rule_id)
        return rule_config.severity if rule_config else self.project_config.default_severity
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches pattern (simple glob-like matching)."""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)
    
    def update_rule(self, rule_id: str, **kwargs):
        """Update rule configuration."""
        current_config = self.get_rule_config(rule_id)
        if current_config:
            # Update existing rule
            for key, value in kwargs.items():
                if hasattr(current_config, key):
                    if key == 'severity' and isinstance(value, str):
                        value = RuleSeverity(value)
                    setattr(current_config, key, value)
            
            self.project_config.rules[rule_id] = current_config
        else:
            # Create new rule
            if 'severity' in kwargs and isinstance(kwargs['severity'], str):
                kwargs['severity'] = RuleSeverity(kwargs['severity'])
            
            new_rule = RuleConfig(rule_id=rule_id, **kwargs)
            self.project_config.rules[rule_id] = new_rule
    
    def disable_rule(self, rule_id: str):
        """Disable a specific rule."""
        self.update_rule(rule_id, enabled=False)
    
    def enable_rule(self, rule_id: str):
        """Enable a specific rule."""
        self.update_rule(rule_id, enabled=True)
    
    def set_rule_severity(self, rule_id: str, severity: str):
        """Set severity for a specific rule."""
        self.update_rule(rule_id, severity=severity)
    
    def get_rules_by_tag(self, tag: str) -> List[RuleConfig]:
        """Get all rules with a specific tag."""
        rules = []
        for rule_config in self.project_config.rules.values():
            if tag in rule_config.tags:
                rules.append(rule_config)
        return rules
    
    def enable_rule_set(self, rule_set_name: str):
        """Enable all rules in a named rule set."""
        if rule_set_name in self.project_config.rule_sets:
            for rule_id in self.project_config.rule_sets[rule_set_name]:
                self.enable_rule(rule_id)
    
    def disable_rule_set(self, rule_set_name: str):
        """Disable all rules in a named rule set."""
        if rule_set_name in self.project_config.rule_sets:
            for rule_id in self.project_config.rule_sets[rule_set_name]:
                self.disable_rule(rule_id)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        enabled_rules = sum(1 for rule in self.project_config.rules.values() if rule.enabled)
        total_rules = len(self.project_config.rules)
        
        by_severity = {}
        for rule in self.project_config.rules.values():
            if rule.enabled:
                severity = rule.severity.value
                by_severity[severity] = by_severity.get(severity, 0) + 1
        
        by_tag = {}
        for rule in self.project_config.rules.values():
            if rule.enabled:
                for tag in rule.tags:
                    by_tag[tag] = by_tag.get(tag, 0) + 1
        
        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "disabled_rules": total_rules - enabled_rules,
            "by_severity": by_severity,
            "by_tag": by_tag,
            "rule_sets": list(self.project_config.rule_sets.keys()),
            "config_file": self.config_file
        }


# Global instance
_rule_manager = None


def get_rule_manager() -> GranularRuleManager:
    """Get or create global rule manager instance."""
    global _rule_manager
    if _rule_manager is None:
        _rule_manager = GranularRuleManager()
    return _rule_manager