"""
Custom rule loader for extensible code analysis rules.
Allows loading rules from JSON/YAML files without modifying core code.
"""

import json
import re
import os
from typing import List, Dict, Any, Tuple
from models import Issue, Fix


class CustomRuleEngine:
    """Engine for loading and applying custom analysis rules from external files."""
    
    def __init__(self, rules_dir: str = "rules"):
        self.rules_dir = rules_dir
        self.rules = []
        self.load_all_rules()
    
    def load_all_rules(self):
        """Load all rule files from the rules directory."""
        self.rules = []
        
        if not os.path.exists(self.rules_dir):
            return
            
        for filename in os.listdir(self.rules_dir):
            if filename.endswith('.json'):
                rule_file = os.path.join(self.rules_dir, filename)
                try:
                    self.rules.extend(self.load_rules(rule_file))
                except Exception as e:
                    # Log error but continue with other rules
                    print(f"Warning: Failed to load rules from {filename}: {e}")
    
    def load_rules(self, path: str) -> List[Dict[str, Any]]:
        """Load rules from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def analyze_file(self, filename: str, content: str) -> Tuple[List[Issue], List[Fix]]:
        """Apply all loaded rules to a file and return issues and fixes."""
        issues = []
        fixes = []
        
        for rule in self.rules:
            rule_issues, rule_fixes = self._apply_single_rule(rule, filename, content)
            issues.extend(rule_issues)
            fixes.extend(rule_fixes)
        
        return issues, fixes
    
    def _apply_single_rule(self, rule: Dict[str, Any], filename: str, content: str) -> Tuple[List[Issue], List[Fix]]:
        """Apply a single rule to file content."""
        issues = []
        fixes = []
        
        rule_name = rule.get("name", "unnamed_rule")
        description = rule.get("description", "No description.")
        severity = rule.get("severity", "warning")
        tags = rule.get("tags", [])
        
        lines = content.splitlines()
        
        # Check different rule types
        if self._check_pattern_rule(rule, content):
            # Pattern-based rule (entire file scope)
            issues.append(Issue(
                filename=filename,
                line=1,
                type=self._map_severity_to_type(severity),
                description=f"[{rule_name}] {description}",
                source="custom_rules",
                severity=severity
            ))
            
            # Generate fix if available
            fix = self._generate_rule_fix(rule, filename, 1, content)
            if fix:
                fixes.append(fix)
        
        # Check line-by-line rules
        for line_num, line in enumerate(lines, start=1):
            if self._check_line_rule(rule, line, content):
                issues.append(Issue(
                    filename=filename,
                    line=line_num,
                    type=self._map_severity_to_type(severity),
                    description=f"[{rule_name}] {description}",
                    source="custom_rules",
                    severity=severity
                ))
                
                # Generate fix if available
                fix = self._generate_rule_fix(rule, filename, line_num, line)
                if fix:
                    fixes.append(fix)
                
                # Only report one instance per rule per file for cleaner output
                break
        
        return issues, fixes
    
    def _matches_pattern(self, rule: Dict[str, Any], text: str) -> bool:
        """Check if text matches rule patterns."""
        # Check contains rule
        if "contains" in rule and rule["contains"] in text:
            return True
        
        # Check pattern rule
        if "pattern" in rule and rule["pattern"] in text:
            return True
        
        # Check regex rule
        if "regex" in rule:
            try:
                if re.search(rule["regex"], text):
                    return True
            except re.error:
                # Invalid regex, skip this rule
                pass
        
        return False
    
    def _check_exclusions(self, rule: Dict[str, Any], content: str) -> bool:
        """Check if content matches exclusion patterns."""
        if "not_contains" in rule:
            for exclude in rule["not_contains"]:
                if exclude in content:
                    return True
        return False
    
    def _check_pattern_rule(self, rule: Dict[str, Any], content: str) -> bool:
        """Check if a pattern-based rule matches the entire file content."""
        if not self._matches_pattern(rule, content):
            return False
        
        return not self._check_exclusions(rule, content)
    
    def _check_line_rule(self, rule: Dict[str, Any], line: str, full_content: str) -> bool:
        """Check if a line-based rule matches a specific line."""
        return self._matches_pattern(rule, line)
    
    def _map_severity_to_type(self, severity: str) -> str:
        """Map severity level to issue type."""
        mapping = {
            "error": "error",
            "warning": "best_practice",
            "info": "info",
            "style": "style"
        }
        return mapping.get(severity, "best_practice")
    
    def _generate_rule_fix(self, rule: Dict[str, Any], filename: str, line: int, content: str) -> Fix:
        """Generate fix suggestions based on rule configuration."""
        rule_name = rule.get("name", "")
        
        # Generate specific fixes based on rule type
        if rule_name == "missing_seed":
            return Fix(
                filename=filename,
                line=line,
                suggestion="Add random seeding for reproducibility",
                replacement_code="import random\nimport numpy as np\ntorch.manual_seed(42)\nnp.random.seed(42)\nrandom.seed(42)",
                auto_fixable=True
            )
        
        elif rule_name == "print_logging":
            # Replace print with logging
            if "print(" in content:
                new_content = content.replace("print(", "logger.info(")
                return Fix(
                    filename=filename,
                    line=line,
                    suggestion="Replace print with logging",
                    replacement_code=new_content,
                    auto_fixable=True
                )
        
        elif rule_name == "wildcard_import":
            return Fix(
                filename=filename,
                line=line,
                suggestion="Replace wildcard import with specific imports",
                auto_fixable=False
            )
        
        elif rule_name == "model_save_full":
            return Fix(
                filename=filename,
                line=line,
                suggestion="Save model.state_dict() instead of entire model",
                replacement_code="torch.save(model.state_dict(), path)",
                auto_fixable=True
            )
        
        elif rule_name == "missing_model_eval":
            return Fix(
                filename=filename,
                line=line,
                suggestion="Add model.eval() for validation/testing",
                replacement_code="model.eval()\nwith torch.no_grad():",
                auto_fixable=True
            )
        
        elif rule_name == "cuda_without_cleanup":
            return Fix(
                filename=filename,
                line=line,
                suggestion="Add GPU memory cleanup",
                replacement_code="torch.cuda.empty_cache()",
                auto_fixable=True
            )
        
        elif rule_name == "missing_env_reset":
            return Fix(
                filename=filename,
                line=line,
                suggestion="Add environment reset at start of episode",
                replacement_code="obs = env.reset()",
                auto_fixable=False
            )
        
        # Generic fix for rules without specific implementations
        return Fix(
            filename=filename,
            line=line,
            suggestion=f"Address {rule_name}: {rule.get('description', 'See rule documentation')}",
            auto_fixable=False
        )
    
    def reload_rules(self):
        """Reload all rules from files (useful for development/testing)."""
        self.load_all_rules()
    
    def get_rule_count(self) -> int:
        """Return the number of loaded rules."""
        return len(self.rules)
    
    def get_rules_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Get all rules that contain a specific tag."""
        return [rule for rule in self.rules if tag in rule.get("tags", [])]
    
    def get_rule_summary(self) -> Dict[str, Any]:
        """Get a summary of loaded rules for debugging/monitoring."""
        tags = {}
        severities = {}
        
        for rule in self.rules:
            # Count tags
            for tag in rule.get("tags", []):
                tags[tag] = tags.get(tag, 0) + 1
            
            # Count severities
            severity = rule.get("severity", "unknown")
            severities[severity] = severities.get(severity, 0) + 1
        
        return {
            "total_rules": len(self.rules),
            "tags": tags,
            "severities": severities,
            "rule_names": [rule.get("name", "unnamed") for rule in self.rules]
        }