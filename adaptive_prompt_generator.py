"""
Adaptive Prompt Generator for CodeGuard API.
Creates custom system prompts based on audit results to provide targeted AI improvements.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter
import json

from models import Issue, Fix, CodeFile


@dataclass
class PromptTemplate:
    """Template for generating AI prompts based on audit patterns."""
    name: str
    trigger_conditions: List[str]  # Issue types or patterns that trigger this template
    system_prompt: str
    improvement_focus: List[str]  # Areas of focus for this prompt
    confidence_boost: float = 0.1  # How much this specialized prompt improves confidence


class AdaptivePromptGenerator:
    """Generates custom system prompts based on audit results and code patterns."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.framework_patterns = {
            'pytorch': ['torch', 'nn.Module', 'torch.tensor', 'cuda'],
            'tensorflow': ['tf.', 'keras', 'tensorflow'],
            'sklearn': ['sklearn', 'fit(', 'predict('],
            'gym': ['gym.make', 'env.step', 'env.reset'],
            'stable_baselines': ['stable_baselines', 'PPO', 'DQN'],
            'jax': ['jax.', 'flax', 'optax'],
            'numpy': ['np.', 'numpy', 'ndarray'],
            'pandas': ['pd.', 'DataFrame', 'pandas']
        }
    
    def _initialize_templates(self) -> List[PromptTemplate]:
        """Initialize specialized prompt templates for different issue patterns."""
        return [
            PromptTemplate(
                name="Security-Focused",
                trigger_conditions=["security", "pickle", "eval", "exec", "subprocess"],
                system_prompt="""You are a cybersecurity-focused code improvement specialist. Your primary goal is to identify and fix security vulnerabilities while maintaining code functionality.

SECURITY PRIORITIES:
1. Replace dangerous functions (eval, exec, pickle.loads) with safe alternatives
2. Sanitize user inputs and validate data
3. Use secure random number generation
4. Implement proper error handling to prevent information leakage
5. Add input validation and bounds checking

When fixing security issues:
- Always provide safe alternative implementations
- Explain why the original code was dangerous
- Ensure the fix maintains the same functionality
- Add comments explaining the security improvement""",
                improvement_focus=["security", "input_validation", "safe_alternatives"],
                confidence_boost=0.15
            ),
            
            PromptTemplate(
                name="ML-Reproducibility",
                trigger_conditions=["ml", "reproducibility", "random", "seed"],
                system_prompt="""You are a machine learning reproducibility expert. Your goal is to make ML code deterministic and reproducible across different runs and environments.

REPRODUCIBILITY PRIORITIES:
1. Add proper random seeding for all random number generators (torch, numpy, random, tf)
2. Set deterministic algorithms where available
3. Handle GPU non-determinism appropriately
4. Add environment and version logging
5. Ensure consistent data loading and preprocessing

For random seeding, use this pattern:
```python
import random
import numpy as np
import torch

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Always explain why reproducibility matters in ML experiments.""",
                improvement_focus=["reproducibility", "seeding", "determinism"],
                confidence_boost=0.12
            ),
            
            PromptTemplate(
                name="RL-Environment",
                trigger_conditions=["rl", "reinforcement", "gym", "environment", "reset"],
                system_prompt="""You are a reinforcement learning expert specializing in proper environment handling and training loops.

RL BEST PRACTICES:
1. Always reset environments at episode boundaries
2. Handle done flags correctly in training loops
3. Properly manage observation and action spaces
4. Implement proper reward normalization and clipping
5. Add environment seeding for reproducibility
6. Handle terminal vs timeout done conditions

For environment resets:
```python
# Proper RL training loop structure
obs = env.reset()  # Always reset at start
for episode in range(num_episodes):
    done = False
    while not done:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        # Store transition
        obs = next_obs
    
    # Reset for next episode
    obs = env.reset()
```

Focus on environment consistency and proper episode handling.""",
                improvement_focus=["environment_handling", "episode_management", "rl_loops"],
                confidence_boost=0.14
            ),
            
            PromptTemplate(
                name="Code-Quality",
                trigger_conditions=["style", "quality", "formatting", "imports"],
                system_prompt="""You are a Python code quality specialist focused on clean, maintainable code that follows PEP 8 and best practices.

CODE QUALITY PRIORITIES:
1. Remove unused imports and variables
2. Fix line length issues by breaking long lines appropriately
3. Improve function and variable naming
4. Add proper docstrings and type hints
5. Organize imports properly (stdlib, third-party, local)
6. Fix indentation and spacing issues

For import organization:
```python
# Standard library imports
import os
import sys

# Third-party imports
import numpy as np
import torch

# Local imports
from .models import MyModel
```

Always preserve the original functionality while improving readability.""",
                improvement_focus=["code_style", "imports", "formatting"],
                confidence_boost=0.08
            ),
            
            PromptTemplate(
                name="Performance-Optimization",
                trigger_conditions=["performance", "memory", "gpu", "optimization"],
                system_prompt="""You are a performance optimization expert specializing in Python and ML code efficiency.

PERFORMANCE PRIORITIES:
1. Optimize memory usage and prevent memory leaks
2. Improve GPU utilization and batch processing
3. Use vectorized operations instead of loops
4. Implement proper data loading and caching
5. Add memory profiling and performance monitoring
6. Optimize model inference and training speed

For GPU optimization:
```python
# Efficient GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device, non_blocking=True)

# Memory management
with torch.no_grad():  # For inference
    output = model(data)
    
torch.cuda.empty_cache()  # Clear unused memory
```

Focus on measurable performance improvements.""",
                improvement_focus=["performance", "memory", "gpu_usage"],
                confidence_boost=0.11
            ),
            
            PromptTemplate(
                name="Error-Handling",
                trigger_conditions=["error", "exception", "try", "except"],
                system_prompt="""You are an error handling and robustness expert focused on making code resilient to failures.

ERROR HANDLING PRIORITIES:
1. Add comprehensive try-catch blocks for risky operations
2. Implement graceful degradation for non-critical failures
3. Add proper logging for debugging
4. Validate inputs and handle edge cases
5. Provide meaningful error messages
6. Add timeout handling for external calls

For robust error handling:
```python
import logging

try:
    result = risky_operation()
except SpecificException as e:
    logging.error(f"Operation failed: {e}")
    # Handle gracefully or re-raise
    raise
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    # Provide fallback or cleanup
    return default_value
```

Always make failures visible but not catastrophic.""",
                improvement_focus=["error_handling", "robustness", "logging"],
                confidence_boost=0.10
            )
        ]
    
    def analyze_audit_results(self, issues: List[Issue], fixes: List[Fix], 
                            code_files: List[CodeFile]) -> Dict[str, Any]:
        """Analyze audit results to understand code patterns and issues."""
        analysis = {
            'issue_types': Counter(),
            'severity_levels': Counter(),
            'frameworks_detected': set(),
            'file_types': Counter(),
            'common_patterns': [],
            'complexity_indicators': []
        }
        
        # Analyze issues
        for issue in issues:
            analysis['issue_types'][issue.type] += 1
            analysis['severity_levels'][issue.severity] += 1
            
            # Check for specific patterns in issue descriptions
            desc_lower = issue.description.lower()
            if any(pattern in desc_lower for pattern in ['pickle', 'eval', 'exec']):
                analysis['common_patterns'].append('security_risk')
            if 'seed' in desc_lower or 'random' in desc_lower:
                analysis['common_patterns'].append('reproducibility_issue')
            if 'reset' in desc_lower or 'environment' in desc_lower:
                analysis['common_patterns'].append('rl_environment_issue')
        
        # Detect frameworks from code content
        all_code = ' '.join(file.content for file in code_files)
        for framework, patterns in self.framework_patterns.items():
            if any(pattern in all_code for pattern in patterns):
                analysis['frameworks_detected'].add(framework)
        
        # Analyze file types
        for file in code_files:
            if file.filename.endswith('.py'):
                analysis['file_types']['python'] += 1
            elif file.filename.endswith('.yaml') or file.filename.endswith('.yml'):
                analysis['file_types']['config'] += 1
        
        return analysis
    
    def select_optimal_templates(self, analysis: Dict[str, Any]) -> List[PromptTemplate]:
        """Select the most appropriate prompt templates based on analysis."""
        selected_templates = []
        
        # Check each template's trigger conditions
        for template in self.templates:
            relevance_score = 0
            
            # Check issue type matches
            for condition in template.trigger_conditions:
                if condition in analysis['issue_types']:
                    relevance_score += analysis['issue_types'][condition]
                
                # Check patterns
                if condition in analysis['common_patterns']:
                    relevance_score += 2
                
                # Check framework matches
                if condition in analysis['frameworks_detected']:
                    relevance_score += 1
            
            if relevance_score > 0:
                selected_templates.append((template, relevance_score))
        
        # Sort by relevance and return top templates
        selected_templates.sort(key=lambda x: x[1], reverse=True)
        return [template for template, score in selected_templates[:3]]  # Max 3 templates
    
    def generate_custom_prompt(self, issues: List[Issue], fixes: List[Fix], 
                             code_files: List[CodeFile], ai_provider: str = "openai") -> Dict[str, Any]:
        """Generate a custom system prompt based on audit results."""
        
        # Analyze the audit results
        analysis = self.analyze_audit_results(issues, fixes, code_files)
        
        # Select optimal templates
        optimal_templates = self.select_optimal_templates(analysis)
        
        if not optimal_templates:
            # Fallback to general template
            return self._generate_general_prompt(analysis, ai_provider)
        
        # Build custom prompt by combining templates
        custom_prompt = self._build_combined_prompt(optimal_templates, analysis, ai_provider)
        
        # Calculate confidence boost
        confidence_boost = sum(template.confidence_boost for template in optimal_templates)
        
        return {
            'system_prompt': custom_prompt,
            'templates_used': [t.name for t in optimal_templates],
            'analysis': analysis,
            'confidence_boost': min(confidence_boost, 0.3),  # Cap at 30% boost
            'improvement_focus': list(set(
                focus for template in optimal_templates 
                for focus in template.improvement_focus
            ))
        }
    
    def _build_combined_prompt(self, templates: List[PromptTemplate], 
                             analysis: Dict[str, Any], ai_provider: str) -> str:
        """Build a combined system prompt from multiple templates."""
        
        # Base prompt
        base_prompt = f"""You are an expert code improvement specialist with deep knowledge of Python, machine learning, and software engineering best practices.

DETECTED CONTEXT:
- Primary issue types: {', '.join(analysis['issue_types'].most_common(3))}
- Frameworks detected: {', '.join(analysis['frameworks_detected'])}
- Total issues to address: {sum(analysis['issue_types'].values())}

"""
        
        # Add specialized instructions from selected templates
        for i, template in enumerate(templates, 1):
            base_prompt += f"\n=== SPECIALIZATION {i}: {template.name.upper()} ===\n"
            base_prompt += template.system_prompt + "\n"
        
        # Add provider-specific instructions
        if ai_provider.lower() == "deepseek":
            base_prompt += """
=== DEEPSEEK SPECIFIC INSTRUCTIONS ===
Return your response as a valid JSON object with these exact keys:
- "improved_code": The complete improved code
- "applied_fixes": Array of fix descriptions
- "improvement_summary": Summary of changes made
- "confidence_score": Number between 0.0 and 1.0
- "warnings": Array of any warnings or notes

Focus on implementing the most critical fixes first, then style improvements.
"""
        
        # Add general improvement guidelines
        base_prompt += """
=== GENERAL IMPROVEMENT GUIDELINES ===
1. Preserve the original functionality and intent of the code
2. Apply fixes in order of importance: security > errors > performance > style
3. Add helpful comments explaining complex changes
4. Ensure all imports are necessary and properly organized
5. Test that your changes don't break existing functionality
6. Provide clear explanations for each improvement made

Remember: Your goal is to make the code better while keeping it functionally identical.
"""
        
        return base_prompt
    
    def _generate_general_prompt(self, analysis: Dict[str, Any], ai_provider: str) -> Dict[str, Any]:
        """Generate a general-purpose prompt when no specific templates match."""
        
        prompt = f"""You are a Python code improvement expert. Based on the analysis, you need to address:

DETECTED ISSUES:
- Issue types: {dict(analysis['issue_types'])}
- Severity levels: {dict(analysis['severity_levels'])}
- Frameworks: {list(analysis['frameworks_detected'])}

Focus on fixing the issues while maintaining code functionality. Apply best practices for Python development and the detected frameworks.
"""
        
        if ai_provider.lower() == "deepseek":
            prompt += """
Return a valid JSON response with improved_code, applied_fixes, improvement_summary, confidence_score, and warnings.
"""
        
        return {
            'system_prompt': prompt,
            'templates_used': ['general'],
            'analysis': analysis,
            'confidence_boost': 0.05,
            'improvement_focus': ['general_improvements']
        }


# Global instance
_adaptive_prompt_generator = None

def get_adaptive_prompt_generator() -> AdaptivePromptGenerator:
    """Get or create adaptive prompt generator instance."""
    global _adaptive_prompt_generator
    if _adaptive_prompt_generator is None:
        _adaptive_prompt_generator = AdaptivePromptGenerator()
    return _adaptive_prompt_generator


# Example usage and testing
if __name__ == "__main__":
    # Demo with sample issues
    generator = AdaptivePromptGenerator()
    
    sample_issues = [
        Issue(
            filename="train.py",
            line=10,
            type="security",
            description="Use of eval() function detected",
            source="custom_rules",
            severity="error"
        ),
        Issue(
            filename="train.py",
            line=5,
            type="ml",
            description="Missing random seed for reproducibility",
            source="ml_rules",
            severity="warning"
        )
    ]
    
    sample_files = [
        CodeFile(
            filename="train.py",
            content="""
import torch
import numpy as np

def train_model():
    model = torch.nn.Linear(10, 1)
    data = np.random.random((100, 10))
    config = eval(user_input)  # Security issue
    return model
"""
        )
    ]
    
    result = generator.generate_custom_prompt(sample_issues, [], sample_files, "deepseek")
    print("Generated Custom Prompt:")
    print("=" * 50)
    print(result['system_prompt'])
    print("\nTemplates used:", result['templates_used'])
    print("Confidence boost:", result['confidence_boost'])
    print("Focus areas:", result['improvement_focus'])