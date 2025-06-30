
"""
Advanced Master Prompt System for CodeGuard API.
Implements sophisticated prompting strategies including persona-driven role-playing,
chain-of-thought reasoning, and specialized improvement tasks.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from models import Issue, Fix, CodeFile
from enum import Enum


class ImprovementType(Enum):
    """Types of code improvement tasks."""
    GENERAL = "general"
    SECURITY = "security"
    PERFORMANCE = "performance"
    READABILITY = "readability"
    MAINTAINABILITY = "maintainability"
    ML_OPTIMIZATION = "ml_optimization"
    RL_PATTERNS = "rl_patterns"


@dataclass
class ContextInfo:
    """Context information for prompt generation."""
    project_description: str = ""
    file_tree: str = ""
    dependencies: List[str] = None
    related_files: Dict[str, str] = None
    static_analysis_results: List[str] = None


class AdvancedMasterPromptGenerator:
    """Generates sophisticated, persona-driven prompts for different improvement tasks."""
    
    def __init__(self):
        self.personas = {
            ImprovementType.GENERAL: "a world-class Python expert and principal-level software engineer",
            ImprovementType.SECURITY: "a cybersecurity expert specializing in Python application security and penetration testing",
            ImprovementType.PERFORMANCE: "a performance engineering expert with deep knowledge of Python's internals and optimization techniques",
            ImprovementType.READABILITY: "a senior code reviewer and clean code advocate with expertise in Python best practices",
            ImprovementType.MAINTAINABILITY: "a software architecture expert focused on long-term code maintainability and team collaboration",
            ImprovementType.ML_OPTIMIZATION: "a machine learning infrastructure expert specializing in PyTorch and TensorFlow optimization",
            ImprovementType.RL_PATTERNS: "a reinforcement learning expert with deep knowledge of Gym, Stable Baselines, and environment handling"
        }
    
    def generate_master_prompt(self, 
                             improvement_type: ImprovementType,
                             filename: str,
                             original_code: str,
                             context: ContextInfo,
                             ai_provider: str = "openai") -> str:
        """Generate a sophisticated master prompt for code improvement."""
        
        persona = self.personas[improvement_type]
        
        # Base system prompt with persona and chain-of-thought
        system_prompt = f"""[SYSTEM]
You are {persona}. Your task is to perform a comprehensive review of the provided Python code. You must be meticulous, detail-oriented, and provide clear, actionable feedback.

Your response must be a single JSON object, with no extra text or explanations outside of the JSON structure. The JSON object must have the following keys:
- "thought_process": A string where you will think step-by-step about the code, analyzing its strengths and weaknesses before deciding on the final improvements.
- "improved_code": A string containing the full, refactored Python code.
- "explanation": A markdown-formatted string that details the changes you made. For each change, you must specify the original line number(s) and provide a clear justification for the improvement, categorized by type (e.g., Security, Performance, Readability, Best Practices).
- "confidence_score": A number between 0.0 and 1.0 indicating your confidence in the improvements.
- "applied_fixes": An array of strings describing each fix applied.
- "warnings": An array of any warnings or notes about the changes.

[USER]
Please analyze and improve the following Python code.

### **File Information**
- **Filename:** `{filename}`"""

        # Add project context if available
        if context.project_description:
            system_prompt += f"\n- **Project Context:** {context.project_description}"
        
        # Add repository context
        if context.file_tree or context.dependencies or context.related_files:
            system_prompt += "\n\n### **Repository Context**"
            
            if context.file_tree:
                system_prompt += f"\n#### File Tree:\n```\n{context.file_tree}\n```"
            
            if context.dependencies:
                system_prompt += "\n#### Key Dependencies:"
                for dep in context.dependencies:
                    system_prompt += f"\n- `{dep}`"
            
            if context.related_files:
                system_prompt += "\n#### Related File Snippets:"
                for file_name, content in context.related_files.items():
                    system_prompt += f"\n*`{file_name}`*\n```python\n{content}\n```\n"
        
        # Add static analysis results
        if context.static_analysis_results:
            system_prompt += "\n\n### **Analysis from Static Tools**\n"
            system_prompt += "The following issues were identified by static analysis tools:\n"
            for result in context.static_analysis_results:
                system_prompt += f"- {result}\n"
        
        # Add the original code
        system_prompt += f"\n\n### **Original Code (`{filename}`)**\n\n```python\n{original_code}\n```\n"
        
        # Add improvement-specific instructions
        system_prompt += self._get_improvement_specific_instructions(improvement_type)
        
        # Add provider-specific instructions
        if ai_provider.lower() == "deepseek":
            system_prompt += """

### **DeepSeek-Specific Instructions**
Use your reasoning capabilities to analyze the code thoroughly. Take time to understand the context and implications of each change before implementing fixes. Focus on producing clean, complete code replacements."""
        
        return system_prompt
    
    def _get_improvement_specific_instructions(self, improvement_type: ImprovementType) -> str:
        """Get specific instructions based on improvement type."""
        
        instructions = {
            ImprovementType.GENERAL: """
### **Your Task**
1. First, in the `thought_process` field, reason about the code's purpose, the provided context, and any identified issues.
2. Then, provide the fully refactored code in the `improved_code` field.
3. Finally, provide a detailed, categorized explanation of your changes in the `explanation` field.

Focus on:
- Code correctness and functionality
- Python best practices and PEP 8 compliance
- Error handling and robustness
- Code clarity and maintainability""",

            ImprovementType.SECURITY: """
### **Your Security Audit Task**
1. Perform a thorough security analysis in the `thought_process` field, identifying potential vulnerabilities.
2. Provide secure, refactored code in the `improved_code` field.
3. Detail each security improvement with CWE references where applicable.

Focus on identifying and fixing:
- Injection flaws (SQL, Command, etc.)
- Insecure deserialization (pickle, eval, exec)
- Hardcoded secrets and credentials
- Insufficient input validation
- Broken access control
- Insecure direct object references
- Cryptographic weaknesses""",

            ImprovementType.PERFORMANCE: """
### **Your Performance Optimization Task**
1. Analyze performance bottlenecks in the `thought_process` field.
2. Provide optimized code in the `improved_code` field.
3. Quantify improvements where possible (e.g., time complexity changes).

Focus on:
- Algorithmic complexity optimization
- Efficient use of data structures (lists vs. sets vs. dicts)
- Avoiding unnecessary computations in loops
- Memory allocation and garbage collection impact
- Vectorization opportunities (NumPy, pandas)
- Caching and memoization where appropriate""",

            ImprovementType.READABILITY: """
### **Your Code Readability Task**
1. Assess readability issues in the `thought_process` field.
2. Provide clean, readable code in the `improved_code` field.
3. Explain how each change improves code clarity.

Focus on:
- Clear variable and function naming
- Proper code organization and structure
- Meaningful comments and docstrings
- Elimination of code duplication
- Consistent formatting and style
- Breaking down complex functions""",

            ImprovementType.ML_OPTIMIZATION: """
### **Your ML Optimization Task**
1. Analyze ML-specific patterns and issues in the `thought_process` field.
2. Provide optimized ML code in the `improved_code` field.
3. Explain ML-specific improvements and their benefits.

Focus on:
- Proper random seeding for reproducibility
- Efficient data loading and preprocessing
- GPU memory management and optimization
- Model training best practices
- Gradient computation efficiency
- Batch processing optimization
- Memory leak prevention""",

            ImprovementType.RL_PATTERNS: """
### **Your RL Pattern Optimization Task**
1. Analyze RL-specific patterns and environment handling in the `thought_process` field.
2. Provide improved RL code in the `improved_code` field.
3. Explain RL-specific improvements and their impact on training.

Focus on:
- Proper environment reset patterns
- Action space and observation space handling
- Episode boundary management
- Reward normalization and clipping
- Environment seeding and reproducibility
- Training loop structure and efficiency
- Experience replay and data collection patterns"""
        }
        
        return instructions.get(improvement_type, instructions[ImprovementType.GENERAL])
    
    def generate_specialized_audit_prompt(self, issues: List[Issue], 
                                        fixes: List[Fix],
                                        code_files: List[CodeFile],
                                        improvement_type: ImprovementType = ImprovementType.GENERAL) -> Dict[str, Any]:
        """Generate a specialized prompt based on detected issues and improvement type."""
        
        if not code_files:
            raise ValueError("At least one code file is required")
        
        # Analyze the primary file and issues
        primary_file = code_files[0]
        
        # Create context from available information
        context = ContextInfo()
        
        # Extract project context from files
        if len(code_files) > 1:
            context.related_files = {
                file.filename: file.content[:500] + "..." if len(file.content) > 500 else file.content
                for file in code_files[1:3]  # Limit to 2 related files
            }
        
        # Convert issues to static analysis results
        if issues:
            context.static_analysis_results = [
                f"**{issue.source}:** {issue.description} (Line {issue.line}, Severity: {issue.severity})"
                for issue in issues
            ]
        
        # Detect framework dependencies
        all_code = ' '.join(file.content for file in code_files)
        detected_deps = []
        framework_patterns = {
            'torch': 'PyTorch',
            'tensorflow': 'TensorFlow', 
            'sklearn': 'Scikit-learn',
            'gym': 'OpenAI Gym',
            'numpy': 'NumPy',
            'pandas': 'Pandas'
        }
        
        for pattern, name in framework_patterns.items():
            if pattern in all_code:
                detected_deps.append(name)
        
        context.dependencies = detected_deps
        
        # Auto-detect improvement type based on issues if not specified
        if improvement_type == ImprovementType.GENERAL and issues:
            issue_types = [issue.type.lower() for issue in issues]
            if any('security' in t for t in issue_types):
                improvement_type = ImprovementType.SECURITY
            elif any('performance' in t for t in issue_types):
                improvement_type = ImprovementType.PERFORMANCE
            elif any('ml' in t or 'pytorch' in t or 'tensorflow' in t for t in issue_types):
                improvement_type = ImprovementType.ML_OPTIMIZATION
            elif any('rl' in t or 'gym' in t for t in issue_types):
                improvement_type = ImprovementType.RL_PATTERNS
        
        # Generate the master prompt
        prompt = self.generate_master_prompt(
            improvement_type=improvement_type,
            filename=primary_file.filename,
            original_code=primary_file.content,
            context=context
        )
        
        return {
            'system_prompt': prompt,
            'improvement_type': improvement_type.value,
            'persona_used': self.personas[improvement_type],
            'context_elements': {
                'has_related_files': bool(context.related_files),
                'has_dependencies': bool(context.dependencies),
                'has_static_analysis': bool(context.static_analysis_results)
            },
            'confidence_boost': self._calculate_confidence_boost(improvement_type, context)
        }
    
    def _calculate_confidence_boost(self, improvement_type: ImprovementType, context: ContextInfo) -> float:
        """Calculate confidence boost based on improvement type and available context."""
        base_boost = {
            ImprovementType.SECURITY: 0.15,
            ImprovementType.PERFORMANCE: 0.12,
            ImprovementType.ML_OPTIMIZATION: 0.14,
            ImprovementType.RL_PATTERNS: 0.13,
            ImprovementType.READABILITY: 0.08,
            ImprovementType.MAINTAINABILITY: 0.10,
            ImprovementType.GENERAL: 0.06
        }
        
        boost = base_boost[improvement_type]
        
        # Add context bonuses
        if context.related_files:
            boost += 0.03
        if context.dependencies:
            boost += 0.02
        if context.static_analysis_results:
            boost += 0.02
        
        return min(boost, 0.25)  # Cap at 25%


# Global instance
_advanced_prompt_generator = None

def get_advanced_prompt_generator() -> AdvancedMasterPromptGenerator:
    """Get or create advanced prompt generator instance."""
    global _advanced_prompt_generator
    if _advanced_prompt_generator is None:
        _advanced_prompt_generator = AdvancedMasterPromptGenerator()
    return _advanced_prompt_generator


# Example usage
if __name__ == "__main__":
    generator = AdvancedMasterPromptGenerator()
    
    # Test security-focused prompt
    context = ContextInfo(
        project_description="Financial services web API handling sensitive user data",
        dependencies=["fastapi", "pydantic", "sqlalchemy"],
        static_analysis_results=["Bandit: B404 - Use of subprocess detected"]
    )
    
    test_code = """
import subprocess
import pickle

def process_user_data(user_input):
    # Security risk: eval
    config = eval(user_input)
    
    # Security risk: subprocess
    result = subprocess.run(f"echo {config['name']}", shell=True)
    
    return result
"""
    
    prompt = generator.generate_master_prompt(
        improvement_type=ImprovementType.SECURITY,
        filename="user_processor.py",
        original_code=test_code,
        context=context
    )
    
    print("Generated Advanced Security Prompt:")
    print("=" * 60)
    print(prompt[:500] + "...")
