"""
LLM-Powered Prompt Generator for CodeGuard API.
Uses AI to create highly customized system prompts based on specific audit results.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from models import Issue, Fix, CodeFile
from collections import Counter


@dataclass
class CustomPromptResponse:
    """Response containing the generated custom prompt and metadata."""
    system_prompt: str
    confidence_boost: float
    focus_areas: List[str]
    prompt_strategy: str
    estimated_effectiveness: float


class LLMPromptGenerator:
    """Generates custom system prompts using LLM analysis of audit results."""
    
    def __init__(self):
        self.openai_client = None
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize OpenAI client for prompt generation."""
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                print(f"Failed to initialize OpenAI for prompt generation: {e}")
                self.openai_client = None
    
    def generate_custom_prompt(self, issues: List[Issue], fixes: List[Fix], 
                             code_files: List[CodeFile], ai_provider: str = "openai") -> CustomPromptResponse:
        """Generate a custom system prompt based on audit results using LLM analysis."""
        
        if not self.openai_client:
            return self._fallback_prompt_generation(issues, fixes, code_files, ai_provider)
        
        # Analyze the audit results
        analysis = self._analyze_audit_context(issues, fixes, code_files)
        
        # Generate custom prompt using LLM
        prompt_request = self._build_prompt_generation_request(analysis, ai_provider)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using mini for cost efficiency
                messages=[
                    {"role": "system", "content": self._get_prompt_generator_system_prompt()},
                    {"role": "user", "content": prompt_request}
                ],
                max_tokens=1500,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI")
            result = json.loads(content)
            
            return CustomPromptResponse(
                system_prompt=result.get('system_prompt', ''),
                confidence_boost=float(result.get('confidence_boost', 0.1)),
                focus_areas=result.get('focus_areas', []),
                prompt_strategy=result.get('prompt_strategy', 'adaptive'),
                estimated_effectiveness=float(result.get('estimated_effectiveness', 0.7))
            )
            
        except Exception as e:
            print(f"LLM prompt generation failed: {e}")
            return self._fallback_prompt_generation(issues, fixes, code_files, ai_provider)
    
    def _analyze_audit_context(self, issues: List[Issue], fixes: List[Fix], 
                             code_files: List[CodeFile]) -> Dict[str, Any]:
        """Analyze audit results to understand the context and patterns."""
        
        analysis = {
            'total_issues': len(issues),
            'issue_breakdown': {},
            'severity_distribution': {},
            'source_tools': set(),
            'code_patterns': [],
            'frameworks_detected': set(),
            'file_types': [],
            'complexity_indicators': []
        }
        
        # Analyze issues
        issue_types = Counter()
        severities = Counter()
        
        for issue in issues:
            issue_types[issue.type] += 1
            severities[issue.severity] += 1
            analysis['source_tools'].add(issue.source)
            
            # Pattern detection
            desc_lower = issue.description.lower()
            if any(term in desc_lower for term in ['pickle', 'eval', 'exec']):
                analysis['code_patterns'].append('security_risk')
            if any(term in desc_lower for term in ['seed', 'random']):
                analysis['code_patterns'].append('reproducibility')
            if any(term in desc_lower for term in ['import', 'unused']):
                analysis['code_patterns'].append('unused_imports')
            if any(term in desc_lower for term in ['line too long', 'formatting']):
                analysis['code_patterns'].append('style_issues')
        
        analysis['issue_breakdown'] = dict(issue_types.most_common())
        analysis['severity_distribution'] = dict(severities.most_common())
        
        # Analyze code content
        all_code = '\n'.join(file.content for file in code_files)
        
        # Framework detection
        framework_patterns = {
            'pytorch': ['torch', 'nn.Module', 'cuda'],
            'tensorflow': ['tf.', 'keras', 'tensorflow'],
            'sklearn': ['sklearn', 'fit(', 'predict('],
            'gym': ['gym.make', 'env.step', 'env.reset'],
            'numpy': ['np.', 'numpy'],
            'pandas': ['pd.', 'DataFrame']
        }
        
        for framework, patterns in framework_patterns.items():
            if any(pattern in all_code for pattern in patterns):
                analysis['frameworks_detected'].add(framework)
        
        # File analysis
        for file in code_files:
            analysis['file_types'].append(file.filename.split('.')[-1])
            
            # Complexity indicators
            lines = file.content.split('\n')
            analysis['complexity_indicators'].extend([
                f"file_length_{len(lines)}",
                f"avg_line_length_{sum(len(line) for line in lines) // max(len(lines), 1)}"
            ])
        
        return analysis
    
    def _get_prompt_generator_system_prompt(self) -> str:
        """Get the system prompt for the LLM that generates custom prompts."""
        return """You are an expert prompt engineer specializing in code improvement tasks. Your job is to create highly effective system prompts for AI assistants that will fix specific code issues.

You will receive audit results and need to create a custom system prompt that:
1. Addresses the specific types of issues found
2. Leverages the detected frameworks and patterns
3. Provides clear, actionable guidance
4. Maximizes the AI's effectiveness for this specific codebase

Your response must be valid JSON with these fields:
- system_prompt: The complete system prompt (be specific and detailed)
- confidence_boost: Number 0.0-0.3 indicating how much this custom prompt improves results
- focus_areas: Array of 3-5 specific areas this prompt targets
- prompt_strategy: One of "security_focused", "ml_reproducibility", "style_cleanup", "performance", "comprehensive"
- estimated_effectiveness: Number 0.0-1.0 indicating expected prompt effectiveness

Make the system prompt:
- Specific to the detected issues and frameworks
- Include concrete examples when relevant
- Provide clear priorities and action steps
- Be authoritative but helpful in tone
- Include provider-specific instructions if needed"""
    
    def _build_prompt_generation_request(self, analysis: Dict[str, Any], ai_provider: str) -> str:
        """Build the request for LLM prompt generation."""
        
        request = f"""Generate a custom system prompt for fixing code issues with the following context:

AUDIT ANALYSIS:
- Total issues: {analysis['total_issues']}
- Issue types: {analysis['issue_breakdown']}
- Severity levels: {analysis['severity_distribution']}
- Analysis tools used: {list(analysis['source_tools'])}
- Code patterns detected: {analysis['code_patterns']}
- Frameworks detected: {list(analysis['frameworks_detected'])}
- File types: {set(analysis['file_types'])}

TARGET AI PROVIDER: {ai_provider}

The AI will need to:
1. Fix the specific issues listed above
2. Work with the detected frameworks
3. Maintain code functionality while improving quality
4. Provide explanations for changes made

Create a system prompt that maximizes effectiveness for this specific scenario."""

        if ai_provider.lower() == "deepseek":
            request += """

DEEPSEEK SPECIFIC REQUIREMENTS:
- Must include JSON response format instructions
- Should leverage DeepSeek's reasoning capabilities
- Include structured output requirements
- Add timeout considerations for complex analysis"""
        
        return request
    
    def _fallback_prompt_generation(self, issues: List[Issue], fixes: List[Fix], 
                                  code_files: List[CodeFile], ai_provider: str) -> CustomPromptResponse:
        """Fallback prompt generation when LLM is unavailable."""
        
        # Simple pattern-based prompt selection
        issue_types = [issue.type for issue in issues]
        
        if any(t in ['security', 'error'] for t in issue_types):
            strategy = "security_focused"
            prompt = """You are a security-focused code improvement specialist. Priority: Fix security vulnerabilities and critical errors first, then address other issues. Always provide safe alternatives and explain security improvements."""
            boost = 0.15
        elif any(t in ['ml', 'reproducibility'] for t in issue_types):
            strategy = "ml_reproducibility"
            prompt = """You are a machine learning code specialist. Priority: Ensure reproducibility through proper seeding, fix ML-specific issues, and optimize for training/inference performance."""
            boost = 0.12
        elif any(t in ['style', 'formatting'] for t in issue_types):
            strategy = "style_cleanup"
            prompt = """You are a Python code quality specialist. Priority: Clean up imports, fix formatting, improve readability while preserving functionality."""
            boost = 0.08
        else:
            strategy = "comprehensive"
            prompt = """You are a comprehensive code improvement specialist. Apply fixes systematically: errors first, then performance, then style. Maintain functionality while improving quality."""
            boost = 0.10
        
        if ai_provider.lower() == "deepseek":
            prompt += "\n\nReturn response as valid JSON with improved_code, applied_fixes, improvement_summary, confidence_score, and warnings fields."
        
        return CustomPromptResponse(
            system_prompt=prompt,
            confidence_boost=boost,
            focus_areas=[strategy],
            prompt_strategy=strategy,
            estimated_effectiveness=0.6
        )


# Global instance
_llm_prompt_generator = None

def get_llm_prompt_generator() -> LLMPromptGenerator:
    """Get or create LLM prompt generator instance."""
    global _llm_prompt_generator
    if _llm_prompt_generator is None:
        _llm_prompt_generator = LLMPromptGenerator()
    return _llm_prompt_generator


# Testing function
if __name__ == "__main__":
    # Test the LLM prompt generator
    generator = LLMPromptGenerator()
    
    # Sample test data
    test_issues = [
        Issue(
            filename="test.py",
            line=10,
            type="security",
            description="Use of eval() function detected - security risk",
            source="custom_rules",
            severity="error"
        ),
        Issue(
            filename="test.py",
            line=5,
            type="ml",
            description="Missing random seed for reproducibility",
            source="ml_rules",
            severity="warning"
        )
    ]
    
    test_files = [
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
    
    print("Testing LLM Prompt Generator...")
    result = generator.generate_custom_prompt(test_issues, [], test_files, "deepseek")
    
    print("\n" + "="*50)
    print("GENERATED CUSTOM PROMPT:")
    print("="*50)
    print(result.system_prompt)
    print("\nMetadata:")
    print(f"Strategy: {result.prompt_strategy}")
    print(f"Confidence boost: {result.confidence_boost}")
    print(f"Focus areas: {result.focus_areas}")
    print(f"Estimated effectiveness: {result.estimated_effectiveness}")