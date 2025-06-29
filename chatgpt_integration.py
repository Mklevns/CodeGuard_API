"""
Multi-LLM Integration for CodeGuard API.
Enables AI-powered code improvement suggestions with OpenAI, DeepSeek R1, and other providers.
"""

import os
import json
import logging
import re
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
from models import Issue, Fix, CodeFile

@dataclass
class CodeImprovementRequest:
    """Request for AI to improve code based on CodeGuard analysis."""
    original_code: str
    filename: str
    issues: List[Issue]
    fixes: List[Fix]
    improvement_level: str = "moderate"  # conservative, moderate, aggressive
    preserve_functionality: bool = True
    ai_provider: str = "openai"  # openai, deepseek, gemini, claude
    ai_api_key: Optional[str] = None

@dataclass
class CodeImprovementResponse:
    """Response containing improved code and explanations."""
    improved_code: str
    applied_fixes: List[Dict[str, Any]]
    improvement_summary: str
    confidence_score: float
    warnings: List[str]

class MultiLLMCodeImprover:
    """Multi-LLM integration for implementing CodeGuard suggestions with OpenAI, DeepSeek R1, and others."""
    
    def __init__(self):
        self.openai_client = None
        self.deepseek_api_key = None
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available LLM providers."""
        # OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
        
        # DeepSeek R1
        self.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        
    def _call_deepseek_r1(self, prompt: str, api_key: Optional[str] = None) -> str:
        """Call DeepSeek R1 API for code improvement."""
        used_api_key = api_key or self.deepseek_api_key
        if not used_api_key:
            raise ValueError("DeepSeek API key not available")
        
        # DeepSeek R1 API endpoint
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {used_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-reasoner",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 8000  # Increased for reasoning model
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            message = result["choices"][0]["message"]
            
            # DeepSeek reasoner provides both reasoning_content and content
            reasoning_content = message.get("reasoning_content", "")
            final_content = message.get("content", "")
            
            # Return the final answer, optionally log reasoning for debugging
            if reasoning_content and len(reasoning_content) > 100:
                print(f"DeepSeek reasoning: {reasoning_content[:200]}...")
            
            return final_content or reasoning_content
            
        except requests.exceptions.Timeout:
            raise Exception("DeepSeek reasoner is thinking deeply about your code - this may take longer than expected")
        except requests.exceptions.RequestException as e:
            raise Exception(f"DeepSeek API request failed: {str(e)}")
        except (KeyError, IndexError) as e:
            raise Exception(f"DeepSeek API response format error: {str(e)}")
    
    def improve_code(self, request: CodeImprovementRequest) -> CodeImprovementResponse:
        """
        Use AI provider to implement CodeGuard suggestions and improve code.
        
        Args:
            request: Code improvement request with original code and issues
            
        Returns:
            Response with improved code and explanations
        """
        ai_provider = request.ai_provider.lower() if request.ai_provider else "openai"
        
        try:
            # Route to appropriate AI provider
            if ai_provider == "deepseek":
                return self._improve_with_deepseek(request)
            elif ai_provider == "openai":
                return self._improve_with_openai(request)
            else:
                # Default to OpenAI if provider not recognized
                return self._improve_with_openai(request)
                
        except Exception as e:
            return self._handle_improvement_error(request, str(e))
    
    def _improve_with_openai(self, request: CodeImprovementRequest) -> CodeImprovementResponse:
        """Use OpenAI GPT-4o for code improvement."""
        if not self.openai_client and not request.ai_api_key:
            return self._fallback_improvement(request)
        
        # Use provided API key if available
        client = self.openai_client
        if request.ai_api_key:
            client = OpenAI(api_key=request.ai_api_key)
        
        prompt = self._build_improvement_prompt(request)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Python developer specializing in ML/RL code improvements. "
                    "Apply the suggested fixes while maintaining code functionality and readability. "
                    "Return your response in JSON format with 'improved_code', 'applied_fixes', "
                    "'improvement_summary', 'confidence_score', and 'warnings' fields."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=4000
        )
        
        content = response.choices[0].message.content
        if content:
            result = json.loads(content)
        else:
            result = {}
        
        return CodeImprovementResponse(
            improved_code=result.get("improved_code", request.original_code),
            applied_fixes=result.get("applied_fixes", []),
            improvement_summary=result.get("improvement_summary", "No improvements applied"),
            confidence_score=float(result.get("confidence_score", 0.5)),
            warnings=result.get("warnings", [])
        )
    
    def _improve_with_deepseek(self, request: CodeImprovementRequest) -> CodeImprovementResponse:
        """Use DeepSeek R1 for code improvement."""
        prompt = self._build_improvement_prompt(request)
        
        # Add JSON format instruction to prompt for DeepSeek
        prompt += "\n\nIMPORTANT: Return your response as valid JSON with these exact keys: 'improved_code', 'applied_fixes', 'improvement_summary', 'confidence_score', 'warnings'."
        
        response_text = self._call_deepseek_r1(prompt, request.ai_api_key)
        
        # Parse DeepSeek response
        try:
            # Extract JSON from response if it contains additional text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback: try to parse entire response as JSON
                result = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, create structured response
            result = {
                "improved_code": request.original_code,
                "applied_fixes": [],
                "improvement_summary": "DeepSeek R1 provided text response but JSON parsing failed",
                "confidence_score": 0.3,
                "warnings": ["Response format parsing issue"]
            }
        
        return CodeImprovementResponse(
            improved_code=result.get("improved_code", request.original_code),
            applied_fixes=result.get("applied_fixes", []),
            improvement_summary=result.get("improvement_summary", "DeepSeek R1 code improvement completed"),
            confidence_score=float(result.get("confidence_score", 0.7)),
            warnings=result.get("warnings", [])
        )
    
    def _build_improvement_prompt(self, request: CodeImprovementRequest) -> str:
        """Build comprehensive prompt for ChatGPT code improvement."""
        
        issues_summary = self._format_issues_for_prompt(request.issues)
        fixes_summary = self._format_fixes_for_prompt(request.fixes)
        
        prompt = f"""
Improve the following Python code by implementing the suggested fixes from CodeGuard analysis.

**Original Code ({request.filename}):**
```python
{request.original_code}
```

**Issues Found:**
{issues_summary}

**Suggested Fixes:**
{fixes_summary}

**Improvement Level:** {request.improvement_level}
**Preserve Functionality:** {request.preserve_functionality}

Please:
1. Apply the suggested fixes where appropriate
2. Maintain the original functionality unless explicitly improving it
3. Add missing imports, error handling, and best practices
4. Fix security issues (pickle, eval usage) with safe alternatives
5. Improve ML/RL specific patterns (seeding, env resets, etc.)
6. Ensure code follows PEP 8 and modern Python practices

Return JSON with:
- "improved_code": The complete improved code
- "applied_fixes": List of fixes applied with descriptions
- "improvement_summary": Brief summary of changes made
- "confidence_score": Float 0-1 indicating confidence in improvements
- "warnings": Any potential issues or considerations
"""
        return prompt
    
    def _format_issues_for_prompt(self, issues: List[Issue]) -> str:
        """Format issues for ChatGPT prompt."""
        if not issues:
            return "No specific issues found."
        
        formatted = []
        for i, issue in enumerate(issues[:10], 1):  # Limit to top 10 issues
            formatted.append(f"{i}. {issue.type}: {issue.description} (Line {issue.line})")
        
        return "\n".join(formatted)
    
    def _format_fixes_for_prompt(self, fixes: List[Fix]) -> str:
        """Format fixes for ChatGPT prompt."""
        if not fixes:
            return "No specific fixes suggested."
        
        formatted = []
        for i, fix in enumerate(fixes[:10], 1):  # Limit to top 10 fixes
            formatted.append(f"{i}. {fix.suggestion}")
            if hasattr(fix, 'replacement_code') and fix.replacement_code:
                formatted.append(f"   Suggested: {fix.replacement_code}")
        
        return "\n".join(formatted)
    
    def _fallback_improvement(self, request: CodeImprovementRequest) -> CodeImprovementResponse:
        """Fallback improvement when OpenAI is not available."""
        
        # Apply basic auto-fixable improvements
        improved_code = request.original_code
        applied_fixes = []
        
        # Apply simple fixes that don't require AI
        for fix in request.fixes:
            if fix.auto_fixable and hasattr(fix, 'replacement_code') and fix.replacement_code:
                try:
                    # Simple string replacement for auto-fixable issues
                    if fix.line and fix.line <= len(improved_code.splitlines()):
                        lines = improved_code.splitlines()
                        lines[fix.line - 1] = fix.replacement_code
                        improved_code = "\n".join(lines)
                        applied_fixes.append({
                            "type": "auto_fix",
                            "description": fix.suggestion,
                            "line": fix.line
                        })
                except Exception:
                    continue
        
        return CodeImprovementResponse(
            improved_code=improved_code,
            applied_fixes=applied_fixes,
            improvement_summary=f"Applied {len(applied_fixes)} automatic fixes. OpenAI integration not available for advanced improvements.",
            confidence_score=0.7 if applied_fixes else 0.3,
            warnings=["OpenAI API key not configured. Only basic auto-fixes applied."]
        )
    
    def _handle_improvement_error(self, request: CodeImprovementRequest, error: str) -> CodeImprovementResponse:
        """Handle errors during code improvement."""
        return CodeImprovementResponse(
            improved_code=request.original_code,
            applied_fixes=[],
            improvement_summary=f"Error during improvement: {error}",
            confidence_score=0.0,
            warnings=[f"ChatGPT integration failed: {error}"]
        )

class BatchCodeImprover:
    """Handles batch improvement of multiple files."""
    
    def __init__(self):
        self.improver = MultiLLMCodeImprover()
    
    def improve_project(self, files: List[CodeFile], audit_results: Dict[str, Any]) -> Dict[str, CodeImprovementResponse]:
        """
        Improve multiple files in a project based on audit results.
        
        Args:
            files: List of code files to improve
            audit_results: Complete audit results from CodeGuard
            
        Returns:
            Dictionary mapping filenames to improvement responses
        """
        improvements = {}
        
        for file in files:
            # Extract issues and fixes for this specific file
            file_issues = [
                issue for issue in audit_results.get("issues", [])
                if hasattr(issue, 'filename') and issue.filename == file.filename
            ]
            file_fixes = [
                fix for fix in audit_results.get("fixes", [])
                if hasattr(fix, 'filename') and fix.filename == file.filename
            ]
            
            if file_issues or file_fixes:
                request = CodeImprovementRequest(
                    original_code=file.content,
                    filename=file.filename,
                    issues=file_issues,
                    fixes=file_fixes,
                    improvement_level="moderate"
                )
                
                improvements[file.filename] = self.improver.improve_code(request)
        
        return improvements

# Global instances
_code_improver = None
_batch_improver = None

def get_code_improver() -> MultiLLMCodeImprover:
    """Get or create multi-LLM code improver instance."""
    global _code_improver
    if _code_improver is None:
        _code_improver = MultiLLMCodeImprover()
    return _code_improver

def get_batch_improver() -> BatchCodeImprover:
    """Get or create batch code improver instance."""
    global _batch_improver
    if _batch_improver is None:
        _batch_improver = BatchCodeImprover()
    return _batch_improver