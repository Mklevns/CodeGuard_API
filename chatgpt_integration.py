"""
ChatGPT Integration for CodeGuard API.
Enables AI-powered code improvement suggestions and automatic fix implementation.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
from models import Issue, Fix, CodeFile

@dataclass
class CodeImprovementRequest:
    """Request for ChatGPT to improve code based on CodeGuard analysis."""
    original_code: str
    filename: str
    issues: List[Issue]
    fixes: List[Fix]
    improvement_level: str = "moderate"  # conservative, moderate, aggressive
    preserve_functionality: bool = True

@dataclass
class CodeImprovementResponse:
    """Response containing improved code and explanations."""
    improved_code: str
    applied_fixes: List[Dict[str, Any]]
    improvement_summary: str
    confidence_score: float
    warnings: List[str]

class ChatGPTCodeImprover:
    """ChatGPT integration for implementing CodeGuard suggestions."""
    
    def __init__(self):
        self.openai_client = None
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize OpenAI client with API key."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
    
    def improve_code(self, request: CodeImprovementRequest) -> CodeImprovementResponse:
        """
        Use ChatGPT to implement CodeGuard suggestions and improve code.
        
        Args:
            request: Code improvement request with original code and issues
            
        Returns:
            Response with improved code and explanations
        """
        if not self.openai_client:
            return self._fallback_improvement(request)
        
        try:
            # Build comprehensive prompt for ChatGPT
            prompt = self._build_improvement_prompt(request)
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.openai_client.chat.completions.create(
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
                temperature=0.1,  # Low temperature for consistent code improvements
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
            
        except Exception as e:
            return self._handle_improvement_error(request, str(e))
    
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
        self.improver = ChatGPTCodeImprover()
    
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

def get_code_improver() -> ChatGPTCodeImprover:
    """Get or create ChatGPT code improver instance."""
    global _code_improver
    if _code_improver is None:
        _code_improver = ChatGPTCodeImprover()
    return _code_improver

def get_batch_improver() -> BatchCodeImprover:
    """Get or create batch code improver instance."""
    global _batch_improver
    if _batch_improver is None:
        _batch_improver = BatchCodeImprover()
    return _batch_improver