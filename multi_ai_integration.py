"""
Multi-AI Provider Integration for CodeGuard
Supports OpenAI GPT, Google Gemini, and Anthropic Claude
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

# Import AI provider clients
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    anthropic = None
    Anthropic = None

from models import Issue, Fix
from chatgpt_integration import CodeImprovementRequest, CodeImprovementResponse


class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def improve_code(self, request: CodeImprovementRequest) -> CodeImprovementResponse:
        """Improve code using the AI provider"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured"""
        pass


class OpenAIProvider(AIProvider):
    """OpenAI GPT provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None
        if self.api_key and OpenAI:
            self.client = OpenAI(api_key=self.api_key)
    
    def is_available(self) -> bool:
        return bool(self.client and self.api_key)
    
    def improve_code(self, request: CodeImprovementRequest) -> CodeImprovementResponse:
        if not self.is_available():
            return self._fallback_response(request)
        
        try:
            prompt = self._build_openai_prompt(request)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert Python developer specializing in ML/RL code improvements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            improved_code = response.choices[0].message.content
            
            return CodeImprovementResponse(
                improved_code=improved_code,
                applied_fixes=[{"provider": "openai", "model": "gpt-4o"}],
                improvement_summary="Applied OpenAI GPT-4o improvements for security, ML best practices, and code quality",
                confidence_score=0.9,
                warnings=[]
            )
            
        except Exception as e:
            logging.error(f"OpenAI provider error: {e}")
            return self._fallback_response(request)
    
    def _build_openai_prompt(self, request: CodeImprovementRequest) -> str:
        return f"""
        Improve this Python code by implementing the following CodeGuard suggestions:
        
        Original Code:
        ```python
        {request.original_code}
        ```
        
        Issues to fix:
        {self._format_issues(request.issues)}
        
        Fixes to implement:
        {self._format_fixes(request.fixes)}
        
        Requirements:
        - Preserve all functionality
        - Apply ML/RL best practices
        - Fix security issues
        - Improve code quality
        - Return only the improved code
        """
    
    def _format_issues(self, issues: List[Issue]) -> str:
        return "\n".join([f"- Line {issue.line}: {issue.description}" for issue in issues])
    
    def _format_fixes(self, fixes: List[Fix]) -> str:
        return "\n".join([f"- Line {fix.line}: {fix.suggestion}" for fix in fixes])
    
    def _fallback_response(self, request: CodeImprovementRequest) -> CodeImprovementResponse:
        return CodeImprovementResponse(
            improved_code=request.original_code,
            applied_fixes=[],
            improvement_summary="OpenAI provider unavailable - no changes made",
            confidence_score=0.0,
            warnings=["OpenAI API key not configured or service unavailable"]
        )


class GeminiProvider(AIProvider):
    """Google Gemini provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        if self.api_key and genai:
            self.client = genai.Client(api_key=self.api_key)
    
    def is_available(self) -> bool:
        return bool(self.client and self.api_key)
    
    def improve_code(self, request: CodeImprovementRequest) -> CodeImprovementResponse:
        if not self.is_available():
            return self._fallback_response(request)
        
        try:
            prompt = self._build_gemini_prompt(request)
            
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=4000
                )
            )
            
            improved_code = response.text or request.original_code
            
            return CodeImprovementResponse(
                improved_code=improved_code,
                applied_fixes=[{"provider": "gemini", "model": "gemini-2.5-pro"}],
                improvement_summary="Applied Google Gemini improvements for ML/RL optimization and security",
                confidence_score=0.85,
                warnings=[]
            )
            
        except Exception as e:
            logging.error(f"Gemini provider error: {e}")
            return self._fallback_response(request)
    
    def _build_gemini_prompt(self, request: CodeImprovementRequest) -> str:
        return f"""
        As an expert ML/RL Python developer, improve this code by fixing the identified issues:
        
        Original Code:
        {request.original_code}
        
        Issues: {len(request.issues)} problems found
        Fixes: {len(request.fixes)} suggestions available
        
        Apply improvements for:
        - Security vulnerabilities
        - ML/RL best practices
        - Code quality and performance
        - Type safety and error handling
        
        Return only the improved Python code.
        """
    
    def _fallback_response(self, request: CodeImprovementRequest) -> CodeImprovementResponse:
        return CodeImprovementResponse(
            improved_code=request.original_code,
            applied_fixes=[],
            improvement_summary="Gemini provider unavailable - no changes made",
            confidence_score=0.0,
            warnings=["Gemini API key not configured or service unavailable"]
        )


class ClaudeProvider(AIProvider):
    """Anthropic Claude provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        if self.api_key and Anthropic:
            self.client = Anthropic(api_key=self.api_key)
    
    def is_available(self) -> bool:
        return bool(self.client and self.api_key)
    
    def improve_code(self, request: CodeImprovementRequest) -> CodeImprovementResponse:
        if not self.is_available():
            return self._fallback_response(request)
        
        try:
            prompt = self._build_claude_prompt(request)
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            improved_code = response.content[0].text
            
            return CodeImprovementResponse(
                improved_code=improved_code,
                applied_fixes=[{"provider": "claude", "model": "claude-3-5-sonnet"}],
                improvement_summary="Applied Anthropic Claude improvements with focus on safety and best practices",
                confidence_score=0.88,
                warnings=[]
            )
            
        except Exception as e:
            logging.error(f"Claude provider error: {e}")
            return self._fallback_response(request)
    
    def _build_claude_prompt(self, request: CodeImprovementRequest) -> str:
        issues_text = "\n".join([f"Line {issue.line}: {issue.description}" for issue in request.issues])
        fixes_text = "\n".join([f"Line {fix.line}: {fix.suggestion}" for fix in request.fixes])
        
        return f"""
        Please improve this Python ML/RL code by implementing the suggested fixes:
        
        Original Code:
        ```python
        {request.original_code}
        ```
        
        Issues to Address:
        {issues_text}
        
        Suggested Fixes:
        {fixes_text}
        
        Please provide only the improved code with:
        - All security issues fixed
        - ML/RL best practices applied
        - Proper error handling
        - Type hints where appropriate
        - Preserved functionality
        """
    
    def _fallback_response(self, request: CodeImprovementRequest) -> CodeImprovementResponse:
        return CodeImprovementResponse(
            improved_code=request.original_code,
            applied_fixes=[],
            improvement_summary="Claude provider unavailable - no changes made",
            confidence_score=0.0,
            warnings=["Anthropic API key not configured or service unavailable"]
        )


class MultiAIManager:
    """Manages multiple AI providers for code improvement"""
    
    def __init__(self):
        self.providers = {
            "openai": OpenAIProvider(),
            "gemini": GeminiProvider(),
            "claude": ClaudeProvider()
        }
    
    def get_provider(self, provider_name: str, api_key: Optional[str] = None) -> AIProvider:
        """Get AI provider instance with optional custom API key"""
        if provider_name == "openai":
            return OpenAIProvider(api_key)
        elif provider_name == "gemini":
            return GeminiProvider(api_key)
        elif provider_name == "claude":
            return ClaudeProvider(api_key)
        else:
            # Default to OpenAI
            return OpenAIProvider(api_key)
    
    def improve_code_with_provider(
        self, 
        request: CodeImprovementRequest, 
        provider_name: str = "openai",
        api_key: Optional[str] = None
    ) -> CodeImprovementResponse:
        """Improve code using specified AI provider"""
        
        provider = self.get_provider(provider_name, api_key)
        
        if not provider.is_available():
            # Try fallback providers
            for fallback_name, fallback_provider in self.providers.items():
                if fallback_name != provider_name and fallback_provider.is_available():
                    logging.info(f"Falling back to {fallback_name} provider")
                    return fallback_provider.improve_code(request)
            
            # No providers available
            return CodeImprovementResponse(
                improved_code=request.original_code,
                applied_fixes=[],
                improvement_summary="No AI providers available - configure API keys in settings",
                confidence_score=0.0,
                warnings=["No AI providers configured. Add API keys in VS Code settings."]
            )
        
        return provider.improve_code(request)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [name for name, provider in self.providers.items() if provider.is_available()]


# Global multi-AI manager instance
multi_ai_manager = MultiAIManager()

def get_multi_ai_manager() -> MultiAIManager:
    """Get or create multi-AI manager instance"""
    return multi_ai_manager