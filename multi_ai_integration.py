"""
Multi-AI Provider Integration for CodeGuard
Optimized for fast response times to prevent timeouts
"""

import os
import logging
from typing import List, Optional

from chatgpt_integration import ChatGPTCodeImprover, CodeImprovementRequest, CodeImprovementResponse


class MultiAIManager:
    """Simplified multi-AI manager with timeout optimization"""
    
    def __init__(self):
        self.chatgpt_improver = ChatGPTCodeImprover()
    
    async def improve_code_with_provider(
        self, 
        request: CodeImprovementRequest, 
        provider_name: str = "openai",
        api_key: Optional[str] = None
    ) -> CodeImprovementResponse:
        """Improve code using specified AI provider with async optimization"""
        
        # Use OpenAI implementation with custom API key if provided
        if provider_name == "openai" or not provider_name:
            return await self._improve_with_openai_async(request, api_key)
        
        # Fallback for other providers
        return self._fallback_improvement(request, provider_name)
    
    async def _improve_with_openai_async(self, request: CodeImprovementRequest, api_key: Optional[str]) -> CodeImprovementResponse:
        """Async OpenAI improvement with timeout optimization"""
        import asyncio
        
        def run_improvement():
            if api_key:
                # Temporarily set custom API key
                original_key = os.environ.get("OPENAI_API_KEY")
                os.environ["OPENAI_API_KEY"] = api_key
                try:
                    response = self.chatgpt_improver.improve_code(request)
                    response.applied_fixes.append({"provider": "openai", "custom_key": True})
                    return response
                except Exception as e:
                    logging.error(f"OpenAI with custom key failed: {e}")
                    return self._fallback_improvement(request, "openai")
                finally:
                    # Restore original key
                    if original_key:
                        os.environ["OPENAI_API_KEY"] = original_key
                    elif "OPENAI_API_KEY" in os.environ:
                        del os.environ["OPENAI_API_KEY"]
            else:
                # Use existing configuration
                try:
                    response = self.chatgpt_improver.improve_code(request)
                    response.applied_fixes.append({"provider": "openai", "environment_key": True})
                    return response
                except Exception as e:
                    logging.error(f"OpenAI with environment key failed: {e}")
                    return self._fallback_improvement(request, "openai")
        
        # Run with timeout to prevent hanging
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(run_improvement), 
                timeout=25.0  # 25 second timeout to prevent 30s client timeout
            )
        except asyncio.TimeoutError:
            logging.error("OpenAI request timed out after 25 seconds")
            return CodeImprovementResponse(
                improved_code=request.original_code,
                applied_fixes=[],
                improvement_summary="Request timed out - try with shorter code or simpler fixes",
                confidence_score=0.0,
                warnings=["AI request exceeded time limit - consider breaking code into smaller sections"]
            )
    
    def _fallback_improvement(self, request: CodeImprovementRequest, provider: str) -> CodeImprovementResponse:
        """Fast fallback when AI providers are unavailable"""
        return CodeImprovementResponse(
            improved_code=request.original_code,
            applied_fixes=[],
            improvement_summary=f"{provider} provider not available - configure API key in VS Code settings",
            confidence_score=0.0,
            warnings=[f"Configure {provider.upper()}_API_KEY in VS Code settings for AI improvements"]
        )
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        available = []
        if os.environ.get("OPENAI_API_KEY"):
            available.append("openai")
        if os.environ.get("GEMINI_API_KEY"):
            available.append("gemini")
        if os.environ.get("ANTHROPIC_API_KEY"):
            available.append("claude")
        return available or ["openai"]  # Default to openai


# Global multi-AI manager instance
_multi_ai_manager = None

def get_multi_ai_manager() -> MultiAIManager:
    """Get or create multi-AI manager instance"""
    global _multi_ai_manager
    if _multi_ai_manager is None:
        _multi_ai_manager = MultiAIManager()
    return _multi_ai_manager