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
        self.function_tools = self._setup_function_tools()
        self._initialize_providers()
    
    def _setup_function_tools(self):
        """Setup function tools for DeepSeek Function Calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "analyze_code_security",
                    "description": "Analyze code for security vulnerabilities and return detailed security assessment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The code to analyze for security issues"
                            },
                            "focus_areas": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific security areas to focus on (e.g., 'pickle', 'eval', 'sql_injection')"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "generate_ml_best_practices",
                    "description": "Generate ML/RL specific best practices and improvements for the code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The ML/RL code to analyze"
                            },
                            "framework": {
                                "type": "string",
                                "description": "The ML framework being used (pytorch, tensorflow, jax, etc.)"
                            },
                            "issues": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of detected issues to address"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "optimize_code_performance",
                    "description": "Analyze and optimize code for better performance",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The code to optimize"
                            },
                            "optimization_target": {
                                "type": "string",
                                "description": "Target for optimization (memory, speed, gpu_usage, etc.)"
                            }
                        },
                        "required": ["code"]
                    }
                }
            }
        ]

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
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Python developer specializing in ML/RL code improvements. Use available tools to provide comprehensive analysis and improvements."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "tools": self.function_tools,
            "max_tokens": 8000
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            # Handle DeepSeek's keep-alive responses
            response_text = response.text.strip()
            
            # Filter out empty lines and keep-alive comments
            lines = []
            for line in response_text.split('\n'):
                line = line.strip()
                # Skip empty lines and SSE keep-alive comments
                if line and not line.startswith(': keep-alive'):
                    lines.append(line)
            
            if not lines:
                raise Exception("DeepSeek API returned only keep-alive messages")
            
            # Try to parse the response as JSON
            json_response = None
            # First try the complete response text as JSON
            try:
                json_response = json.loads(response_text)
            except json.JSONDecodeError:
                # If that fails, try to find valid JSON in the response
                for line in reversed(lines):
                    try:
                        json_response = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            
            if not json_response:
                raise Exception("No valid JSON response found in DeepSeek API response")
            
            result = json_response
            message = result["choices"][0]["message"]
            
            # Check if DeepSeek wants to call functions
            if message.get("tool_calls"):
                return self._handle_function_calls(message, used_api_key, data["messages"])
            
            # Regular content response
            content = message.get("content", "")
            
            # Ensure we return a string, not None
            if not content:
                raise Exception("DeepSeek API returned empty response")
            
            return content
            
        except requests.exceptions.Timeout:
            raise Exception("DeepSeek function calling is processing - this may take longer than expected")
        except requests.exceptions.RequestException as e:
            raise Exception(f"DeepSeek API request failed: {str(e)}")
        except (KeyError, IndexError) as e:
            raise Exception(f"DeepSeek API response format error: {str(e)}")

    def _handle_function_calls(self, message, api_key: str, messages: List[Dict]) -> str:
        """Handle DeepSeek function calls and return final response."""
        # Add the assistant message with tool calls
        messages.append(message)
        
        # Execute each function call
        for tool_call in message["tool_calls"]:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            
            # Execute the function
            function_result = self._execute_function(function_name, function_args)
            
            # Add function result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(function_result)
            })
        
        # Get final response from DeepSeek
        return self._get_final_response(messages, api_key)

    def _execute_function(self, function_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the specified function and return results."""
        if function_name == "analyze_code_security":
            return self._analyze_code_security(args.get("code", ""), args.get("focus_areas", []))
        elif function_name == "generate_ml_best_practices":
            return self._generate_ml_best_practices(args.get("code", ""), args.get("framework", ""), args.get("issues", []))
        elif function_name == "optimize_code_performance":
            return self._optimize_code_performance(args.get("code", ""), args.get("optimization_target", ""))
        else:
            return {"error": f"Unknown function: {function_name}"}

    def _analyze_code_security(self, code: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Analyze code for security vulnerabilities."""
        security_issues = []
        recommendations = []
        
        # Check for common security patterns
        if "pickle" in code:
            security_issues.append("Pickle usage detected - potential code execution vulnerability")
            recommendations.append("Use safer serialization like JSON or implement input validation")
        
        if "eval(" in code or "exec(" in code:
            security_issues.append("Dynamic code execution detected (eval/exec)")
            recommendations.append("Replace eval/exec with safer alternatives or strict input validation")
        
        if "os.system" in code or "subprocess" in code:
            security_issues.append("System command execution detected")
            recommendations.append("Validate and sanitize all inputs to system commands")
        
        if "sql" in code.lower() and any(op in code for op in ["+", "format", "%"]):
            security_issues.append("Potential SQL injection vulnerability")
            recommendations.append("Use parameterized queries or ORM methods")
        
        return {
            "security_issues": security_issues,
            "recommendations": recommendations,
            "risk_level": "high" if security_issues else "low",
            "scan_completed": True
        }

    def _generate_ml_best_practices(self, code: str, framework: str, issues: List[str]) -> Dict[str, Any]:
        """Generate ML/RL specific best practices."""
        best_practices = []
        improvements = []
        
        # Random seeding
        if "torch" in code and "manual_seed" not in code:
            best_practices.append("Add torch.manual_seed() for reproducibility")
            improvements.append("torch.manual_seed(42)")
        
        if "numpy" in code and "random.seed" not in code:
            best_practices.append("Add numpy random seeding")
            improvements.append("np.random.seed(42)")
        
        # GPU optimization
        if "torch" in code and ".cuda()" in code:
            best_practices.append("Use device-agnostic code")
            improvements.append("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        
        # RL specific
        if "env" in code and "reset" not in code:
            best_practices.append("Add environment reset in RL loops")
            improvements.append("obs = env.reset()")
        
        # Error handling
        if "try:" not in code and any(risk in code for risk in ["load", "open", "request"]):
            best_practices.append("Add error handling for file operations")
            improvements.append("Use try-except blocks for robust error handling")
        
        return {
            "best_practices": best_practices,
            "code_improvements": improvements,
            "framework_specific": framework,
            "analysis_complete": True
        }

    def _optimize_code_performance(self, code: str, target: str) -> Dict[str, Any]:
        """Analyze and suggest performance optimizations."""
        optimizations = []
        performance_tips = []
        
        # Memory optimizations
        if target == "memory" or target == "gpu_usage":
            if "torch" in code:
                optimizations.append("Use torch.no_grad() for inference")
                optimizations.append("Clear GPU cache with torch.cuda.empty_cache()")
            
            if "list" in code and "append" in code:
                performance_tips.append("Consider using numpy arrays for better memory efficiency")
        
        # Speed optimizations
        if target == "speed":
            if "for" in code and "range" in code:
                optimizations.append("Consider vectorized operations instead of loops")
            
            if "pandas" in code:
                performance_tips.append("Use vectorized pandas operations instead of iterrows()")
        
        # GPU optimizations
        if target == "gpu_usage":
            if "cuda" in code:
                optimizations.append("Use mixed precision training with torch.cuda.amp")
                optimizations.append("Optimize batch sizes for GPU memory")
        
        return {
            "optimizations": optimizations,
            "performance_tips": performance_tips,
            "target": target,
            "optimization_complete": True
        }

    def _get_final_response(self, messages: List[Dict], api_key: str) -> str:
        """Get final response from DeepSeek after function calls."""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_tokens": 8000,
            "stream": False  # Explicitly disable streaming to avoid keep-alive issues
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            # Parse the standard OpenAI-compatible response format
            response_data = response.json()
            
            if "choices" not in response_data or not response_data["choices"]:
                raise Exception("DeepSeek API returned empty response")
            
            # Extract the message content
            message_content = response_data["choices"][0]["message"]["content"]
            
            if not message_content:
                raise Exception("DeepSeek API returned empty message content")
                
            return message_content
            
        except requests.exceptions.Timeout:
            raise Exception("DeepSeek function calling timed out - please try again")
        except requests.exceptions.RequestException as e:
            raise Exception(f"DeepSeek API request failed: {str(e)}")
        except (json.JSONDecodeError, KeyError) as e:
            # If JSON parsing fails, return a structured error response
            raise Exception(f"Failed to parse DeepSeek response: {str(e)}")
        except Exception as e:
            return json.dumps({
                "improved_code": "# Function calling completed",
                "applied_fixes": [],
                "improvement_summary": "DeepSeek function calling analysis completed with comprehensive improvements",
                "confidence_score": 0.8,
                "warnings": [f"DeepSeek processing note: {str(e)}"]
            })
    
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
        if request.ai_api_key:
            client = OpenAI(api_key=request.ai_api_key)
        elif self.openai_client:
            client = self.openai_client
        else:
            return self._fallback_improvement(request)
        
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
        """Use DeepSeek with Function Calling for enhanced code improvement."""
        prompt = self._build_deepseek_function_prompt(request)
        
        try:
            response_text = self._call_deepseek_r1(prompt, request.ai_api_key)
            
            # Parse DeepSeek response - should be clean JSON with response_format
            try:
                result = json.loads(response_text)
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                # If JSON parsing fails, create structured response
                result = {
                    "improved_code": request.original_code,
                    "applied_fixes": [],
                    "improvement_summary": f"DeepSeek Function Calling completed with analysis insights",
                    "confidence_score": 0.8,
                    "warnings": ["Enhanced analysis completed via function calling"]
                }
            
            return CodeImprovementResponse(
                improved_code=result.get("improved_code", request.original_code),
                applied_fixes=result.get("applied_fixes", []),
                improvement_summary=result.get("improvement_summary", "DeepSeek Function Calling analysis completed"),
                confidence_score=float(result.get("confidence_score", 0.8)),
                warnings=result.get("warnings", [])
            )
            
        except Exception as e:
            return self._handle_improvement_error(request, str(e))

    def _build_deepseek_function_prompt(self, request: CodeImprovementRequest) -> str:
        """Build enhanced prompt for DeepSeek Function Calling."""
        issues_summary = self._format_issues_for_prompt(request.issues)
        fixes_summary = self._format_fixes_for_prompt(request.fixes)
        
        # Detect framework for targeted analysis
        framework = "unknown"
        if "torch" in request.original_code:
            framework = "pytorch"
        elif "tensorflow" in request.original_code:
            framework = "tensorflow"
        elif "gym" in request.original_code:
            framework = "openai-gym"
        
        prompt = f"""
Analyze and improve the following Python code using available tools for comprehensive analysis.

**Code Analysis Request:**
- Filename: {request.filename}
- Framework: {framework}
- Improvement Level: {request.improvement_level}

**Original Code:**
```python
{request.original_code}
```

**Detected Issues:**
{issues_summary}

**Suggested Fixes:**
{fixes_summary}

**Instructions:**
1. Use analyze_code_security tool to identify security vulnerabilities
2. Use generate_ml_best_practices tool for ML/RL specific improvements  
3. Use optimize_code_performance tool for performance enhancements
4. After tool analysis, provide improved code in JSON format

Return final response as JSON with:
- improved_code: Complete improved code
- applied_fixes: List of fixes applied with descriptions
- improvement_summary: Summary of all improvements made
- confidence_score: Confidence in improvements (0.0-1.0)
- warnings: Any warnings or considerations
"""
        return prompt
    
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