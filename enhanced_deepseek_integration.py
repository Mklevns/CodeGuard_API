"""
Enhanced DeepSeek API Integration based on official documentation.
Implements the latest DeepSeek features including reasoning models, FIM completion, and optimized prompting.
"""

import json
import requests
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek API integration."""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"  # Default chat model
    reasoning_model: str = "deepseek-reasoner"  # For complex reasoning tasks
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 30


class EnhancedDeepSeekClient:
    """Enhanced DeepSeek client following official API guidelines."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[DeepSeekConfig] = None):
        self.api_key = api_key
        self.config = config or DeepSeekConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        })
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       use_reasoning: bool = False, 
                       stream: bool = False,
                       response_format: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create a chat completion using DeepSeek API.
        
        Args:
            messages: List of message objects with role and content
            use_reasoning: Whether to use the reasoning model for complex tasks
            stream: Whether to stream the response
            response_format: Optional response format (e.g., {"type": "json_object"})
        """
        model = self.config.reasoning_model if use_reasoning else self.config.model
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": stream
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        try:
            response = self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle reasoning model response format
            if use_reasoning and "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice:
                    message = choice["message"]
                    # DeepSeek reasoning model returns both reasoning_content and content
                    if "reasoning_content" in message:
                        result["reasoning"] = message["reasoning_content"]
                    if "content" in message:
                        result["final_response"] = message["content"]
            
            return result
            
        except requests.exceptions.Timeout:
            raise Exception("DeepSeek API timeout - reasoning models may take longer to process")
        except requests.exceptions.RequestException as e:
            raise Exception(f"DeepSeek API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse DeepSeek response: {str(e)}")
    
    def fim_completion(self, prefix: str, suffix: str = "", 
                      max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Fill-in-the-middle completion using DeepSeek FIM API.
        
        Args:
            prefix: Code before the completion point
            suffix: Code after the completion point
            max_tokens: Maximum tokens to generate
        """
        payload = {
            "model": "deepseek-coder",  # Use coder model for FIM
            "prompt": f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>",
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stop": ["<fim_middle>", "<fim_suffix>", "<fim_prefix>"]
        }
        
        try:
            response = self.session.post(
                f"{self.config.base_url}/completions",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"DeepSeek FIM completion failed: {str(e)}")
    
    def generate_code_improvement(self, code: str, issues: List[Dict], 
                                 use_reasoning: bool = True) -> Dict[str, Any]:
        """
        Generate code improvements using DeepSeek's reasoning capabilities.
        
        Args:
            code: Original code to improve
            issues: List of issues detected by CodeGuard
            use_reasoning: Whether to use reasoning model for complex analysis
        """
        # Format issues for the prompt
        issues_text = "\n".join([
            f"- {issue.get('type', 'unknown')}: {issue.get('description', '')}"
            for issue in issues
        ])
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert Python code improvement assistant specializing in ML/RL code.
Your task is to fix the specific issues while maintaining code functionality.

Return your response as valid JSON with these fields:
- improved_code: The complete improved code
- applied_fixes: Array of descriptions of what was fixed
- improvement_summary: Summary of all changes made
- confidence_score: Number between 0.0 and 1.0
- warnings: Array of any warnings or considerations"""
            },
            {
                "role": "user",
                "content": f"""Please improve this Python code by fixing the identified issues:

**Original Code:**
```python
{code}
```

**Issues to Fix:**
{issues_text}

**Requirements:**
1. Fix all identified issues
2. Maintain the original functionality
3. Keep the same code structure where possible
4. Add comments explaining any significant changes
5. Return valid JSON response"""
            }
        ]
        
        return self.chat_completion(
            messages=messages,
            use_reasoning=use_reasoning,
            response_format={"type": "json_object"}
        )
    
    def analyze_code_patterns(self, code: str, framework: str = "unknown") -> Dict[str, Any]:
        """
        Analyze code patterns using DeepSeek reasoning for insights.
        
        Args:
            code: Code to analyze
            framework: ML/RL framework being used
        """
        messages = [
            {
                "role": "system",
                "content": f"""You are a code analysis expert specializing in {framework} and ML/RL patterns.
Analyze the provided code and identify patterns, potential issues, and improvement opportunities.

Focus on:
1. Framework-specific best practices
2. Security vulnerabilities
3. Performance optimization opportunities
4. Code maintainability
5. ML/RL specific patterns (reproducibility, environment handling, etc.)

Return analysis as JSON with detailed findings."""
            },
            {
                "role": "user",
                "content": f"Analyze this {framework} code:\n\n```python\n{code}\n```"
            }
        ]
        
        return self.chat_completion(
            messages=messages,
            use_reasoning=True,  # Use reasoning for detailed analysis
            response_format={"type": "json_object"}
        )


def create_deepseek_client(api_key: Optional[str] = None) -> EnhancedDeepSeekClient:
    """Create an enhanced DeepSeek client with optimal configuration."""
    config = DeepSeekConfig(
        model="deepseek-chat",
        reasoning_model="deepseek-reasoner",
        max_tokens=4000,
        temperature=0.1,
        timeout=45  # Longer timeout for reasoning tasks
    )
    
    return EnhancedDeepSeekClient(api_key=api_key, config=config)


# Testing and demonstration functions
if __name__ == "__main__":
    import os
    
    # Test the enhanced DeepSeek integration
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("⚠️ No DEEPSEEK_API_KEY found - skipping live API test")
        print("  Set DEEPSEEK_API_KEY environment variable to test integration")
        exit(0)
    
    print("🔧 Testing Enhanced DeepSeek Integration")
    print("=" * 50)
    
    client = create_deepseek_client(api_key)
    
    # Test 1: Basic chat completion
    print("\n1. Testing basic chat completion...")
    try:
        response = client.chat_completion([
            {"role": "user", "content": "Explain the difference between PyTorch and TensorFlow in 2 sentences."}
        ])
        
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            print(f"✓ Basic chat: {content[:100]}...")
        else:
            print("❌ Unexpected response format")
    except Exception as e:
        print(f"❌ Basic chat failed: {e}")
    
    # Test 2: Reasoning model
    print("\n2. Testing reasoning model...")
    try:
        response = client.chat_completion([
            {"role": "user", "content": "What are the key considerations when implementing a reproducible ML training pipeline?"}
        ], use_reasoning=True)
        
        if "reasoning" in response:
            print(f"✓ Reasoning process: {response['reasoning'][:100]}...")
        if "final_response" in response:
            print(f"✓ Final answer: {response['final_response'][:100]}...")
    except Exception as e:
        print(f"❌ Reasoning model failed: {e}")
    
    # Test 3: Code improvement
    print("\n3. Testing code improvement...")
    test_code = """
import torch

def train_model():
    model = torch.nn.Linear(10, 1)
    data = torch.randn(100, 10)
    return model
"""
    
    test_issues = [
        {"type": "ml", "description": "Missing random seed for reproducibility"},
        {"type": "style", "description": "Missing docstring"}
    ]
    
    try:
        response = client.generate_code_improvement(test_code, test_issues)
        
        if "final_response" in response:
            try:
                result = json.loads(response["final_response"])
                print(f"✓ Improved code generated: {len(result.get('improved_code', ''))} characters")
                print(f"✓ Applied fixes: {len(result.get('applied_fixes', []))}")
                print(f"✓ Confidence: {result.get('confidence_score', 0):.1%}")
            except json.JSONDecodeError:
                print("❌ Failed to parse improvement response as JSON")
        else:
            print("❌ No final response from reasoning model")
    except Exception as e:
        print(f"❌ Code improvement failed: {e}")
    
    print("\n✅ Enhanced DeepSeek integration testing complete!")
    print("   Features tested: Chat completion, reasoning model, code improvement")
    print("   Ready for integration with CodeGuard's LLM prompt generation system")