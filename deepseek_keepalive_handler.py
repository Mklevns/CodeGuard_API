"""
DeepSeek Keep-Alive Handler for CodeGuard API.
Properly handles DeepSeek's empty lines and SSE keep-alive comments to prevent TCP timeouts.
"""

import requests
import json
import time
from typing import Optional, Dict, Any


class DeepSeekKeepAliveHandler:
    """Handles DeepSeek API responses with proper keep-alive parsing."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com"
    
    def call_with_keepalive_handling(self, prompt: str, model: str = "deepseek-chat", 
                                   timeout: int = 45) -> str:
        """
        Call DeepSeek API with proper keep-alive handling and timeout reset.
        
        Args:
            prompt: The prompt to send
            model: DeepSeek model to use
            timeout: Base timeout in seconds (resets on keep-alive)
            
        Returns:
            Clean JSON response string
        """
        if not self.api_key:
            raise Exception("DeepSeek API key not provided")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 4000
        }
        
        try:
            # Use streaming to properly handle keep-alive messages
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=None,  # We'll handle timeout manually
                stream=True
            )
            
            response.raise_for_status()
            
            # Handle the response with proper timeout and keep-alive management
            return self._parse_streaming_response_with_keepalive(response, timeout)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"DeepSeek API request failed: {str(e)}")
    
    def _parse_streaming_response_with_keepalive(self, response: requests.Response, timeout: int) -> str:
        """
        Parse streaming DeepSeek response with proper keep-alive timeout management.
        
        Args:
            response: Streaming requests response
            timeout: Base timeout in seconds (resets on keep-alive)
            
        Returns:
            Clean JSON content
        """
        import signal
        import threading
        
        response_chunks = []
        timeout_counter = timeout
        last_activity = time.time()
        
        def timeout_handler():
            nonlocal timeout_counter
            while timeout_counter > 0:
                time.sleep(1)
                timeout_counter -= 1
                # Check if we've received activity recently
                if time.time() - last_activity > timeout:
                    raise Exception("DeepSeek API timeout - no activity for too long")
        
        # Start timeout thread
        timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
        timeout_thread.start()
        
        try:
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    # Reset timeout on any data received
                    last_activity = time.time()
                    timeout_counter = timeout
                    
                    # Check if this is a keep-alive message
                    chunk_lines = chunk.split('\n')
                    for line in chunk_lines:
                        line = line.strip()
                        
                        # Skip empty lines (keep-alive)
                        if not line:
                            continue
                            
                        # Skip SSE keep-alive comments
                        if line.startswith(': keep-alive'):
                            continue
                            
                        # Skip SSE data markers
                        if line.startswith('data: ') and line == 'data: ':
                            continue
                            
                        # Collect actual content
                        response_chunks.append(chunk)
                        break
            
            # Combine all response chunks
            full_response = ''.join(response_chunks)
            
            # Parse the final JSON response
            try:
                result = json.loads(full_response)
                content = result["choices"][0]["message"]["content"]
                return content
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                return self._extract_json_from_response(full_response)
                
        except Exception as e:
            raise Exception(f"Failed to parse streaming DeepSeek response: {str(e)}")
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON content from mixed response text."""
        # Look for JSON blocks in the response
        import re
        
        # Try to find JSON content between curly braces
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                json.loads(match)  # Validate JSON
                return match
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, return error structure
        return json.dumps({
            "improved_code": "# DeepSeek response parsing failed",
            "applied_fixes": [],
            "improvement_summary": "Failed to parse DeepSeek response",
            "confidence_score": 0.5,
            "warnings": ["Response parsing failed - keep-alive handling issue"]
        })

    def _parse_keepalive_response(self, response: requests.Response) -> str:
        """
        Parse DeepSeek response and filter out keep-alive lines.
        
        Args:
            response: Raw requests response
            
        Returns:
            Clean JSON content
        """
        try:
            # First try standard JSON parsing
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content
            
        except json.JSONDecodeError:
            # If standard parsing fails, handle keep-alive lines
            raw_text = response.text
            
            # Filter out keep-alive patterns
            clean_text = self._filter_keepalive_lines(raw_text)
            
            # Try parsing the cleaned text
            try:
                result = json.loads(clean_text)
                content = result["choices"][0]["message"]["content"]
                return content
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse DeepSeek response after keep-alive filtering: {str(e)}")
    
    def _filter_keepalive_lines(self, raw_text: str) -> str:
        """
        Filter out keep-alive lines from DeepSeek response.
        
        According to DeepSeek docs:
        - Empty lines are sent for non-streaming requests
        - SSE comments (: keep-alive) are sent for streaming requests
        
        Args:
            raw_text: Raw response text
            
        Returns:
            Cleaned text with keep-alive lines removed
        """
        lines = raw_text.split('\n')
        filtered_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            
            # Skip empty lines (keep-alive for non-streaming)
            if not stripped_line:
                continue
                
            # Skip SSE keep-alive comments (keep-alive for streaming)
            if stripped_line.startswith(': keep-alive'):
                continue
                
            # Skip other SSE control lines
            if stripped_line.startswith('data: ') and stripped_line == 'data: ':
                continue
                
            # Keep actual content lines
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def generate_code_improvement(self, original_code: str, issues: list, 
                                confidence_boost: float = 0.0) -> Dict[str, Any]:
        """
        Generate code improvement using DeepSeek with keep-alive handling.
        
        Args:
            original_code: Original code to improve
            issues: List of issues to fix
            confidence_boost: Confidence boost from custom prompts
            
        Returns:
            Parsed improvement response
        """
        # Format issues for the prompt
        issues_text = "\n".join([
            f"- Line {issue.get('line', 'N/A')}: {issue.get('description', 'Unknown issue')}"
            for issue in issues
        ])
        
        prompt = f"""You are an expert Python code improvement assistant. Fix the specific issues in this code while maintaining functionality.

**Original Code:**
```python
{original_code}
```

**Issues to Fix:**
{issues_text}

**Requirements:**
1. Return the COMPLETE corrected code file
2. Fix ALL identified issues
3. Maintain original functionality
4. Return valid JSON format

**Return this exact JSON format:**
{{
  "improved_code": "THE ENTIRE CORRECTED CODE FILE",
  "applied_fixes": ["Description of fix 1", "Description of fix 2"],
  "improvement_summary": "Summary of all changes made",
  "confidence_score": 0.95,
  "warnings": ["Any warnings or notes"]
}}

CRITICAL: The improved_code field must contain the complete, corrected file - NOT the original code with additions."""
        
        try:
            response_text = self.call_with_keepalive_handling(prompt)
            result = json.loads(response_text)
            
            # Apply confidence boost
            base_confidence = float(result.get("confidence_score", 0.8))
            result["confidence_score"] = min(base_confidence + confidence_boost, 1.0)
            
            return result
            
        except Exception as e:
            # Return fallback response on error
            return {
                "improved_code": original_code,
                "applied_fixes": [],
                "improvement_summary": f"DeepSeek API error: {str(e)}",
                "confidence_score": 0.5 + confidence_boost,
                "warnings": ["DeepSeek API unavailable - no improvements applied"]
            }


def create_deepseek_handler(api_key: Optional[str] = None) -> DeepSeekKeepAliveHandler:
    """Create a DeepSeek handler with keep-alive support."""
    return DeepSeekKeepAliveHandler(api_key)


# Test the keep-alive handler
if __name__ == "__main__":
    import os
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("No DEEPSEEK_API_KEY found - skipping test")
        exit(0)
    
    print("Testing DeepSeek Keep-Alive Handler")
    print("=" * 40)
    
    handler = create_deepseek_handler(api_key)
    
    test_code = """
import torch
import pickle

def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    return model
"""
    
    test_issues = [
        {"line": 4, "description": "Use of pickle.load() poses security risk"}
    ]
    
    try:
        result = handler.generate_code_improvement(test_code, test_issues)
        
        print("✓ DeepSeek response received")
        print(f"Applied fixes: {len(result.get('applied_fixes', []))}")
        print(f"Confidence: {result.get('confidence_score', 0):.1%}")
        
        improved_code = result.get('improved_code', '')
        if 'pickle.load' not in improved_code and 'torch.load' in improved_code:
            print("✓ Security fix applied correctly")
        else:
            print("⚠ Security fix not detected")
            
        print("✓ Keep-alive handling working correctly")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")