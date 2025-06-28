"""
OpenAI GPT Connector for CodeGuard API.
Enables natural language queries about past audits and provides intelligent explanations.
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class AuditQueryResult:
    """Result of a natural language audit query."""
    query: str
    results: List[Dict[str, Any]]
    summary: str
    total_matches: int


class GPTConnector:
    """Connector for OpenAI GPT integration with audit data."""
    
    def __init__(self, telemetry_collector):
        self.telemetry = telemetry_collector
        self.openai_available = bool(os.getenv("OPENAI_API_KEY"))
        
        if self.openai_available:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                self.openai_available = False
    
    def query_audits(self, natural_query: str) -> AuditQueryResult:
        """Process natural language queries about past audits."""
        # Parse query to extract intent and parameters
        query_intent = self._parse_query_intent(natural_query)
        
        # Execute query based on intent
        if query_intent["type"] == "framework_issues":
            results = self._query_framework_issues(query_intent)
        elif query_intent["type"] == "error_pattern":
            results = self._query_error_patterns(query_intent)
        elif query_intent["type"] == "recent_audits":
            results = self._query_recent_audits(query_intent)
        elif query_intent["type"] == "performance":
            results = self._query_performance_data(query_intent)
        else:
            results = self._general_audit_search(query_intent)
        
        # Generate summary
        summary = self._generate_query_summary(natural_query, results, query_intent)
        
        return AuditQueryResult(
            query=natural_query,
            results=results,
            summary=summary,
            total_matches=len(results)
        )
    
    def _parse_query_intent(self, query: str) -> Dict[str, Any]:
        """Parse natural language query to extract intent and parameters."""
        query_lower = query.lower()
        
        # Framework-specific queries
        if any(fw in query_lower for fw in ['gym', 'pytorch', 'tensorflow', 'jax', 'sklearn']):
            framework = None
            for fw in ['gym', 'pytorch', 'tensorflow', 'jax', 'sklearn', 'stable-baselines']:
                if fw in query_lower:
                    framework = fw
                    break
            
            return {
                "type": "framework_issues",
                "framework": framework,
                "limit": self._extract_number(query, default=10)
            }
        
        # Error pattern queries
        if any(word in query_lower for word in ['error', 'issue', 'problem', 'bug']):
            error_type = None
            for err_type in ['style', 'error', 'best_practice', 'complexity', 'security']:
                if err_type in query_lower:
                    error_type = err_type
                    break
            
            return {
                "type": "error_pattern",
                "error_type": error_type,
                "limit": self._extract_number(query, default=10)
            }
        
        # Recent audit queries
        if any(word in query_lower for word in ['recent', 'last', 'latest']):
            return {
                "type": "recent_audits",
                "limit": self._extract_number(query, default=10),
                "days": self._extract_days(query, default=7)
            }
        
        # Performance queries
        if any(word in query_lower for word in ['slow', 'fast', 'performance', 'time']):
            return {
                "type": "performance",
                "limit": self._extract_number(query, default=10)
            }
        
        # General search
        return {
            "type": "general",
            "terms": self._extract_search_terms(query),
            "limit": self._extract_number(query, default=10)
        }
    
    def _extract_number(self, text: str, default: int = 10) -> int:
        """Extract number from query text."""
        numbers = re.findall(r'\b(\d+)\b', text)
        return int(numbers[0]) if numbers else default
    
    def _extract_days(self, text: str, default: int = 7) -> int:
        """Extract time period from query text."""
        if 'day' in text:
            numbers = re.findall(r'(\d+)\s*days?', text)
            return int(numbers[0]) if numbers else default
        elif 'week' in text:
            numbers = re.findall(r'(\d+)\s*weeks?', text)
            return int(numbers[0]) * 7 if numbers else default * 7
        return default
    
    def _extract_search_terms(self, text: str) -> List[str]:
        """Extract search terms from query text."""
        # Remove common query words
        stop_words = {'show', 'me', 'the', 'last', 'with', 'find', 'get', 'all'}
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _query_framework_issues(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query issues related to specific frameworks."""
        framework = intent["framework"]
        limit = intent["limit"]
        
        # Use in-memory data for demonstration
        if hasattr(self.telemetry, 'memory_sessions'):
            sessions = self.telemetry.memory_sessions
            framework_sessions = [
                s for s in sessions 
                if s.framework_detected == framework
            ][:limit]
            
            return [self._session_to_result(s) for s in framework_sessions]
        else:
            # Generate sample data for demonstration
            return self._generate_sample_framework_results(framework, limit)
    
    def _query_error_patterns(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query specific error patterns."""
        error_type = intent["error_type"]
        limit = intent["limit"]
        
        # Use in-memory data if available
        if hasattr(self.telemetry, 'memory_errors'):
            errors = self.telemetry.memory_errors
            if error_type:
                filtered_errors = [e for e in errors if error_type in str(e)][:limit]
            else:
                filtered_errors = errors[:limit]
            
            return [{"error": str(e), "timestamp": datetime.now().isoformat()} for e in filtered_errors]
        else:
            return self._generate_sample_error_results(error_type, limit)
    
    def _query_recent_audits(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query recent audit sessions."""
        limit = intent["limit"]
        days = intent["days"]
        cutoff_date = datetime.now() - timedelta(days=days)
        
        if hasattr(self.telemetry, 'memory_sessions'):
            sessions = self.telemetry.memory_sessions
            recent_sessions = [
                s for s in sessions 
                if s.timestamp >= cutoff_date
            ][:limit]
            
            return [self._session_to_result(s) for s in recent_sessions]
        else:
            return self._generate_sample_recent_results(limit)
    
    def _query_performance_data(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query performance-related audit data."""
        limit = intent["limit"]
        
        if hasattr(self.telemetry, 'memory_sessions'):
            sessions = self.telemetry.memory_sessions
            # Sort by analysis time
            sorted_sessions = sorted(sessions, key=lambda s: s.analysis_time_ms, reverse=True)[:limit]
            
            return [self._session_to_performance_result(s) for s in sorted_sessions]
        else:
            return self._generate_sample_performance_results(limit)
    
    def _general_audit_search(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform general search across audit data."""
        terms = intent["terms"]
        limit = intent["limit"]
        
        # Simple keyword matching for demonstration
        results = []
        if hasattr(self.telemetry, 'memory_sessions'):
            for session in self.telemetry.memory_sessions[:limit]:
                if any(term in str(session.error_types) for term in terms):
                    results.append(self._session_to_result(session))
        
        return results[:limit]
    
    def _session_to_result(self, session) -> Dict[str, Any]:
        """Convert audit session to result format."""
        return {
            "session_id": session.session_id,
            "timestamp": session.timestamp.isoformat(),
            "file_count": session.file_count,
            "total_issues": session.total_issues,
            "framework": session.framework_detected,
            "error_types": session.error_types,
            "analysis_time_ms": session.analysis_time_ms
        }
    
    def _session_to_performance_result(self, session) -> Dict[str, Any]:
        """Convert session to performance-focused result."""
        return {
            "session_id": session.session_id,
            "timestamp": session.timestamp.isoformat(),
            "analysis_time_ms": session.analysis_time_ms,
            "files_per_second": session.file_count / (session.analysis_time_ms / 1000) if session.analysis_time_ms > 0 else 0,
            "issues_per_file": session.total_issues / session.file_count if session.file_count > 0 else 0
        }
    
    def _generate_sample_framework_results(self, framework: str, limit: int) -> List[Dict[str, Any]]:
        """Generate sample framework-specific results."""
        results = []
        for i in range(min(limit, 5)):
            results.append({
                "session_id": f"sample-{framework}-{i}",
                "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                "framework": framework,
                "issues_found": [
                    f"Missing {framework} best practices",
                    f"{framework.title()} deprecation warnings",
                    f"Inefficient {framework} usage patterns"
                ][:i+1],
                "total_issues": (i + 1) * 3
            })
        return results
    
    def _generate_sample_error_results(self, error_type: str, limit: int) -> List[Dict[str, Any]]:
        """Generate sample error pattern results."""
        results = []
        for i in range(min(limit, 5)):
            results.append({
                "error_pattern": error_type or "general",
                "occurrences": 10 - i,
                "latest_occurrence": (datetime.now() - timedelta(hours=i)).isoformat(),
                "description": f"Common {error_type or 'coding'} issue pattern #{i+1}"
            })
        return results
    
    def _generate_sample_recent_results(self, limit: int) -> List[Dict[str, Any]]:
        """Generate sample recent audit results."""
        results = []
        for i in range(min(limit, 5)):
            results.append({
                "session_id": f"recent-{i}",
                "timestamp": (datetime.now() - timedelta(hours=i*2)).isoformat(),
                "file_count": 2 + i,
                "total_issues": 15 + i*3,
                "framework": ["pytorch", "tensorflow", "gym", "sklearn"][i % 4],
                "summary": f"Audit session {i+1} with multiple framework detections"
            })
        return results
    
    def _generate_sample_performance_results(self, limit: int) -> List[Dict[str, Any]]:
        """Generate sample performance results."""
        results = []
        base_times = [8500, 6200, 4100, 3200, 2800]
        for i, time_ms in enumerate(base_times[:limit]):
            results.append({
                "session_id": f"perf-{i}",
                "analysis_time_ms": time_ms,
                "performance_category": "slow" if time_ms > 5000 else "fast",
                "bottleneck": "pylint analysis" if time_ms > 6000 else "normal processing"
            })
        return results
    
    def _generate_query_summary(self, query: str, results: List[Dict], intent: Dict[str, Any]) -> str:
        """Generate natural language summary of query results."""
        if not results:
            return f"No results found for query: '{query}'"
        
        count = len(results)
        query_type = intent["type"]
        
        if query_type == "framework_issues":
            framework = intent["framework"]
            return f"Found {count} audit sessions with {framework} issues. Most recent issues include framework-specific patterns and best practice violations."
        
        elif query_type == "error_pattern":
            error_type = intent.get("error_type", "general")
            return f"Found {count} instances of {error_type} error patterns. Common issues include style violations and code quality concerns."
        
        elif query_type == "recent_audits":
            days = intent["days"]
            return f"Found {count} audit sessions from the last {days} days. Recent activity shows consistent code analysis usage."
        
        elif query_type == "performance":
            avg_time = sum(r.get("analysis_time_ms", 0) for r in results) / count if count > 0 else 0
            return f"Found {count} performance data points. Average analysis time: {avg_time:.0f}ms."
        
        else:
            return f"Found {count} results matching your query. Results include various audit sessions and error patterns."


class IssueExplainer:
    """Provides natural language explanations for code issues."""
    
    def __init__(self):
        self.openai_available = bool(os.getenv("OPENAI_API_KEY"))
        
        if self.openai_available:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                self.openai_available = False
        
        # Fallback explanations for common issues
        self.static_explanations = {
            "pickle": {
                "why_unsafe": "Pickle is unsafe because it can execute arbitrary code during deserialization. When you unpickle data from an untrusted source, malicious code embedded in the pickle can be executed on your system.",
                "alternatives": "Use JSON for simple data, joblib for ML models, or msgpack for binary serialization. These formats are safer because they don't execute code during loading.",
                "example": "Instead of: pickle.load(file)\nUse: json.load(file) or joblib.load(file)"
            },
            "eval": {
                "why_unsafe": "The eval() function executes any Python code passed to it as a string. This makes it extremely dangerous if the input comes from users, as they can execute arbitrary commands.",
                "alternatives": "Use ast.literal_eval() for safe evaluation of literals, or parse data with specific parsers like JSON.",
                "example": "Instead of: eval(user_input)\nUse: ast.literal_eval(user_input) or json.loads(user_input)"
            },
            "global_variables": {
                "why_problematic": "Global variables make code harder to test, debug, and maintain. They create hidden dependencies and can cause issues in multi-threaded environments.",
                "alternatives": "Pass data as function parameters, use class attributes, or employ dependency injection patterns.",
                "example": "Instead of: global_var = value\nUse: def function(parameter): return result"
            },
            "hardcoded_paths": {
                "why_problematic": "Hardcoded paths break when code runs on different systems or environments. They make code less portable and harder to deploy.",
                "alternatives": "Use environment variables, configuration files, or relative paths with pathlib.",
                "example": "Instead of: '/tmp/data.csv'\nUse: os.getenv('DATA_PATH', 'data.csv')"
            },
            "print_statements": {
                "why_avoid": "Print statements in production code create uncontrolled output and can't be easily filtered or redirected. They don't provide log levels or timestamps.",
                "alternatives": "Use the logging module which provides levels, formatting, and configurable output destinations.",
                "example": "Instead of: print('Error occurred')\nUse: logger.error('Error occurred')"
            }
        }
    
    def explain_issue(self, issue_description: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Provide natural language explanation for a code issue."""
        # Extract key terms from issue description
        key_terms = self._extract_key_terms(issue_description)
        
        # Try to find static explanation first
        static_explanation = self._get_static_explanation(key_terms)
        if static_explanation:
            return {
                "issue": issue_description,
                "explanation": static_explanation,
                "source": "static",
                "confidence": "high"
            }
        
        # If OpenAI is available, generate dynamic explanation
        if self.openai_available:
            try:
                dynamic_explanation = self._generate_dynamic_explanation(issue_description, context)
                return {
                    "issue": issue_description,
                    "explanation": dynamic_explanation,
                    "source": "gpt",
                    "confidence": "high"
                }
            except Exception as e:
                return {
                    "issue": issue_description,
                    "explanation": f"Unable to generate explanation: {str(e)}",
                    "source": "error",
                    "confidence": "low"
                }
        
        # Fallback to generic explanation
        return {
            "issue": issue_description,
            "explanation": self._generate_generic_explanation(issue_description),
            "source": "generic",
            "confidence": "medium"
        }
    
    def _extract_key_terms(self, description: str) -> List[str]:
        """Extract key terms from issue description."""
        description_lower = description.lower()
        terms = []
        
        for term in self.static_explanations.keys():
            if term in description_lower:
                terms.append(term)
        
        return terms
    
    def _get_static_explanation(self, key_terms: List[str]) -> Optional[Dict[str, str]]:
        """Get static explanation for known issue types."""
        for term in key_terms:
            if term in self.static_explanations:
                return self.static_explanations[term]
        return None
    
    def _generate_dynamic_explanation(self, issue: str, context: Optional[str]) -> Dict[str, str]:
        """Generate explanation using OpenAI GPT."""
        prompt = f"""Explain this code issue in simple terms:

Issue: {issue}
{f'Context: {context}' if context else ''}

Please explain:
1. Why this is a problem
2. What could go wrong
3. How to fix it
4. Provide a simple code example

Keep the explanation beginner-friendly and practical."""

        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful programming mentor who explains code issues clearly and provides practical solutions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        
        explanation_text = response.choices[0].message.content
        
        # Parse the response into structured format
        return {
            "explanation": explanation_text,
            "generated": True
        }
    
    def _generate_generic_explanation(self, issue: str) -> Dict[str, str]:
        """Generate generic explanation for unknown issues."""
        return {
            "explanation": f"This issue ({issue}) indicates a code quality concern that should be addressed. Consider reviewing the code against best practices and style guidelines.",
            "suggestion": "Check the documentation for your analysis tool or consult coding standards for more specific guidance.",
            "generic": True
        }


# Global instances
gpt_connector = None
issue_explainer = None

def get_gpt_connector():
    """Get or create GPT connector instance."""
    global gpt_connector
    if gpt_connector is None:
        from telemetry import telemetry_collector
        gpt_connector = GPTConnector(telemetry_collector)
    return gpt_connector

def get_issue_explainer():
    """Get or create issue explainer instance."""
    global issue_explainer
    if issue_explainer is None:
        issue_explainer = IssueExplainer()
    return issue_explainer