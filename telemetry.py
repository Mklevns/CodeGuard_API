"""
Telemetry and usage metrics system for CodeGuard API.
Tracks audit sessions, error patterns, framework usage, and performance metrics.
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from models import AuditRequest, AuditResponse, Issue


@dataclass
class AuditSession:
    """Represents a complete audit session for telemetry tracking."""
    session_id: str
    timestamp: datetime
    file_count: int
    total_issues: int
    analysis_time_ms: float
    framework_detected: Optional[str]
    error_types: Dict[str, int]
    severity_breakdown: Dict[str, int]
    tools_used: List[str]
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None


class TelemetryCollector:
    """Collects and stores telemetry data for CodeGuard audit sessions."""
    
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.logger = logging.getLogger(__name__)
        self.db_available = False
        try:
            self._init_database()
            self.db_available = True
        except Exception as e:
            self.logger.warning(f"Database unavailable, using in-memory telemetry: {e}")
            self._init_memory_storage()
    
    def _init_database(self):
        """Initialize database tables for telemetry data."""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    # Create audit_sessions table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS audit_sessions (
                            id SERIAL PRIMARY KEY,
                            session_id VARCHAR(36) UNIQUE NOT NULL,
                            timestamp TIMESTAMP NOT NULL,
                            file_count INTEGER NOT NULL,
                            total_issues INTEGER NOT NULL,
                            analysis_time_ms FLOAT NOT NULL,
                            framework_detected VARCHAR(50),
                            error_types JSONB,
                            severity_breakdown JSONB,
                            tools_used JSONB,
                            user_agent TEXT,
                            ip_address INET,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Create error_patterns table for detailed issue tracking
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS error_patterns (
                            id SERIAL PRIMARY KEY,
                            session_id VARCHAR(36) REFERENCES audit_sessions(session_id),
                            error_code VARCHAR(20),
                            error_type VARCHAR(50),
                            source_tool VARCHAR(20),
                            severity VARCHAR(20),
                            filename VARCHAR(255),
                            line_number INTEGER,
                            description TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Create performance_metrics table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS performance_metrics (
                            id SERIAL PRIMARY KEY,
                            session_id VARCHAR(36) REFERENCES audit_sessions(session_id),
                            tool_name VARCHAR(20),
                            execution_time_ms FLOAT,
                            issues_found INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Create framework_usage table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS framework_usage (
                            id SERIAL PRIMARY KEY,
                            date DATE DEFAULT CURRENT_DATE,
                            framework VARCHAR(50),
                            usage_count INTEGER DEFAULT 1,
                            UNIQUE(date, framework)
                        )
                    """)
                    
                    conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize telemetry database: {e}")
            raise
    
    def _init_memory_storage(self):
        """Initialize in-memory storage for telemetry when database is unavailable."""
        self.memory_sessions = []
        self.memory_errors = []
        self.memory_frameworks = defaultdict(int)
    
    def record_audit_session(self, session: AuditSession):
        """Record a complete audit session in the database or memory."""
        if self.db_available:
            try:
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO audit_sessions 
                            (session_id, timestamp, file_count, total_issues, analysis_time_ms,
                             framework_detected, error_types, severity_breakdown, tools_used,
                             user_agent, ip_address)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (session_id) DO NOTHING
                        """, (
                            session.session_id,
                            session.timestamp,
                            session.file_count,
                            session.total_issues,
                            session.analysis_time_ms,
                            session.framework_detected,
                            json.dumps(session.error_types),
                            json.dumps(session.severity_breakdown),
                            json.dumps(session.tools_used),
                            session.user_agent,
                            session.ip_address
                        ))
                        conn.commit()
            except Exception as e:
                self.logger.error(f"Failed to record audit session: {e}")
        else:
            # Store in memory
            self.memory_sessions.append(session)
    
    def record_error_patterns(self, session_id: str, issues: List[Issue]):
        """Record detailed error patterns for analysis."""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    for issue in issues:
                        # Extract error code from description
                        error_code = self._extract_error_code(issue.description)
                        
                        cur.execute("""
                            INSERT INTO error_patterns 
                            (session_id, error_code, error_type, source_tool, severity,
                             filename, line_number, description)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            session_id,
                            error_code,
                            issue.type,
                            issue.source,
                            issue.severity,
                            issue.filename,
                            issue.line,
                            issue.description
                        ))
                    conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to record error patterns: {e}")
    
    def record_framework_usage(self, framework: str):
        """Record framework usage statistics."""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO framework_usage (framework)
                        VALUES (%s)
                        ON CONFLICT (date, framework) 
                        DO UPDATE SET usage_count = framework_usage.usage_count + 1
                    """, (framework,))
                    conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to record framework usage: {e}")
    
    def _extract_error_code(self, description: str) -> Optional[str]:
        """Extract error codes like F401, E302 from issue descriptions."""
        match = re.search(r'([A-Z]\d{3,4})', description)
        return match.group(1) if match else None
    
    def get_usage_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive usage metrics for the specified time period."""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    start_date = datetime.now() - timedelta(days=days)
                    
                    # Get session statistics
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_sessions,
                            SUM(file_count) as total_files,
                            SUM(total_issues) as total_issues,
                            AVG(analysis_time_ms) as avg_analysis_time,
                            AVG(total_issues::float / file_count) as avg_issues_per_file
                        FROM audit_sessions 
                        WHERE timestamp >= %s
                    """, (start_date,))
                    
                    result = cur.fetchone()
                    session_stats = dict(result) if result else {
                        'total_sessions': 0, 'total_files': 0, 'total_issues': 0,
                        'avg_analysis_time': 0, 'avg_issues_per_file': 0
                    }
                    
                    # Get framework usage
                    cur.execute("""
                        SELECT framework, SUM(usage_count) as count
                        FROM framework_usage 
                        WHERE date >= %s
                        GROUP BY framework 
                        ORDER BY count DESC
                    """, (start_date.date(),))
                    
                    framework_stats = {row['framework']: row['count'] for row in cur.fetchall()}
                    
                    # Get most common error types
                    cur.execute("""
                        SELECT error_type, COUNT(*) as count
                        FROM error_patterns ep
                        JOIN audit_sessions asm ON ep.session_id = asm.session_id
                        WHERE asm.timestamp >= %s
                        GROUP BY error_type 
                        ORDER BY count DESC 
                        LIMIT 10
                    """, (start_date,))
                    
                    error_stats = {row['error_type']: row['count'] for row in cur.fetchall()}
                    
                    # Get tool performance
                    cur.execute("""
                        SELECT tool_name, 
                               AVG(execution_time_ms) as avg_time,
                               SUM(issues_found) as total_issues
                        FROM performance_metrics pm
                        JOIN audit_sessions asm ON pm.session_id = asm.session_id
                        WHERE asm.timestamp >= %s
                        GROUP BY tool_name
                    """, (start_date,))
                    
                    tool_stats = {
                        row['tool_name']: {
                            'avg_time': float(row['avg_time']) if row['avg_time'] else 0,
                            'total_issues': row['total_issues']
                        } for row in cur.fetchall()
                    }
                    
                    return {
                        'period_days': days,
                        'session_statistics': session_stats,
                        'framework_usage': framework_stats,
                        'common_errors': error_stats,
                        'tool_performance': tool_stats,
                        'generated_at': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get usage metrics: {e}")
            return {"error": str(e)}


class MetricsAnalyzer:
    """Analyzes audit requests and responses to extract telemetry data."""
    
    def __init__(self):
        self.framework_patterns = {
            'pytorch': [r'import torch', r'from torch', r'torch\.', r'nn\.Module'],
            'tensorflow': [r'import tensorflow', r'from tensorflow', r'tf\.', r'keras'],
            'jax': [r'import jax', r'from jax', r'jax\.'],
            'scikit-learn': [r'from sklearn', r'import sklearn', r'sklearn\.'],
            'huggingface': [r'from transformers', r'import transformers', r'from datasets'],
            'gym': [r'import gym', r'gym\.make', r'env\.step', r'env\.reset'],
            'stable-baselines': [r'stable_baselines', r'sb3', r'from stable_baselines'],
            'wandb': [r'import wandb', r'wandb\.'],
            'mlflow': [r'import mlflow', r'mlflow\.']
        }
    
    def analyze_request(self, request: AuditRequest) -> Dict[str, Any]:
        """Analyze audit request to extract framework and patterns."""
        all_content = '\n'.join([file.content for file in request.files])
        
        # Detect frameworks
        detected_frameworks = []
        for framework, patterns in self.framework_patterns.items():
            for pattern in patterns:
                if re.search(pattern, all_content, re.IGNORECASE):
                    detected_frameworks.append(framework)
                    break
        
        # Get primary framework (first detected or most specific)
        primary_framework = None
        if detected_frameworks:
            # Prioritize ML frameworks over general ones
            ml_frameworks = ['pytorch', 'tensorflow', 'jax', 'huggingface']
            for fw in ml_frameworks:
                if fw in detected_frameworks:
                    primary_framework = fw
                    break
            if not primary_framework:
                primary_framework = detected_frameworks[0]
        
        return {
            'file_count': len(request.files),
            'total_lines': sum(len(file.content.splitlines()) for file in request.files),
            'detected_frameworks': detected_frameworks,
            'primary_framework': primary_framework,
            'has_options': request.options is not None,
            'analysis_level': request.options.level if request.options else 'default'
        }
    
    def analyze_response(self, response: AuditResponse) -> Dict[str, Any]:
        """Analyze audit response to extract issue patterns and metrics."""
        # Count issues by type and severity
        issue_types = Counter(issue.type for issue in response.issues)
        severities = Counter(issue.severity for issue in response.issues)
        sources = Counter(issue.source for issue in response.issues)
        
        # Extract error codes
        error_codes = []
        for issue in response.issues:
            code = re.search(r'([A-Z]\d{3,4})', issue.description)
            if code:
                error_codes.append(code.group(1))
        
        error_code_counts = Counter(error_codes)
        
        # Calculate fix statistics
        auto_fixable_count = sum(1 for fix in response.fixes if fix.auto_fixable)
        
        return {
            'total_issues': len(response.issues),
            'total_fixes': len(response.fixes),
            'auto_fixable_fixes': auto_fixable_count,
            'issue_types': dict(issue_types),
            'severities': dict(severities),
            'sources': dict(sources),
            'error_codes': dict(error_code_counts),
            'has_diffs': sum(1 for fix in response.fixes if fix.diff),
            'has_replacements': sum(1 for fix in response.fixes if fix.replacement_code)
        }
    
    def create_session(self, session_id: str, request_analysis: Dict, response_analysis: Dict, 
                      analysis_time: float, user_agent: str = None, ip_address: str = None) -> AuditSession:
        """Create an AuditSession object from analysis data."""
        return AuditSession(
            session_id=session_id,
            timestamp=datetime.now(),
            file_count=request_analysis['file_count'],
            total_issues=response_analysis['total_issues'],
            analysis_time_ms=analysis_time,
            framework_detected=request_analysis.get('primary_framework'),
            error_types=response_analysis['issue_types'],
            severity_breakdown=response_analysis['severities'],
            tools_used=list(response_analysis['sources'].keys()),
            user_agent=user_agent,
            ip_address=ip_address
        )


# Global telemetry collector instance
telemetry_collector = TelemetryCollector()
metrics_analyzer = MetricsAnalyzer()