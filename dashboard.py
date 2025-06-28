"""
Lightweight analytics dashboard for CodeGuard API metrics.
Provides usage statistics, framework trends, and error patterns.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import Counter, defaultdict


class AnalyticsDashboard:
    """Generates analytics dashboard data for CodeGuard metrics."""
    
    def __init__(self, telemetry_collector):
        self.telemetry = telemetry_collector
    
    def generate_dashboard_data(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive dashboard data."""
        try:
            # Get base metrics
            metrics = self.telemetry.get_usage_metrics(days)
            
            # Generate additional analytics
            dashboard_data = {
                'overview': self._generate_overview(metrics),
                'charts': self._generate_chart_data(metrics),
                'insights': self._generate_insights(metrics),
                'alerts': self._generate_alerts(metrics),
                'period': f"Last {days} days",
                'generated_at': datetime.now().isoformat()
            }
            
            return dashboard_data
            
        except Exception as e:
            return {'error': f'Dashboard generation failed: {str(e)}'}
    
    def _generate_overview(self, metrics: Dict) -> Dict[str, Any]:
        """Generate overview statistics."""
        session_stats = metrics.get('session_statistics', {})
        
        return {
            'total_audits': session_stats.get('total_sessions', 0),
            'files_analyzed': session_stats.get('total_files', 0),
            'issues_found': session_stats.get('total_issues', 0),
            'avg_analysis_time': round(session_stats.get('avg_analysis_time', 0), 2),
            'avg_issues_per_file': round(session_stats.get('avg_issues_per_file', 0), 2),
            'top_framework': self._get_top_framework(metrics.get('framework_usage', {}))
        }
    
    def _generate_chart_data(self, metrics: Dict) -> Dict[str, Any]:
        """Generate data for charts and visualizations."""
        framework_usage = metrics.get('framework_usage', {})
        error_stats = metrics.get('common_errors', {})
        tool_performance = metrics.get('tool_performance', {})
        
        return {
            'framework_distribution': {
                'labels': list(framework_usage.keys()),
                'data': list(framework_usage.values()),
                'total': sum(framework_usage.values())
            },
            'error_types': {
                'labels': list(error_stats.keys())[:10],  # Top 10
                'data': list(error_stats.values())[:10],
                'total': sum(error_stats.values())
            },
            'tool_efficiency': {
                'tools': list(tool_performance.keys()),
                'avg_times': [tool_performance[tool].get('avg_time', 0) for tool in tool_performance],
                'issue_counts': [tool_performance[tool].get('total_issues', 0) for tool in tool_performance]
            }
        }
    
    def _generate_insights(self, metrics: Dict) -> List[Dict[str, str]]:
        """Generate actionable insights from metrics."""
        insights = []
        
        session_stats = metrics.get('session_statistics', {})
        framework_usage = metrics.get('framework_usage', {})
        error_stats = metrics.get('common_errors', {})
        
        # Usage insights
        total_sessions = session_stats.get('total_sessions', 0)
        if total_sessions > 0:
            insights.append({
                'type': 'usage',
                'title': 'API Adoption',
                'message': f'{total_sessions} audit sessions completed with {session_stats.get("total_files", 0)} files analyzed'
            })
        
        # Framework insights
        if framework_usage:
            top_framework = max(framework_usage.items(), key=lambda x: x[1])
            insights.append({
                'type': 'framework',
                'title': 'Popular Framework',
                'message': f'{top_framework[0]} is the most used framework ({top_framework[1]} detections)'
            })
        
        # Error pattern insights
        if error_stats:
            top_error = max(error_stats.items(), key=lambda x: x[1])
            insights.append({
                'type': 'quality',
                'title': 'Common Issue',
                'message': f'Most frequent issue type: {top_error[0]} ({top_error[1]} occurrences)'
            })
        
        # Performance insights
        avg_time = session_stats.get('avg_analysis_time', 0)
        if avg_time > 0:
            if avg_time < 1000:  # Less than 1 second
                insights.append({
                    'type': 'performance',
                    'title': 'Fast Analysis',
                    'message': f'Average analysis time: {avg_time:.0f}ms - excellent performance'
                })
            elif avg_time > 5000:  # More than 5 seconds
                insights.append({
                    'type': 'performance',
                    'title': 'Slow Analysis',
                    'message': f'Average analysis time: {avg_time:.0f}ms - consider optimization'
                })
        
        return insights
    
    def _generate_alerts(self, metrics: Dict) -> List[Dict[str, str]]:
        """Generate system alerts and warnings."""
        alerts = []
        
        session_stats = metrics.get('session_statistics', {})
        
        # Check for performance issues
        avg_time = session_stats.get('avg_analysis_time', 0)
        if avg_time > 10000:  # More than 10 seconds
            alerts.append({
                'level': 'warning',
                'title': 'Performance Alert',
                'message': f'Analysis time averaging {avg_time:.0f}ms - investigate bottlenecks'
            })
        
        # Check for high error rates
        avg_issues = session_stats.get('avg_issues_per_file', 0)
        if avg_issues > 20:
            alerts.append({
                'level': 'info',
                'title': 'High Issue Count',
                'message': f'Averaging {avg_issues:.1f} issues per file - users may need coding guidelines'
            })
        
        return alerts
    
    def _get_top_framework(self, framework_usage: Dict) -> str:
        """Get the most commonly used framework."""
        if not framework_usage:
            return 'Unknown'
        return max(framework_usage.items(), key=lambda x: x[1])[0]
    
    def export_report(self, data: Dict, format: str = 'json') -> str:
        """Export dashboard data in various formats."""
        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        
        elif format.lower() == 'markdown':
            return self._generate_markdown_report(data)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_markdown_report(self, data: Dict) -> str:
        """Generate a Markdown report from dashboard data."""
        overview = data.get('overview', {})
        insights = data.get('insights', [])
        alerts = data.get('alerts', [])
        
        md_report = f"""# CodeGuard Analytics Report
Generated: {data.get('generated_at', 'Unknown')}
Period: {data.get('period', 'Unknown')}

## Overview
- **Total Audits**: {overview.get('total_audits', 0)}
- **Files Analyzed**: {overview.get('files_analyzed', 0)}
- **Issues Found**: {overview.get('issues_found', 0)}
- **Average Analysis Time**: {overview.get('avg_analysis_time', 0)}ms
- **Average Issues per File**: {overview.get('avg_issues_per_file', 0)}
- **Top Framework**: {overview.get('top_framework', 'Unknown')}

## Key Insights
"""
        
        for insight in insights:
            md_report += f"- **{insight.get('title', 'Unknown')}**: {insight.get('message', 'No details')}\n"
        
        if alerts:
            md_report += "\n## Alerts\n"
            for alert in alerts:
                level_emoji = {'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}.get(alert.get('level', 'info'), 'ℹ️')
                md_report += f"{level_emoji} **{alert.get('title', 'Alert')}**: {alert.get('message', 'No details')}\n"
        
        return md_report


# Initialize dashboard
dashboard = None

def get_dashboard():
    """Get or create dashboard instance."""
    global dashboard
    if dashboard is None:
        from telemetry import telemetry_collector
        dashboard = AnalyticsDashboard(telemetry_collector)
    return dashboard