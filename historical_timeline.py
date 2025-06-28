"""
Historical Audit Timeline System for CodeGuard API.
Tracks issue trends over time and provides timeline visualization data.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class TimelineDataPoint:
    """Represents a single data point in the audit timeline."""
    date: str
    total_sessions: int
    total_issues: int
    avg_issues_per_file: float
    top_error_type: str
    framework_breakdown: Dict[str, int]
    severity_breakdown: Dict[str, int]


class HistoricalTimelineGenerator:
    """Generates historical timeline data for audit trends."""
    
    def __init__(self, telemetry_collector):
        self.telemetry = telemetry_collector
    
    def generate_timeline(self, days: int = 30, granularity: str = "daily") -> Dict[str, Any]:
        """Generate historical timeline data for the specified period."""
        try:
            if granularity == "daily":
                return self._generate_daily_timeline(days)
            elif granularity == "weekly":
                return self._generate_weekly_timeline(days)
            else:
                return {"error": f"Unsupported granularity: {granularity}"}
        except Exception as e:
            return {"error": f"Timeline generation failed: {str(e)}"}
    
    def _generate_daily_timeline(self, days: int) -> Dict[str, Any]:
        """Generate daily timeline data."""
        timeline_data = []
        trends = {"improving": 0, "worsening": 0, "stable": 0}
        
        # Use in-memory data for demonstration
        if hasattr(self.telemetry, 'memory_sessions'):
            sessions = self.telemetry.memory_sessions
            
            # Group sessions by date
            daily_data = defaultdict(list)
            for session in sessions:
                date_key = session.timestamp.strftime('%Y-%m-%d')
                daily_data[date_key].append(session)
            
            # Generate timeline points
            for i in range(days):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                day_sessions = daily_data.get(date, [])
                
                if day_sessions:
                    total_issues = sum(s.total_issues for s in day_sessions)
                    total_files = sum(s.file_count for s in day_sessions)
                    avg_issues = total_issues / total_files if total_files > 0 else 0
                    
                    # Aggregate error types
                    all_errors = {}
                    frameworks = defaultdict(int)
                    severities = defaultdict(int)
                    
                    for session in day_sessions:
                        for error_type, count in session.error_types.items():
                            all_errors[error_type] = all_errors.get(error_type, 0) + count
                        
                        if session.framework_detected:
                            frameworks[session.framework_detected] += 1
                        
                        for severity, count in session.severity_breakdown.items():
                            severities[severity] += count
                    
                    top_error = max(all_errors.items(), key=lambda x: x[1])[0] if all_errors else "none"
                    
                    timeline_data.append(TimelineDataPoint(
                        date=date,
                        total_sessions=len(day_sessions),
                        total_issues=total_issues,
                        avg_issues_per_file=round(avg_issues, 2),
                        top_error_type=top_error,
                        framework_breakdown=dict(frameworks),
                        severity_breakdown=dict(severities)
                    ))
                else:
                    # No data for this day
                    timeline_data.append(TimelineDataPoint(
                        date=date,
                        total_sessions=0,
                        total_issues=0,
                        avg_issues_per_file=0.0,
                        top_error_type="none",
                        framework_breakdown={},
                        severity_breakdown={}
                    ))
        else:
            # Generate sample timeline data for demonstration
            timeline_data = self._generate_sample_timeline(days)
        
        # Calculate trends
        if len(timeline_data) >= 2:
            trends = self._calculate_trends(timeline_data)
        
        return {
            "timeline": [self._datapoint_to_dict(dp) for dp in reversed(timeline_data)],
            "trends": trends,
            "summary": self._generate_timeline_summary(timeline_data),
            "period": f"Last {days} days",
            "granularity": "daily",
            "generated_at": datetime.now().isoformat()
        }
    
    def _generate_sample_timeline(self, days: int) -> List[TimelineDataPoint]:
        """Generate sample timeline data for demonstration."""
        import random
        
        timeline_data = []
        base_issues = 25
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            
            # Simulate improving trend over time
            trend_factor = 1 - (i * 0.02)  # Gradual improvement
            issues = max(10, int(base_issues * trend_factor + random.randint(-5, 5)))
            
            timeline_data.append(TimelineDataPoint(
                date=date,
                total_sessions=random.randint(1, 5),
                total_issues=issues,
                avg_issues_per_file=round(issues / random.randint(1, 3), 2),
                top_error_type=random.choice(["style", "error", "best_practice", "complexity"]),
                framework_breakdown={"pytorch": random.randint(0, 3), "tensorflow": random.randint(0, 2)},
                severity_breakdown={"error": random.randint(0, 5), "warning": random.randint(5, 15), "info": random.randint(0, 8)}
            ))
        
        return timeline_data
    
    def _calculate_trends(self, timeline_data: List[TimelineDataPoint]) -> Dict[str, int]:
        """Calculate improvement/worsening trends from timeline data."""
        trends = {"improving": 0, "worsening": 0, "stable": 0}
        
        for i in range(1, len(timeline_data)):
            current = timeline_data[i].avg_issues_per_file
            previous = timeline_data[i-1].avg_issues_per_file
            
            if current < previous * 0.9:  # 10% improvement
                trends["improving"] += 1
            elif current > previous * 1.1:  # 10% worsening
                trends["worsening"] += 1
            else:
                trends["stable"] += 1
        
        return trends
    
    def _generate_timeline_summary(self, timeline_data: List[TimelineDataPoint]) -> Dict[str, Any]:
        """Generate summary statistics for the timeline."""
        if not timeline_data:
            return {}
        
        total_sessions = sum(dp.total_sessions for dp in timeline_data)
        total_issues = sum(dp.total_issues for dp in timeline_data)
        avg_issues = sum(dp.avg_issues_per_file for dp in timeline_data) / len(timeline_data)
        
        # Find best and worst days
        best_day = min(timeline_data, key=lambda x: x.avg_issues_per_file)
        worst_day = max(timeline_data, key=lambda x: x.avg_issues_per_file)
        
        # Framework usage over time
        all_frameworks = set()
        for dp in timeline_data:
            all_frameworks.update(dp.framework_breakdown.keys())
        
        framework_totals = {fw: sum(dp.framework_breakdown.get(fw, 0) for dp in timeline_data) 
                           for fw in all_frameworks}
        
        return {
            "total_audit_sessions": total_sessions,
            "total_issues_found": total_issues,
            "average_issues_per_file": round(avg_issues, 2),
            "best_day": {
                "date": best_day.date,
                "avg_issues": best_day.avg_issues_per_file
            },
            "worst_day": {
                "date": worst_day.date,
                "avg_issues": worst_day.avg_issues_per_file
            },
            "framework_usage": framework_totals,
            "data_points": len(timeline_data)
        }
    
    def _datapoint_to_dict(self, dp: TimelineDataPoint) -> Dict[str, Any]:
        """Convert TimelineDataPoint to dictionary."""
        return {
            "date": dp.date,
            "total_sessions": dp.total_sessions,
            "total_issues": dp.total_issues,
            "avg_issues_per_file": dp.avg_issues_per_file,
            "top_error_type": dp.top_error_type,
            "framework_breakdown": dp.framework_breakdown,
            "severity_breakdown": dp.severity_breakdown
        }
    
    def get_framework_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get framework usage trends over time."""
        timeline = self.generate_timeline(days)
        
        if "error" in timeline:
            return timeline
        
        framework_trends = defaultdict(list)
        
        for datapoint in timeline["timeline"]:
            date = datapoint["date"]
            for framework, count in datapoint["framework_breakdown"].items():
                framework_trends[framework].append({
                    "date": date,
                    "count": count
                })
        
        return {
            "framework_trends": dict(framework_trends),
            "period": f"Last {days} days",
            "generated_at": datetime.now().isoformat()
        }
    
    def get_error_pattern_evolution(self, days: int = 30) -> Dict[str, Any]:
        """Get how error patterns have evolved over time."""
        timeline = self.generate_timeline(days)
        
        if "error" in timeline:
            return timeline
        
        error_evolution = defaultdict(list)
        
        for datapoint in timeline["timeline"]:
            date = datapoint["date"]
            error_evolution[datapoint["top_error_type"]].append({
                "date": date,
                "frequency": 1  # Simplified frequency tracking
            })
        
        return {
            "error_evolution": dict(error_evolution),
            "period": f"Last {days} days",
            "generated_at": datetime.now().isoformat()
        }


# Global timeline generator
timeline_generator = None

def get_timeline_generator():
    """Get or create timeline generator instance."""
    global timeline_generator
    if timeline_generator is None:
        from telemetry import telemetry_collector
        timeline_generator = HistoricalTimelineGenerator(telemetry_collector)
    return timeline_generator