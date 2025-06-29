"""
Git history analyzer for CodeGuard API.
Analyzes Git repository history to identify bug-prone files and code churn patterns.
"""

import os
import re
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

try:
    from git import Repo, GitCommandError
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False


@dataclass
class CommitAnalysis:
    """Analysis of a single commit."""
    commit_hash: str
    author: str
    date: datetime
    message: str
    files_changed: List[str]
    lines_added: int
    lines_deleted: int
    is_bug_fix: bool
    is_feature: bool
    is_refactor: bool


@dataclass
class FileMetrics:
    """Metrics for a single file."""
    filename: str
    commit_count: int
    bug_fix_count: int
    churn_score: float
    last_modified: datetime
    authors: List[str]
    complexity_trend: str  # 'increasing', 'decreasing', 'stable'
    risk_score: float


class GitHistoryAnalyzer:
    """Analyzes Git repository history to identify patterns and trends."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.repo = None
        self.bug_keywords = [
            'fix', 'bug', 'hotfix', 'patch', 'error', 'issue', 
            'broken', 'crash', 'fail', 'wrong', 'incorrect'
        ]
        self.feature_keywords = [
            'add', 'new', 'feature', 'implement', 'create', 
            'enhance', 'improve', 'upgrade'
        ]
        self.refactor_keywords = [
            'refactor', 'cleanup', 'reorganize', 'restructure',
            'optimize', 'simplify', 'clean'
        ]
        
        if GIT_AVAILABLE:
            try:
                self.repo = Repo(repo_path)
            except Exception:
                self.repo = None
    
    def is_available(self) -> bool:
        """Check if Git analysis is available."""
        return GIT_AVAILABLE and self.repo is not None and not self.repo.bare
    
    def analyze_commit_history(self, days: int = 90) -> List[CommitAnalysis]:
        """Analyze commit history for the specified number of days."""
        if not self.is_available():
            return []
        
        commits = []
        since_date = datetime.now() - timedelta(days=days)
        
        try:
            for commit in self.repo.iter_commits(since=since_date):
                analysis = self._analyze_commit(commit)
                if analysis:
                    commits.append(analysis)
        except Exception:
            return []
        
        return commits
    
    def get_file_metrics(self, days: int = 90) -> Dict[str, FileMetrics]:
        """Get metrics for all files in the repository."""
        commits = self.analyze_commit_history(days)
        file_metrics = defaultdict(lambda: {
            'commit_count': 0,
            'bug_fix_count': 0,
            'total_changes': 0,
            'authors': set(),
            'last_modified': None,
            'commits': []
        })
        
        # Aggregate data per file
        for commit in commits:
            for filename in commit.files_changed:
                metrics = file_metrics[filename]
                metrics['commit_count'] += 1
                metrics['total_changes'] += commit.lines_added + commit.lines_deleted
                metrics['authors'].add(commit.author)
                metrics['commits'].append(commit)
                
                if commit.is_bug_fix:
                    metrics['bug_fix_count'] += 1
                
                if not metrics['last_modified'] or commit.date > metrics['last_modified']:
                    metrics['last_modified'] = commit.date
        
        # Convert to FileMetrics objects
        result = {}
        for filename, data in file_metrics.items():
            if filename.endswith('.py'):  # Focus on Python files
                churn_score = data['total_changes'] / max(data['commit_count'], 1)
                risk_score = self._calculate_risk_score(data)
                complexity_trend = self._analyze_complexity_trend(data['commits'])
                
                result[filename] = FileMetrics(
                    filename=filename,
                    commit_count=data['commit_count'],
                    bug_fix_count=data['bug_fix_count'],
                    churn_score=churn_score,
                    last_modified=data['last_modified'] or datetime.now(),
                    authors=list(data['authors']),
                    complexity_trend=complexity_trend,
                    risk_score=risk_score
                )
        
        return result
    
    def get_bug_prone_files(self, days: int = 90, threshold: float = 0.3) -> List[FileMetrics]:
        """Identify files that are prone to bugs based on history."""
        file_metrics = self.get_file_metrics(days)
        
        bug_prone = []
        for metrics in file_metrics.values():
            # Calculate bug ratio
            bug_ratio = metrics.bug_fix_count / max(metrics.commit_count, 1)
            
            if bug_ratio >= threshold and metrics.commit_count > 2:
                bug_prone.append(metrics)
        
        # Sort by risk score
        bug_prone.sort(key=lambda x: x.risk_score, reverse=True)
        return bug_prone
    
    def get_high_churn_files(self, days: int = 90, threshold: float = 50.0) -> List[FileMetrics]:
        """Identify files with high code churn."""
        file_metrics = self.get_file_metrics(days)
        
        high_churn = []
        for metrics in file_metrics.values():
            if metrics.churn_score >= threshold and metrics.commit_count > 3:
                high_churn.append(metrics)
        
        # Sort by churn score
        high_churn.sort(key=lambda x: x.churn_score, reverse=True)
        return high_churn
    
    def get_repository_trends(self, days: int = 90) -> Dict[str, Any]:
        """Get overall repository trends and statistics."""
        commits = self.analyze_commit_history(days)
        
        if not commits:
            return {}
        
        # Calculate trends
        total_commits = len(commits)
        bug_fixes = sum(1 for c in commits if c.is_bug_fix)
        features = sum(1 for c in commits if c.is_feature)
        refactors = sum(1 for c in commits if c.is_refactor)
        
        # Calculate activity by week
        weekly_activity = defaultdict(int)
        for commit in commits:
            week = commit.date.strftime('%Y-W%U')
            weekly_activity[week] += 1
        
        # Most active authors
        author_activity = defaultdict(int)
        for commit in commits:
            author_activity[commit.author] += 1
        
        top_authors = sorted(author_activity.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'analysis_period_days': days,
            'total_commits': total_commits,
            'bug_fix_commits': bug_fixes,
            'feature_commits': features,
            'refactor_commits': refactors,
            'bug_fix_ratio': bug_fixes / max(total_commits, 1),
            'weekly_activity': dict(weekly_activity),
            'top_authors': top_authors,
            'average_commits_per_day': total_commits / max(days, 1),
            'files_with_most_changes': self._get_most_changed_files(commits)
        }
    
    def _analyze_commit(self, commit) -> Optional[CommitAnalysis]:
        """Analyze a single commit."""
        try:
            message = commit.message.lower()
            
            # Classify commit type
            is_bug_fix = any(keyword in message for keyword in self.bug_keywords)
            is_feature = any(keyword in message for keyword in self.feature_keywords)
            is_refactor = any(keyword in message for keyword in self.refactor_keywords)
            
            # Get changed files
            files_changed = []
            lines_added = 0
            lines_deleted = 0
            
            try:
                # Get stats for this commit
                stats = commit.stats
                for file_path, changes in stats.files.items():
                    files_changed.append(file_path)
                    lines_added += changes.get('insertions', 0)
                    lines_deleted += changes.get('deletions', 0)
            except Exception:
                # Fallback for commits without stats
                pass
            
            return CommitAnalysis(
                commit_hash=commit.hexsha[:8],
                author=commit.author.name,
                date=datetime.fromtimestamp(commit.committed_date),
                message=commit.message.strip(),
                files_changed=files_changed,
                lines_added=lines_added,
                lines_deleted=lines_deleted,
                is_bug_fix=is_bug_fix,
                is_feature=is_feature,
                is_refactor=is_refactor
            )
        except Exception:
            return None
    
    def _calculate_risk_score(self, file_data: Dict) -> float:
        """Calculate risk score for a file based on various factors."""
        commit_count = file_data['commit_count']
        bug_fix_count = file_data['bug_fix_count']
        total_changes = file_data['total_changes']
        author_count = len(file_data['authors'])
        
        # Normalize factors
        bug_ratio = bug_fix_count / max(commit_count, 1)
        churn_score = total_changes / max(commit_count, 1)
        author_factor = min(author_count / 5, 1)  # More authors = higher risk
        
        # Weighted risk score
        risk_score = (
            bug_ratio * 0.4 +
            min(churn_score / 100, 1) * 0.3 +
            author_factor * 0.2 +
            min(commit_count / 20, 1) * 0.1
        )
        
        return min(risk_score, 1.0)
    
    def _analyze_complexity_trend(self, commits: List[CommitAnalysis]) -> str:
        """Analyze if code complexity is increasing, decreasing, or stable."""
        if len(commits) < 3:
            return 'stable'
        
        # Sort commits by date
        sorted_commits = sorted(commits, key=lambda c: c.date)
        
        # Simple heuristic: large additions suggest increasing complexity
        recent_additions = sum(c.lines_added for c in sorted_commits[-3:])
        older_additions = sum(c.lines_added for c in sorted_commits[:3])
        
        if recent_additions > older_additions * 1.5:
            return 'increasing'
        elif recent_additions < older_additions * 0.5:
            return 'decreasing'
        else:
            return 'stable'
    
    def _get_most_changed_files(self, commits: List[CommitAnalysis]) -> List[Dict[str, Any]]:
        """Get files that change most frequently."""
        file_changes = defaultdict(int)
        
        for commit in commits:
            for filename in commit.files_changed:
                if filename.endswith('.py'):
                    file_changes[filename] += 1
        
        most_changed = sorted(file_changes.items(), key=lambda x: x[1], reverse=True)[:10]
        return [{'file': file, 'change_count': count} for file, count in most_changed]


def analyze_git_history(repo_path: str = ".", days: int = 90) -> Dict[str, Any]:
    """
    Analyze Git repository history and return comprehensive metrics.
    
    Args:
        repo_path: Path to Git repository
        days: Number of days to analyze
        
    Returns:
        Dictionary containing historical analysis results
    """
    analyzer = GitHistoryAnalyzer(repo_path)
    
    if not analyzer.is_available():
        return {
            'available': False,
            'error': 'Git repository not available or GitPython not installed'
        }
    
    try:
        trends = analyzer.get_repository_trends(days)
        bug_prone_files = analyzer.get_bug_prone_files(days)
        high_churn_files = analyzer.get_high_churn_files(days)
        
        return {
            'available': True,
            'analysis_period_days': days,
            'repository_trends': trends,
            'bug_prone_files': [
                {
                    'filename': f.filename,
                    'commit_count': f.commit_count,
                    'bug_fix_count': f.bug_fix_count,
                    'risk_score': round(f.risk_score, 3),
                    'complexity_trend': f.complexity_trend
                }
                for f in bug_prone_files[:10]
            ],
            'high_churn_files': [
                {
                    'filename': f.filename,
                    'churn_score': round(f.churn_score, 2),
                    'commit_count': f.commit_count,
                    'authors': f.authors
                }
                for f in high_churn_files[:10]
            ],
            'recommendations': _generate_recommendations(bug_prone_files, high_churn_files)
        }
    except Exception as e:
        return {
            'available': False,
            'error': f'Git analysis failed: {str(e)}'
        }


def _generate_recommendations(bug_prone_files: List[FileMetrics], 
                            high_churn_files: List[FileMetrics]) -> List[str]:
    """Generate actionable recommendations based on Git analysis."""
    recommendations = []
    
    if bug_prone_files:
        top_bug_prone = bug_prone_files[0]
        recommendations.append(
            f"Consider refactoring {top_bug_prone.filename} - it has {top_bug_prone.bug_fix_count} "
            f"bug fixes out of {top_bug_prone.commit_count} commits"
        )
    
    if high_churn_files:
        top_churn = high_churn_files[0]
        recommendations.append(
            f"Review {top_churn.filename} for stability - high churn score of {top_churn.churn_score:.1f}"
        )
    
    if len(high_churn_files) > 3:
        recommendations.append(
            f"Multiple files show high churn ({len(high_churn_files)} files) - "
            "consider establishing coding standards and review processes"
        )
    
    if not recommendations:
        recommendations.append("Repository shows good stability with low bug rates and moderate churn")
    
    return recommendations