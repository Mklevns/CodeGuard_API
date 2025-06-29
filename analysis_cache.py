"""
Analysis Results Caching System for CodeGuard API.
Implements performance improvements by caching analysis results for unchanged files.
"""

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from models import Issue, Fix, AuditResponse


class AnalysisCache:
    """Caches analysis results to improve performance for large projects."""
    
    def __init__(self, cache_dir: str = ".codeguard_cache", ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_file_hash(self, filename: str, content: str) -> str:
        """Generate hash for file content."""
        file_data = f"{filename}:{content}"
        return hashlib.sha256(file_data.encode()).hexdigest()
    
    def _get_cache_path(self, file_hash: str) -> str:
        """Get cache file path for a given hash."""
        return os.path.join(self.cache_dir, f"{file_hash}.json")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache entry is still valid based on TTL."""
        if not os.path.exists(cache_path):
            return False
        
        file_age = time.time() - os.path.getmtime(cache_path)
        return file_age < self.ttl_seconds
    
    def get_cached_result(self, filename: str, content: str) -> Optional[Tuple[List[Issue], List[Fix]]]:
        """Get cached analysis result for a file."""
        file_hash = self._get_file_hash(filename, content)
        cache_path = self._get_cache_path(file_hash)
        
        if not self._is_cache_valid(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Reconstruct Issue and Fix objects
            issues = [Issue(**issue_data) for issue_data in cached_data['issues']]
            fixes = [Fix(**fix_data) for fix_data in cached_data['fixes']]
            
            return issues, fixes
        
        except (json.JSONDecodeError, KeyError, TypeError):
            # Invalid cache entry, remove it
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return None
    
    def cache_result(self, filename: str, content: str, issues: List[Issue], fixes: List[Fix]):
        """Cache analysis result for a file."""
        file_hash = self._get_file_hash(filename, content)
        cache_path = self._get_cache_path(file_hash)
        
        # Convert to serializable format
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'issues': [issue.dict() for issue in issues],
            'fixes': [fix.dict() for fix in fixes]
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            # Silently fail if caching doesn't work
            pass
    
    def clear_cache(self):
        """Clear all cached results."""
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, filename)
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not os.path.exists(self.cache_dir):
            return {'entries': 0, 'size_mb': 0, 'valid_entries': 0}
        
        total_files = 0
        total_size = 0
        valid_entries = 0
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.cache_dir, filename)
                total_files += 1
                total_size += os.path.getsize(file_path)
                
                if self._is_cache_valid(file_path):
                    valid_entries += 1
        
        return {
            'entries': total_files,
            'size_mb': round(total_size / (1024 * 1024), 2),
            'valid_entries': valid_entries,
            'cache_dir': self.cache_dir,
            'ttl_hours': self.ttl_seconds / 3600
        }


class ProjectCache:
    """Caches results for entire project analysis."""
    
    def __init__(self, cache_dir: str = ".codeguard_cache"):
        self.cache_dir = cache_dir
        self.project_cache_file = os.path.join(cache_dir, "project_cache.json")
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_project_hash(self, files: List[Dict[str, str]]) -> str:
        """Generate hash for entire project."""
        project_data = json.dumps(sorted(files, key=lambda x: x['filename']), sort_keys=True)
        return hashlib.sha256(project_data.encode()).hexdigest()
    
    def get_cached_project_result(self, files: List[Dict[str, str]]) -> Optional[AuditResponse]:
        """Get cached project analysis result."""
        project_hash = self._get_project_hash(files)
        
        if not os.path.exists(self.project_cache_file):
            return None
        
        try:
            with open(self.project_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            if cache_data.get('project_hash') != project_hash:
                return None
            
            # Check if cache is still valid (1 hour for project cache)
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > timedelta(hours=1):
                return None
            
            # Reconstruct AuditResponse
            response_data = cache_data['response']
            issues = [Issue(**issue_data) for issue_data in response_data['issues']]
            fixes = [Fix(**fix_data) for fix_data in response_data['fixes']]
            
            return AuditResponse(
                summary=response_data['summary'],
                issues=issues,
                fixes=fixes
            )
        
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    
    def cache_project_result(self, files: List[Dict[str, str]], response: AuditResponse):
        """Cache project analysis result."""
        project_hash = self._get_project_hash(files)
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'project_hash': project_hash,
            'response': {
                'summary': response.summary,
                'issues': [issue.dict() for issue in response.issues],
                'fixes': [fix.dict() for fix in response.fixes]
            }
        }
        
        try:
            with open(self.project_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass


# Global cache instances
_file_cache = None
_project_cache = None


def get_file_cache() -> AnalysisCache:
    """Get or create file cache instance."""
    global _file_cache
    if _file_cache is None:
        _file_cache = AnalysisCache()
    return _file_cache


def get_project_cache() -> ProjectCache:
    """Get or create project cache instance."""
    global _project_cache
    if _project_cache is None:
        _project_cache = ProjectCache()
    return _project_cache