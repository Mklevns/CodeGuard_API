"""
GitHub Repository Context Provider for CodeGuard API.
Fetches repository information to provide better context for AI code improvements.
"""

import os
import json
import logging
import requests
import base64
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import re

logger = logging.getLogger(__name__)

@dataclass
class RepositoryInfo:
    """Repository information for AI context."""
    owner: str
    name: str
    description: str
    language: str
    topics: List[str]
    readme_content: str
    file_structure: Dict[str, Any]
    package_files: Dict[str, str]  # filename -> content
    dependencies: List[str]
    framework: str

@dataclass
class FileContent:
    """File content with metadata."""
    path: str
    content: str
    size: int
    type: str  # file, dir
    language: Optional[str] = None

class GitHubRepoContextProvider:
    """Provides GitHub repository context for enhanced AI code improvements."""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.base_url = "https://api.github.com"
        self.session = requests.Session()
        
        if self.github_token:
            self.session.headers.update({
                'Authorization': f'Bearer {self.github_token}',
                'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28',
                'User-Agent': 'CodeGuard-API/1.0'
            })
    
    def extract_repo_info_from_url(self, repo_url: str) -> Tuple[str, str]:
        """Extract owner and repo name from GitHub URL."""
        if not repo_url:
            return "", ""
        
        # Handle various GitHub URL formats
        patterns = [
            r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$',
            r'github\.com/([^/]+)/([^/]+)/.*',
            r'^([^/]+)/([^/]+)$'  # owner/repo format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, repo_url)
            if match:
                return match.group(1), match.group(2)
        
        return "", ""
    
    def get_repository_info(self, repo_url: str) -> Optional[RepositoryInfo]:
        """Fetch comprehensive repository information."""
        try:
            owner, repo_name = self.extract_repo_info_from_url(repo_url)
            if not owner or not repo_name:
                logger.warning(f"Could not extract owner/repo from URL: {repo_url}")
                return None
            
            # Get basic repository info
            repo_info = self._get_repo_basic_info(owner, repo_name)
            if not repo_info:
                return None
            
            # Get README content
            readme_content = self._get_readme_content(owner, repo_name)
            
            # Get file structure
            file_structure = self._get_file_structure(owner, repo_name)
            
            # Get package files (requirements.txt, package.json, pyproject.toml, etc.)
            package_files = self._get_package_files(owner, repo_name)
            
            # Extract dependencies and framework info
            dependencies, framework = self._analyze_dependencies(package_files)
            
            return RepositoryInfo(
                owner=owner,
                name=repo_name,
                description=repo_info.get('description', ''),
                language=repo_info.get('language', ''),
                topics=repo_info.get('topics', []),
                readme_content=readme_content,
                file_structure=file_structure,
                package_files=package_files,
                dependencies=dependencies,
                framework=framework
            )
            
        except Exception as e:
            logger.error(f"Error fetching repository info: {e}")
            return None
    
    def _get_repo_basic_info(self, owner: str, repo_name: str) -> Optional[Dict]:
        """Get basic repository information with enhanced error handling."""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo_name}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 404:
                logger.warning(f"Repository {owner}/{repo_name} not found")
                return None
            elif response.status_code == 403:
                logger.warning(f"Access forbidden to {owner}/{repo_name} - may be private or rate limited")
                return None
            elif response.status_code == 401:
                logger.warning("GitHub authentication failed - check API token")
                return None
            
            response.raise_for_status()
            repo_data = response.json()
            
            # Check rate limits
            remaining = response.headers.get('X-RateLimit-Remaining', '0')
            if int(remaining) < 10:
                logger.warning(f"GitHub API rate limit low: {remaining} requests remaining")
            
            return repo_data
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching repo info for {owner}/{repo_name}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching repo basic info: {e}")
            return None
    
    def _get_readme_content(self, owner: str, repo_name: str) -> str:
        """Get README content."""
        try:
            # Try common README file names
            readme_files = ['README.md', 'README.rst', 'README.txt', 'README']
            
            for readme_file in readme_files:
                url = f"{self.base_url}/repos/{owner}/{repo_name}/contents/{readme_file}"
                response = self.session.get(url)
                
                if response.status_code == 200:
                    content = response.json()
                    if content.get('encoding') == 'base64':
                        return base64.b64decode(content['content']).decode('utf-8', errors='ignore')
            
            return ""
            
        except Exception as e:
            logger.error(f"Error fetching README: {e}")
            return ""
    
    def _get_file_structure(self, owner: str, repo_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get repository file structure using GitHub Trees API."""
        try:
            # Use Git Trees API for more efficient tree traversal
            url = f"{self.base_url}/repos/{owner}/{repo_name}/git/trees/HEAD"
            response = self.session.get(url, params={'recursive': '1'})
            
            if response.status_code != 200:
                # Fallback to contents API for root directory
                return self._get_contents_fallback(owner, repo_name)
            
            tree_data = response.json()
            structure = {}
            
            # Build hierarchical structure from flat tree
            for item in tree_data.get('tree', []):
                path_parts = item['path'].split('/')
                if len(path_parts) > max_depth:
                    continue
                
                current = structure
                for i, part in enumerate(path_parts):
                    if i == len(path_parts) - 1:
                        # Leaf node
                        current[part] = {
                            'type': 'file' if item['type'] == 'blob' else 'directory',
                            'size': item.get('size', 0) if item['type'] == 'blob' else None,
                            'sha': item.get('sha')
                        }
                    else:
                        # Directory node
                        if part not in current:
                            current[part] = {
                                'type': 'directory',
                                'children': {}
                            }
                        current = current[part].setdefault('children', {})
            
            return structure
            
        except Exception as e:
            logger.error(f"Error fetching file structure with Trees API: {e}")
            return self._get_contents_fallback(owner, repo_name)
    
    def _get_contents_fallback(self, owner: str, repo_name: str) -> Dict[str, Any]:
        """Fallback method using Contents API."""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo_name}/contents/"
            response = self.session.get(url)
            
            if response.status_code != 200:
                return {}
            
            contents = response.json()
            if not isinstance(contents, list):
                return {}
            
            structure = {}
            for item in contents[:15]:  # Limit items for performance
                name = item['name']
                structure[name] = {
                    'type': 'file' if item['type'] == 'file' else 'directory',
                    'size': item.get('size', 0) if item['type'] == 'file' else None
                }
            
            return structure
            
        except Exception as e:
            logger.error(f"Error with contents fallback: {e}")
            return {}
    
    def _get_package_files(self, owner: str, repo_name: str) -> Dict[str, str]:
        """Get package/dependency files content."""
        package_files = {}
        
        # Common package files to fetch
        files_to_fetch = [
            'requirements.txt',
            'pyproject.toml', 
            'setup.py',
            'package.json',
            'Pipfile',
            'environment.yml',
            'conda.yml',
            'setup.cfg',
            'poetry.lock',
            'Dockerfile',
            '.github/workflows/main.yml',
            '.github/workflows/ci.yml'
        ]
        
        for file_path in files_to_fetch:
            content = self._get_file_content(owner, repo_name, file_path)
            if content:
                package_files[file_path] = content
        
        return package_files
    
    def _get_file_content(self, owner: str, repo_name: str, file_path: str) -> Optional[str]:
        """Get content of a specific file."""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo_name}/contents/{file_path}"
            response = self.session.get(url)
            
            if response.status_code != 200:
                return None
            
            content = response.json()
            if content.get('encoding') == 'base64':
                return base64.b64decode(content['content']).decode('utf-8', errors='ignore')
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not fetch {file_path}: {e}")
            return None
    
    def get_python_files(self, repo_url: str, max_files: int = 50) -> List[Dict[str, Any]]:
        """Get list of Python files from repository with their metadata."""
        try:
            owner, repo_name = self.extract_repo_info_from_url(repo_url)
            if not owner or not repo_name:
                return []
            
            # Use Git Trees API to get all files recursively
            url = f"{self.base_url}/repos/{owner}/{repo_name}/git/trees/HEAD"
            response = self.session.get(url, params={'recursive': '1'})
            
            if response.status_code != 200:
                return []
            
            tree_data = response.json()
            python_files = []
            
            for item in tree_data.get('tree', []):
                if item['type'] == 'blob' and item['path'].endswith('.py'):
                    # Skip common non-essential files
                    if any(skip in item['path'] for skip in ['__pycache__', '.git', 'test_', 'tests/', 'build/', 'dist/']):
                        continue
                    
                    python_files.append({
                        'path': item['path'],
                        'filename': item['path'].split('/')[-1],
                        'directory': '/'.join(item['path'].split('/')[:-1]) if '/' in item['path'] else '',
                        'size': item.get('size', 0),
                        'sha': item.get('sha', '')
                    })
                    
                    if len(python_files) >= max_files:
                        break
            
            # Sort by directory structure and filename
            python_files.sort(key=lambda x: (x['directory'], x['filename']))
            return python_files
            
        except Exception as e:
            logger.error(f"Error fetching Python files: {e}")
            return []
    
    def get_file_content_by_path(self, repo_url: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Get content of a specific file by its path in the repository."""
        try:
            owner, repo_name = self.extract_repo_info_from_url(repo_url)
            if not owner or not repo_name:
                return None
            
            content = self._get_file_content(owner, repo_name, file_path)
            if content:
                return {
                    'path': file_path,
                    'filename': file_path.split('/')[-1],
                    'content': content,
                    'size': len(content.encode('utf-8'))
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching file content for {file_path}: {e}")
            return None
    
    def _analyze_dependencies(self, package_files: Dict[str, str]) -> Tuple[List[str], str]:
        """Analyze dependencies and detect framework."""
        dependencies = []
        framework = "unknown"
        
        # Analyze requirements.txt
        if 'requirements.txt' in package_files:
            lines = package_files['requirements.txt'].split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before ==, >=, etc.)
                    pkg_name = re.split(r'[>=<!\s]', line)[0].strip()
                    if pkg_name:
                        dependencies.append(pkg_name)
        
        # Analyze pyproject.toml
        if 'pyproject.toml' in package_files:
            content = package_files['pyproject.toml']
            # Simple regex to extract dependencies
            deps_match = re.findall(r'"([^"]+)"', content)
            dependencies.extend(deps_match)
        
        # Analyze package.json
        if 'package.json' in package_files:
            try:
                pkg_json = json.loads(package_files['package.json'])
                deps = pkg_json.get('dependencies', {})
                dev_deps = pkg_json.get('devDependencies', {})
                dependencies.extend(list(deps.keys()))
                dependencies.extend(list(dev_deps.keys()))
            except json.JSONDecodeError:
                pass
        
        # Detect framework based on dependencies
        dep_str = ' '.join(dependencies).lower()
        if any(fw in dep_str for fw in ['torch', 'pytorch']):
            framework = "pytorch"
        elif any(fw in dep_str for fw in ['tensorflow', 'tf-']):
            framework = "tensorflow"
        elif 'gym' in dep_str or 'stable-baselines' in dep_str:
            framework = "reinforcement_learning"
        elif any(fw in dep_str for fw in ['jax', 'flax']):
            framework = "jax"
        elif any(fw in dep_str for fw in ['sklearn', 'scikit-learn']):
            framework = "scikit_learn"
        elif any(fw in dep_str for fw in ['fastapi', 'flask', 'django']):
            framework = "web_framework"
        
        return list(set(dependencies)), framework
    
    def generate_context_summary(self, repo_info: RepositoryInfo) -> str:
        """Generate a comprehensive context summary for AI."""
        if not repo_info:
            return ""
        
        context_parts = []
        
        # Basic info
        context_parts.append(f"Repository: {repo_info.owner}/{repo_info.name}")
        if repo_info.description:
            context_parts.append(f"Description: {repo_info.description}")
        
        # Framework and language
        context_parts.append(f"Primary Language: {repo_info.language}")
        context_parts.append(f"Framework: {repo_info.framework}")
        
        # Topics/tags
        if repo_info.topics:
            context_parts.append(f"Topics: {', '.join(repo_info.topics)}")
        
        # Key dependencies
        if repo_info.dependencies:
            key_deps = repo_info.dependencies[:10]  # Top 10 dependencies
            context_parts.append(f"Key Dependencies: {', '.join(key_deps)}")
        
        # README summary (first few lines)
        if repo_info.readme_content:
            readme_lines = repo_info.readme_content.split('\n')[:5]
            readme_summary = '\n'.join(line for line in readme_lines if line.strip())
            if readme_summary:
                context_parts.append(f"README Summary:\n{readme_summary}")
        
        # File structure overview
        if repo_info.file_structure:
            structure_summary = self._summarize_file_structure(repo_info.file_structure)
            context_parts.append(f"Project Structure: {structure_summary}")
        
        return '\n\n'.join(context_parts)
    
    def _summarize_file_structure(self, structure: Dict[str, Any], max_items: int = 8) -> str:
        """Summarize file structure for context."""
        items = []
        count = 0
        
        for name, info in structure.items():
            if count >= max_items:
                break
            
            if info['type'] == 'directory':
                items.append(f"{name}/ (directory)")
            else:
                items.append(name)
            count += 1
        
        if len(structure) > max_items:
            items.append(f"... and {len(structure) - max_items} more items")
        
        return ', '.join(items)

class RepoContextEnhancedImprover:
    """Enhanced code improver with repository context."""
    
    def __init__(self, github_context_provider: Optional[GitHubRepoContextProvider] = None):
        self.github_provider = github_context_provider or GitHubRepoContextProvider()
    
    def improve_code_with_repo_context(self, 
                                     original_code: str,
                                     filename: str,
                                     issues: List[Dict],
                                     repo_url: Optional[str] = None) -> Dict[str, Any]:
        """Improve code with repository context."""
        try:
            context_info = ""
            
            if repo_url:
                repo_info = self.github_provider.get_repository_info(repo_url)
                if repo_info:
                    context_info = self.github_provider.generate_context_summary(repo_info)
            
            # Build enhanced prompt with repository context
            enhanced_prompt = self._build_context_aware_prompt(
                original_code, filename, issues, context_info
            )
            
            return {
                "enhanced_prompt": enhanced_prompt,
                "repository_context": context_info,
                "context_available": bool(context_info)
            }
            
        except Exception as e:
            logger.error(f"Error improving code with repo context: {e}")
            return {
                "enhanced_prompt": "",
                "repository_context": "",
                "context_available": False,
                "error": str(e)
            }
    
    def _build_context_aware_prompt(self, 
                                  original_code: str, 
                                  filename: str,
                                  issues: List[Dict],
                                  context_info: str) -> str:
        """Build context-aware improvement prompt."""
        prompt_parts = []
        
        if context_info:
            prompt_parts.append("REPOSITORY CONTEXT:")
            prompt_parts.append(context_info)
            prompt_parts.append("\n" + "="*50 + "\n")
        
        prompt_parts.append("CODE IMPROVEMENT REQUEST:")
        prompt_parts.append(f"File: {filename}")
        prompt_parts.append(f"Issues detected: {len(issues)}")
        
        if issues:
            prompt_parts.append("\nISSUES TO FIX:")
            for i, issue in enumerate(issues[:5], 1):  # Top 5 issues
                issue_desc = issue.get('description', 'Unknown issue')
                prompt_parts.append(f"{i}. {issue_desc}")
        
        prompt_parts.append("\nORIGINAL CODE:")
        prompt_parts.append(original_code)
        
        prompt_parts.append("\nINSTRUCTIONS:")
        if context_info:
            prompt_parts.append("- Use the repository context to understand the project's architecture and patterns")
            prompt_parts.append("- Follow the project's coding style and conventions")
            prompt_parts.append("- Consider the framework and dependencies when making improvements")
        
        prompt_parts.append("- Fix all identified issues while preserving functionality")
        prompt_parts.append("- Return the complete improved code ready for replacement")
        prompt_parts.append("- Add comments explaining significant changes")
        
        return '\n'.join(prompt_parts)

def get_repo_context_provider(github_token: Optional[str] = None) -> GitHubRepoContextProvider:
    """Get or create GitHub repository context provider."""
    return GitHubRepoContextProvider(github_token)

def get_repo_context_improver(github_token: Optional[str] = None) -> RepoContextEnhancedImprover:
    """Get or create repository context enhanced improver."""
    provider = get_repo_context_provider(github_token)
    return RepoContextEnhancedImprover(provider)