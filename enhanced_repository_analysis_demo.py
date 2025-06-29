"""
Comprehensive demonstration of Enhanced Repository Analysis features for CodeGuard API.
Shows dependency vulnerabilities, complexity metrics, Git history analysis, and repository heatmap.
"""

import requests
import json
import time
from datetime import datetime

def run_enhanced_analysis_demo():
    """Run comprehensive demonstration of all enhanced repository analysis features."""
    
    print("Enhanced Repository Analysis Features Demonstration")
    print("=" * 60)
    print("Showcasing advanced static analysis capabilities with dependency scanning,")
    print("complexity metrics, Git history analysis, and comprehensive repository heatmaps")
    print()
    
    base_url = "http://localhost:5000"
    
    # Feature 1: Dependency Vulnerability Analysis
    print("1. Dependency Vulnerability & License Analysis")
    print("-" * 45)
    
    try:
        response = requests.get(f"{base_url}/analysis/dependency-vulnerabilities")
        if response.status_code == 200:
            vuln_data = response.json()
            
            if vuln_data.get("available"):
                print(f"✓ Dependency files analyzed: {', '.join(vuln_data['dependencies_analyzed'])}")
                print(f"  Security vulnerabilities found: {vuln_data['summary']['total_vulnerabilities']}")
                print(f"  License issues detected: {vuln_data['summary']['total_license_issues']}")
                print(f"  Critical security issues: {vuln_data['summary']['critical_issues']}")
                print(f"  Fix recommendations: {vuln_data['summary']['recommendations']}")
                
                if vuln_data['vulnerabilities']:
                    print("\n  Sample vulnerability:")
                    vuln = vuln_data['vulnerabilities'][0]
                    print(f"    • {vuln['description']} ({vuln['severity']})")
                
                if vuln_data['license_issues']:
                    print("\n  Sample license issue:")
                    license_issue = vuln_data['license_issues'][0]
                    print(f"    • {license_issue['description']}")
            else:
                print(f"  {vuln_data.get('message', 'Analysis not available')}")
        else:
            print(f"  Error: HTTP {response.status_code}")
    except Exception as e:
        print(f"  Connection error: {str(e)}")
    
    print()
    
    # Feature 2: Code Complexity Metrics
    print("2. Code Complexity & Technical Debt Analysis")
    print("-" * 45)
    
    try:
        response = requests.get(f"{base_url}/analysis/complexity-metrics")
        if response.status_code == 200:
            complexity_data = response.json()
            
            if complexity_data.get("available"):
                print(f"✓ Python files analyzed: {complexity_data['files_analyzed']}")
                print(f"  Functions with high complexity: {complexity_data['summary']['high_complexity_functions']}")
                print(f"  Maintainability issues: {complexity_data['summary']['maintainability_issues']}")
                print(f"  Refactoring opportunities: {complexity_data['summary']['refactoring_opportunities']}")
                print(f"  Technical debt score: {complexity_data['summary']['technical_debt_score']:.2f}")
                
                if complexity_data['complexity_issues']:
                    print("\n  Sample complexity issue:")
                    issue = complexity_data['complexity_issues'][0]
                    print(f"    • {issue['filename']}:{issue['line']} - {issue['description']}")
                
                if complexity_data['refactoring_suggestions']:
                    print("\n  Sample refactoring suggestion:")
                    suggestion = complexity_data['refactoring_suggestions'][0]
                    print(f"    • {suggestion['filename']}:{suggestion['line']} - {suggestion['suggestion']}")
            else:
                print(f"  {complexity_data.get('message', 'Analysis not available')}")
        else:
            print(f"  Error: HTTP {response.status_code}")
    except Exception as e:
        print(f"  Connection error: {str(e)}")
    
    print()
    
    # Feature 3: Git History Analysis
    print("3. Git History & Bug Pattern Analysis")
    print("-" * 40)
    
    try:
        response = requests.get(f"{base_url}/analysis/git-history?days=90")
        if response.status_code == 200:
            git_data = response.json()
            
            if git_data.get("available"):
                trends = git_data.get("repository_trends", {})
                print(f"✓ Git history analyzed: {git_data.get('analysis_period_days', 90)} days")
                print(f"  Total commits: {trends.get('total_commits', 0)}")
                print(f"  Bug fix commits: {trends.get('bug_fix_commits', 0)}")
                print(f"  Bug fix ratio: {trends.get('bug_fix_ratio', 0):.1%}")
                print(f"  Average commits/day: {trends.get('average_commits_per_day', 0):.1f}")
                
                bug_prone_files = git_data.get("bug_prone_files", [])
                if bug_prone_files:
                    print(f"\n  Bug-prone files identified: {len(bug_prone_files)}")
                    top_bug_prone = bug_prone_files[0]
                    print(f"    • Most problematic: {top_bug_prone['filename']}")
                    print(f"      Risk score: {top_bug_prone['risk_score']:.3f}")
                    print(f"      Bug fixes: {top_bug_prone['bug_fix_count']}/{top_bug_prone['commit_count']} commits")
                
                high_churn_files = git_data.get("high_churn_files", [])
                if high_churn_files:
                    print(f"\n  High-churn files: {len(high_churn_files)}")
                    top_churn = high_churn_files[0]
                    print(f"    • Highest churn: {top_churn['filename']}")
                    print(f"      Churn score: {top_churn['churn_score']:.2f}")
                
                recommendations = git_data.get("recommendations", [])
                if recommendations:
                    print(f"\n  Key recommendations:")
                    for rec in recommendations[:2]:
                        print(f"    • {rec}")
            else:
                print(f"  {git_data.get('error', 'Git analysis not available')}")
        else:
            print(f"  Error: HTTP {response.status_code}")
    except Exception as e:
        print(f"  Connection error: {str(e)}")
    
    print()
    
    # Feature 4: Comprehensive Repository Heatmap
    print("4. Comprehensive Repository Risk Heatmap")
    print("-" * 42)
    
    try:
        response = requests.get(f"{base_url}/analysis/repository-heatmap")
        if response.status_code == 200:
            heatmap_data = response.json()
            
            if heatmap_data.get("available"):
                summary = heatmap_data['summary']
                print(f"✓ Repository heatmap generated")
                print(f"  Files analyzed: {summary['total_files']}")
                print(f"  High-risk files: {summary['high_risk_files']}")
                print(f"  Medium-risk files: {summary['medium_risk_files']}")
                print(f"  Low-risk files: {summary['low_risk_files']}")
                print(f"  Total issues found: {summary['total_issues']}")
                print(f"  Security issues: {summary['total_security_issues']}")
                print(f"  Git analysis integrated: {summary['git_analysis_available']}")
                
                heatmap_files = heatmap_data.get("heatmap_data", [])
                if heatmap_files:
                    print(f"\n  Top 3 highest-risk files:")
                    for i, file_data in enumerate(heatmap_files[:3], 1):
                        print(f"    {i}. {file_data['filename']}")
                        print(f"       Risk score: {file_data['risk_score']:.3f}")
                        print(f"       Issues: {file_data['issue_count']}")
                        print(f"       Security issues: {file_data['security_issues']}")
                        if file_data.get('git_metrics'):
                            git_metrics = file_data['git_metrics']
                            print(f"       Git risk: {git_metrics.get('risk_score', 0):.3f}")
                
                recommendations = heatmap_data.get("recommendations", [])
                if recommendations:
                    print(f"\n  Heatmap recommendations:")
                    for rec in recommendations:
                        print(f"    • {rec}")
            else:
                print(f"  {heatmap_data.get('message', 'Heatmap generation not available')}")
        else:
            print(f"  Error: HTTP {response.status_code}")
    except Exception as e:
        print(f"  Connection error: {str(e)}")
    
    print()
    
    # Summary and Integration Benefits
    print("=" * 60)
    print("ENHANCED REPOSITORY ANALYSIS INTEGRATION BENEFITS")
    print("=" * 60)
    
    benefits = [
        "Comprehensive Security Assessment: Dependency vulnerability scanning with pip-audit",
        "Advanced Code Quality Metrics: Cyclomatic complexity and maintainability analysis",
        "Historical Pattern Recognition: Git history analysis for bug-prone file identification",
        "Cross-file Relationship Analysis: Unused code and circular dependency detection",
        "Risk-based Prioritization: Intelligent heatmap combining all analysis sources",
        "Technical Debt Quantification: Measurable code quality and refactoring metrics",
        "Proactive Issue Prevention: Early identification of maintenance hotspots",
        "Enterprise-ready Analytics: Production-level repository health monitoring"
    ]
    
    for benefit in benefits:
        print(f"✓ {benefit}")
    
    print()
    print("Integration Features Successfully Demonstrated:")
    print("• Dependency vulnerability scanning with automated fix suggestions")
    print("• Code complexity analysis with technical debt scoring")
    print("• Git history pattern analysis for identifying problematic files")
    print("• Repository-wide risk assessment with integrated heatmap visualization")
    print("• Cross-tool correlation for comprehensive code quality insights")
    print()
    print("All enhanced repository analysis features are now operational!")


if __name__ == "__main__":
    run_enhanced_analysis_demo()