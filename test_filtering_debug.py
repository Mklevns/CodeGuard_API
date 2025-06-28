"""
Simple test to debug false positive filtering in comprehensive reports.
"""

import requests
import json

def test_filtering_simple():
    """Test with simple code that should definitely trigger filtering."""
    
    # Code with obvious style issues that should be filtered
    simple_code = '''import torch
def test():
    x=1+2  # E225: missing whitespace around operator
    return x'''

    # Test with filtering
    response = requests.post(
        "https://codeguard.replit.app/reports/improvement-analysis",
        json={
            "files": [{"filename": "simple.py", "content": simple_code}],
            "format": "json",
            "include_ai_suggestions": False,
            "apply_false_positive_filtering": True
        },
        timeout=15
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"With filtering: {data['total_issues']} issues")
        
        # Test without filtering
        response2 = requests.post(
            "https://codeguard.replit.app/reports/improvement-analysis",
            json={
                "files": [{"filename": "simple.py", "content": simple_code}],
                "format": "json", 
                "include_ai_suggestions": False,
                "apply_false_positive_filtering": False
            },
            timeout=15
        )
        
        if response2.status_code == 200:
            data2 = response2.json()
            print(f"Without filtering: {data2['total_issues']} issues")
            print(f"Difference: {data2['total_issues'] - data['total_issues']} issues filtered")
            
            # Show actual issues for debugging
            if isinstance(data.get('report'), dict) and 'files' in data['report']:
                issues = data['report']['files'][0].get('issues', [])[:5]  # First 5 issues
                print("\nSample issues (with filtering):")
                for issue in issues:
                    print(f"  - {issue.get('description', 'No description')}")
        else:
            print(f"Error without filtering: {response2.status_code}")
    else:
        print(f"Error with filtering: {response.status_code}")

if __name__ == "__main__":
    test_filtering_simple()