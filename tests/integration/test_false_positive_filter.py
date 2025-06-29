"""
Test script for ChatGPT false positive filtering functionality.
"""

import requests
import json

def test_false_positive_filtering():
    """Test the false positive filtering with sample code that might generate false positives."""
    
    # Sample code that might trigger false positives
    sample_code = '''
import torch
import numpy as np
import os
from typing import Optional

# This import is used dynamically but static analysis might not detect it
import pickle

class MLModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load model - pickle is used intentionally here for demonstration."""
        # This might be flagged as a security issue but could be valid in context
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)  # This should be flagged as security issue
        return model_data
    
    def train(self, data_loader):
        """Training loop with some patterns that might confuse static analysis."""
        # This variable might be flagged as unused but it's used in eval()
        learning_rate = 0.001
        
        for epoch in range(10):
            for batch in data_loader:
                # Dynamic evaluation - might be flagged but could be valid
                optimizer_config = f"torch.optim.Adam(params, lr={learning_rate})"
                optimizer = eval(optimizer_config)  # This should be flagged as security issue
                
                # Missing seed - should be caught
                torch.manual_seed(42)  # This is good practice
                
    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            # Hardcoded path - might be flagged as portability issue
            path = "/tmp/model.pth"  # This should be flagged
        
        torch.save(self.state_dict(), path)

# Missing function docstring - should be caught
def helper_function():
    pass

# Good practices that shouldn't be flagged
def proper_function() -> int:
    """Properly documented function with type hints."""
    torch.manual_seed(42)  # Good: sets seed
    return 42
'''

    # Test data
    test_request = {
        "files": [
            {
                "filename": "test_model.py",
                "content": sample_code
            }
        ],
        "options": {
            "level": "strict",
            "framework": "pytorch",
            "target": "gpu"
        }
    }
    
    base_url = "https://codeguard.replit.app"
    
    print("Testing CodeGuard False Positive Filtering...")
    print("=" * 50)
    
    try:
        # Test with false positive filtering (default)
        print("\n1. Testing WITH false positive filtering:")
        response_with_filter = requests.post(
            f"{base_url}/audit",
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=45
        )
        
        if response_with_filter.status_code == 200:
            data_with_filter = response_with_filter.json()
            print(f"   Status: SUCCESS")
            print(f"   Summary: {data_with_filter['summary']}")
            print(f"   Total Issues: {len(data_with_filter['issues'])}")
            
            # Show issues by category
            issue_types = {}
            for issue in data_with_filter['issues']:
                issue_type = issue.get('type', 'unknown')
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
            print(f"   Issue Breakdown: {issue_types}")
        else:
            print(f"   Status: ERROR {response_with_filter.status_code}")
            print(f"   Error: {response_with_filter.text}")
        
        # Test without false positive filtering
        print("\n2. Testing WITHOUT false positive filtering:")
        response_no_filter = requests.post(
            f"{base_url}/audit/no-filter",
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response_no_filter.status_code == 200:
            data_no_filter = response_no_filter.json()
            print(f"   Status: SUCCESS")
            print(f"   Summary: {data_no_filter['summary']}")
            print(f"   Total Issues: {len(data_no_filter['issues'])}")
            
            # Show issues by category
            issue_types = {}
            for issue in data_no_filter['issues']:
                issue_type = issue.get('type', 'unknown')
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
            print(f"   Issue Breakdown: {issue_types}")
        else:
            print(f"   Status: ERROR {response_no_filter.status_code}")
            print(f"   Error: {response_no_filter.text}")
        
        # Compare results
        if response_with_filter.status_code == 200 and response_no_filter.status_code == 200:
            filtered_count = len(data_no_filter['issues']) - len(data_with_filter['issues'])
            print(f"\n3. Comparison Results:")
            print(f"   Issues without filtering: {len(data_no_filter['issues'])}")
            print(f"   Issues with filtering: {len(data_with_filter['issues'])}")
            print(f"   Potential false positives filtered: {filtered_count}")
            
            if filtered_count > 0:
                print(f"   ✓ False positive filtering is working!")
                
                # Show what was filtered
                print(f"\n   Filtered issues might include:")
                filtered_issues = []
                for issue in data_no_filter['issues']:
                    found_in_filtered = False
                    for filtered_issue in data_with_filter['issues']:
                        if (issue['line'] == filtered_issue['line'] and 
                            issue['description'] == filtered_issue['description']):
                            found_in_filtered = True
                            break
                    if not found_in_filtered:
                        filtered_issues.append(issue)
                
                for issue in filtered_issues[:3]:  # Show first 3 filtered issues
                    print(f"   - Line {issue['line']}: {issue['description']}")
            else:
                print(f"   ⚠ No issues were filtered - this might indicate the filter isn't working or all issues are valid")
                
    except requests.exceptions.Timeout:
        print("   Status: TIMEOUT - Analysis took too long")
    except requests.exceptions.ConnectionError:
        print("   Status: CONNECTION ERROR - Cannot reach server")
    except Exception as e:
        print(f"   Status: ERROR - {str(e)}")

if __name__ == "__main__":
    test_false_positive_filtering()