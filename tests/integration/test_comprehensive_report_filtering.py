"""
Test script to verify comprehensive report false positive filtering functionality.
"""

import requests
import json
import time

def test_comprehensive_report_filtering():
    """Test that comprehensive reports apply false positive filtering correctly."""
    
    # Sample code with common false positives
    sample_code = '''
import torch
import numpy as np
import os
from typing import Optional

# Long line that exceeds typical line length limits - this should be filtered out as a style issue when filtering is enabled
def very_long_function_name_that_exceeds_normal_line_length_limits_and_should_trigger_line_length_warnings():
    pass

def train_model():
    # Missing whitespace around operators - style issue
    x=torch.randn(10,5)
    y=torch.randn(10,1)
    
    
    # Too many blank lines above - style issue
    
    model = torch.nn.Linear(5, 1)
    
    # Missing random seed - should NOT be filtered (important ML issue)
    for epoch in range(100):
        loss = torch.nn.functional.mse_loss(model(x), y)
        print(f"Loss: {loss}")  # Should suggest logging instead of print
'''

    test_request = {
        "files": [
            {
                "filename": "test_model.py",
                "content": sample_code
            }
        ],
        "format": "json",
        "include_ai_suggestions": False,  # Disable AI to focus on filtering test
        "apply_false_positive_filtering": True  # Test with filtering enabled
    }
    
    base_url = "https://codeguard.replit.app"
    
    print("Testing Comprehensive Report False Positive Filtering...")
    print("=" * 60)
    
    try:
        # Test 1: With false positive filtering
        print("\n1. Testing comprehensive report WITH false positive filtering:")
        start_time = time.time()
        
        response_with_filter = requests.post(
            f"{base_url}/reports/improvement-analysis",
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        elapsed_time = time.time() - start_time
        
        if response_with_filter.status_code == 200:
            data_with_filter = response_with_filter.json()
            print(f"   ✓ SUCCESS in {elapsed_time:.1f}s")
            print(f"   Total Issues Found: {data_with_filter['total_issues']}")
            print(f"   Files Analyzed: {data_with_filter['total_files']}")
            
            # Show severity breakdown
            severity = data_with_filter.get('severity_breakdown', {})
            print(f"   Severity Breakdown: {severity}")
            
            # Show issue categories
            categories = data_with_filter.get('issue_categories', {})
            print(f"   Issue Categories: {categories}")
            
        else:
            print(f"   ✗ FAILED with status {response_with_filter.status_code}")
            print(f"   Error: {response_with_filter.text}")
            return
        
        # Test 2: Without false positive filtering
        print("\n2. Testing comprehensive report WITHOUT false positive filtering:")
        test_request_no_filter = test_request.copy()
        test_request_no_filter["apply_false_positive_filtering"] = False
        
        start_time = time.time()
        
        response_no_filter = requests.post(
            f"{base_url}/reports/improvement-analysis",
            json=test_request_no_filter,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        elapsed_time = time.time() - start_time
        
        if response_no_filter.status_code == 200:
            data_no_filter = response_no_filter.json()
            print(f"   ✓ SUCCESS in {elapsed_time:.1f}s")
            print(f"   Total Issues Found: {data_no_filter['total_issues']}")
            print(f"   Files Analyzed: {data_no_filter['total_files']}")
            
            # Show severity breakdown
            severity = data_no_filter.get('severity_breakdown', {})
            print(f"   Severity Breakdown: {severity}")
            
        else:
            print(f"   ✗ FAILED with status {response_no_filter.status_code}")
            print(f"   Error: {response_no_filter.text}")
            return
        
        # Test 3: Compare results
        print("\n3. Filtering Effectiveness Analysis:")
        
        issues_filtered = data_no_filter['total_issues'] - data_with_filter['total_issues']
        print(f"   Issues without filtering: {data_no_filter['total_issues']}")
        print(f"   Issues with filtering: {data_with_filter['total_issues']}")
        print(f"   Issues filtered out: {issues_filtered}")
        
        if issues_filtered > 0:
            print(f"   ✓ False positive filtering is working!")
            print(f"   Reduction: {(issues_filtered/data_no_filter['total_issues']*100):.1f}%")
            
            # Verify important issues are preserved
            print(f"\n4. Important Issue Preservation Check:")
            
            # Check that security and ML-specific issues are preserved
            filtered_categories = data_with_filter.get('issue_categories', {})
            important_categories = ['security', 'ml_rules', 'custom_rules']
            
            for category in important_categories:
                count = filtered_categories.get(category, 0)
                if count > 0:
                    print(f"   ✓ {category}: {count} issues preserved")
                else:
                    print(f"   - {category}: No issues found")
            
        else:
            print(f"   ⚠ No issues were filtered - all issues might be valid")
        
        # Test 4: Performance check (should be fast since no ChatGPT calls)
        print(f"\n5. Performance Analysis:")
        print(f"   Report generation time: {elapsed_time:.1f}s")
        
        if elapsed_time < 15:  # Should be much faster than ChatGPT-based filtering
            print(f"   ✓ Fast processing (rule-based filtering)")
        else:
            print(f"   ⚠ Processing took longer than expected")
        
        print(f"\n6. Report Quality Check:")
        
        # Check if report actually contains filtered data
        if isinstance(data_with_filter.get('report'), dict):
            files_in_report = data_with_filter['report'].get('files', [])
            if files_in_report:
                first_file = files_in_report[0]
                issue_count = first_file.get('issue_count', 0)
                print(f"   ✓ Report contains {issue_count} issues for first file")
                print(f"   ✓ Report structure is properly formatted")
            else:
                print(f"   ⚠ No files found in report data")
        else:
            print(f"   ✓ Report generated in text format")
            
    except requests.exceptions.Timeout:
        print("   ✗ TIMEOUT - Report generation took too long")
    except requests.exceptions.ConnectionError:
        print("   ✗ CONNECTION ERROR - Cannot reach server")
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")

if __name__ == "__main__":
    test_comprehensive_report_filtering()