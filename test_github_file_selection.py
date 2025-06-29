"""
Test script for GitHub file selection feature.
Validates that repository files can be fetched and loaded correctly.
"""

import requests
import json

def test_github_file_selection():
    """Test the new GitHub file selection endpoints."""
    
    base_url = "http://localhost:5000"
    
    # Test with a known public repository
    test_repo = "https://github.com/pytorch/examples"
    
    print("Testing GitHub File Selection Feature")
    print("=" * 40)
    
    # Step 1: Test repository file listing
    print("\n1. Testing repository file listing...")
    
    files_payload = {
        "repo_url": test_repo,
        "max_files": 10
    }
    
    try:
        response = requests.post(
            f"{base_url}/repo/files",
            json=files_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Repository files fetched successfully")
            print(f"  Total files found: {result['total_files']}")
            
            if result['files']:
                print("  Sample files:")
                for file in result['files'][:5]:
                    path_display = f"{file['directory']}/{file['filename']}" if file['directory'] else file['filename']
                    print(f"    - {path_display} ({file['size']} bytes)")
                
                # Step 2: Test loading a specific file
                print("\n2. Testing file content loading...")
                
                # Try to load the first Python file
                test_file = result['files'][0]
                
                content_payload = {
                    "repo_url": test_repo,
                    "file_path": test_file['path']
                }
                
                content_response = requests.post(
                    f"{base_url}/repo/file-content",
                    json=content_payload,
                    timeout=30
                )
                
                if content_response.status_code == 200:
                    content_result = content_response.json()
                    print(f"✓ File content loaded successfully")
                    print(f"  File: {content_result['file']['filename']}")
                    print(f"  Size: {content_result['file']['size']} bytes")
                    print(f"  Content preview: {content_result['file']['content'][:200]}...")
                    
                    return True
                else:
                    print(f"✗ Failed to load file content: {content_response.status_code}")
                    return False
            else:
                print("✗ No Python files found in repository")
                return False
        else:
            print(f"✗ Failed to fetch repository files: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing file selection: {e}")
        return False

def test_playground_integration():
    """Test that playground can access the file selection endpoints."""
    
    print("\n3. Testing playground integration...")
    
    # Test that the playground can reach the main page
    try:
        response = requests.get("http://localhost:5000/playground", timeout=10)
        if response.status_code == 200:
            print("✓ Playground accessible")
            
            # Check if the JavaScript file loads
            js_response = requests.get("http://localhost:5000/static/playground.js", timeout=10)
            if js_response.status_code == 200:
                print("✓ JavaScript file loads correctly")
                
                # Check if our new methods are present
                js_content = js_response.text
                if "fetchRepositoryFiles" in js_content and "loadSelectedRepoFile" in js_content:
                    print("✓ File selection methods present in JavaScript")
                    return True
                else:
                    print("✗ File selection methods missing from JavaScript")
                    return False
            else:
                print("✗ JavaScript file not accessible")
                return False
        else:
            print("✗ Playground not accessible")
            return False
            
    except Exception as e:
        print(f"✗ Error testing playground integration: {e}")
        return False

def main():
    """Run all tests for GitHub file selection feature."""
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: GitHub file selection functionality
    if test_github_file_selection():
        tests_passed += 1
    
    # Test 2: Playground integration
    if test_playground_integration():
        tests_passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All GitHub file selection features working correctly")
        print("✓ Users can now select individual files from GitHub repositories")
        print("✓ File content loads directly into the playground editor")
    else:
        print("⚠ Some tests failed - check API endpoints and connectivity")

if __name__ == "__main__":
    main()