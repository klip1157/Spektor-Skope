#!/usr/bin/env python3

import requests
import sys
import json
from datetime import datetime
from typing import Dict, Any

class GhostCameraAPITester:
    def __init__(self, base_url="https://ghost-cam.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.session_id = None
        self.detection_id = None
        self.screenshot_id = None

    def run_test(self, name: str, method: str, endpoint: str, expected_status: int, data: Dict[Any, Any] = None) -> tuple:
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=10)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=10)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=10)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed - Network Error: {str(e)}")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test GET /api/ endpoint"""
        success, response = self.run_test(
            "Root API Endpoint",
            "GET",
            "",
            200
        )
        return success and "message" in response

    def test_create_session(self):
        """Test POST /api/sessions endpoint"""
        success, response = self.run_test(
            "Create Session",
            "POST",
            "sessions",
            200
        )
        if success and "id" in response:
            self.session_id = response["id"]
            print(f"   Created session ID: {self.session_id}")
            return True
        return False

    def test_get_sessions(self):
        """Test GET /api/sessions endpoint"""
        success, response = self.run_test(
            "Get Sessions",
            "GET",
            "sessions",
            200
        )
        return success and isinstance(response, list)

    def test_update_session(self):
        """Test PUT /api/sessions/{id} endpoint"""
        if not self.session_id:
            print("❌ Skipping session update - no session ID available")
            return False
            
        update_data = {
            "session_end": datetime.now().isoformat(),
            "total_detections": 5,
            "max_emf_level": 3.2
        }
        
        success, response = self.run_test(
            "Update Session",
            "PUT",
            f"sessions/{self.session_id}",
            200,
            update_data
        )
        return success and "id" in response

    def test_create_detection(self):
        """Test POST /api/detections endpoint"""
        detection_data = {
            "detection_type": "pose",
            "confidence": 0.85,
            "keypoints_count": 17,
            "emf_level": 2.3,
            "spirit_box_frequency": 95.5,
            "notes": "Test detection event"
        }
        
        success, response = self.run_test(
            "Create Detection Event",
            "POST",
            "detections",
            200,
            detection_data
        )
        if success and "id" in response:
            self.detection_id = response["id"]
            print(f"   Created detection ID: {self.detection_id}")
            return True
        return False

    def test_get_detections(self):
        """Test GET /api/detections endpoint"""
        success, response = self.run_test(
            "Get Detection Events",
            "GET",
            "detections",
            200
        )
        return success and isinstance(response, list)

    def test_get_detection_by_id(self):
        """Test GET /api/detections/{id} endpoint"""
        if not self.detection_id:
            print("❌ Skipping detection retrieval - no detection ID available")
            return False
            
        success, response = self.run_test(
            "Get Detection by ID",
            "GET",
            f"detections/{self.detection_id}",
            200
        )
        return success and "id" in response

    def test_create_screenshot(self):
        """Test POST /api/screenshots endpoint"""
        # Create a small base64 test image data
        test_image_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        screenshot_data = {
            "image_data": test_image_data,
            "detection_count": 3,
            "emf_level": 2.1,
            "notes": "Test screenshot"
        }
        
        success, response = self.run_test(
            "Save Screenshot",
            "POST",
            "screenshots",
            200,
            screenshot_data
        )
        if success and "id" in response:
            self.screenshot_id = response["id"]
            print(f"   Created screenshot ID: {self.screenshot_id}")
            return True
        return False

    def test_get_screenshots(self):
        """Test GET /api/screenshots endpoint"""
        success, response = self.run_test(
            "Get Screenshots",
            "GET",
            "screenshots",
            200
        )
        return success and isinstance(response, list)

    def test_get_screenshot_by_id(self):
        """Test GET /api/screenshots/{id} endpoint"""
        if not self.screenshot_id:
            print("❌ Skipping screenshot retrieval - no screenshot ID available")
            return False
            
        success, response = self.run_test(
            "Get Screenshot by ID",
            "GET",
            f"screenshots/{self.screenshot_id}",
            200
        )
        return success and "id" in response

    def test_status_endpoints(self):
        """Test status check endpoints"""
        # Test create status check
        status_data = {
            "client_name": "test_client"
        }
        
        success1, response1 = self.run_test(
            "Create Status Check",
            "POST",
            "status",
            200,
            status_data
        )
        
        # Test get status checks
        success2, response2 = self.run_test(
            "Get Status Checks",
            "GET",
            "status",
            200
        )
        
        return success1 and success2 and isinstance(response2, list)

def main():
    print("🚀 Starting GhostTube SLS Camera API Tests")
    print("=" * 50)
    
    tester = GhostCameraAPITester()
    
    # Run all tests
    test_results = []
    
    # Basic API tests
    test_results.append(("Root Endpoint", tester.test_root_endpoint()))
    
    # Session management tests
    test_results.append(("Create Session", tester.test_create_session()))
    test_results.append(("Get Sessions", tester.test_get_sessions()))
    test_results.append(("Update Session", tester.test_update_session()))
    
    # Detection event tests
    test_results.append(("Create Detection", tester.test_create_detection()))
    test_results.append(("Get Detections", tester.test_get_detections()))
    test_results.append(("Get Detection by ID", tester.test_get_detection_by_id()))
    
    # Screenshot tests
    test_results.append(("Create Screenshot", tester.test_create_screenshot()))
    test_results.append(("Get Screenshots", tester.test_get_screenshots()))
    test_results.append(("Get Screenshot by ID", tester.test_get_screenshot_by_id()))
    
    # Status check tests
    test_results.append(("Status Endpoints", tester.test_status_endpoints()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed_tests = []
    failed_tests = []
    
    for test_name, result in test_results:
        if result:
            print(f"✅ {test_name}")
            passed_tests.append(test_name)
        else:
            print(f"❌ {test_name}")
            failed_tests.append(test_name)
    
    print(f"\n📈 Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    print(f"🎯 Success Rate: {(tester.tests_passed/tester.tests_run)*100:.1f}%")
    
    if failed_tests:
        print(f"\n❌ Failed Tests: {', '.join(failed_tests)}")
    
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())