import requests
import time
import io
from PIL import Image

# Configuration
BASE_URL = "http://localhost:8000"
IMAGE_FILENAME = r"tests/img.png"

def create_dummy_image():
    """Generates a temporary dummy image in memory to simulate an X-ray."""
    img = Image.new('RGB', (448, 448), color=(73, 109, 137))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_health():
    """Checks if the server is up and measures latency."""
    print(f"\n[TEST] Pinging Health Endpoint: {BASE_URL}/health")
    
    start_time = time.time()
    try:
        response = requests.get(f"{BASE_URL}/health")
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            print(f" SUCCESS: Server is healthy.")
            print(f"   Response: {response.json()}")
            print(f"   Latency:  {latency:.2f} ms")
            return True
        else:
            print(f" FAILED: Status Code {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f" FAILED: Could not connect to {BASE_URL}. Is the server running?")
        return False

def test_verification_workflow():
    """
    Uploads a dummy image and report to the real verification endpoint.
    Measures total end-to-end processing time.
    """
    endpoint = f"{BASE_URL}/api/verify-report"
    print(f"\n[TEST] Testing Verification Endpoint: {endpoint}")

    # Prepare Payload
    dummy_image = create_dummy_image()
    files = {
        'image': (IMAGE_FILENAME, dummy_image, 'image/png')
    }
    data = {
        'report_text': 'Normal chest X-ray. No acute findings. Heart size is normal.'
    }

    print("   Sending request (this may take time as models load/process)...")
    
    start_time = time.time()
    try:
        response = requests.post(endpoint, files=files, data=data)
        end_time = time.time()
        
        latency = (end_time - start_time)
        
        if response.status_code == 200:
            result = response.json()
            print(f" SUCCESS: Report Verified.")
            print(f"   Is Correct: {result.get('isCorrect')}")
            print(f"   Correction (if any): {result.get('correctReport')}")
            reference_cases = result.get('referenceCases')
            if reference_cases is None:
                print("   ERROR: Response missing 'referenceCases' field.")
            elif not isinstance(reference_cases, list):
                print("   ERROR: 'referenceCases' is not a list as expected.")
            else:
                print(f"   Reference Cases Returned: {len(reference_cases)}")
                if reference_cases:
                    sample_case = reference_cases[0]
                    findings = (sample_case.get('findings') or '')[:60]
                    impression = (sample_case.get('impression') or '')[:60]
                    print("   Sample Reference Case:")
                    print(f"     ID: {sample_case.get('id')}")
                    print(f"     Findings: {findings}...")
                    print(f"     Impression: {impression}...")
                else:
                    print("   WARNING: referenceCases array is empty.")
            print(f"   Total Time: {latency:.2f} seconds")
        else:
            print(f" FAILED: Status Code {response.status_code}")
            print(f"   Detail: {response.text}")

    except Exception as e:
        print(f" ERROR: {str(e)}")

if __name__ == "__main__":
    server_up = test_health()
    if server_up:
        for i in range(3):
            test_verification_workflow()