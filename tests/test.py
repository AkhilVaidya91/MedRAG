"""
Example client script for the Radiology Report Verification API

This demonstrates how to call the API from a Python client application.
"""

import requests
import sys
from pathlib import Path


def verify_report(image_path: str, report_text: str, api_url: str = "http://localhost:8000") -> dict:
    """
    Send an X-ray image and report text to the verification API
    
    Args:
        image_path: Path to the X-ray image file
        report_text: Text of the radiology report
        api_url: Base URL of the API (default: http://localhost:8000)
    
    Returns:
        dict: Response containing relevant_report_ids
    """
    # Check if image file exists
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Prepare the request
    endpoint = f"{api_url}/api/verify-report"
    
    # Open and send the image file
    with open(image_path, 'rb') as image_file:
        files = {
            'image': image_file
        }
        data = {
            'report_text': report_text
        }
        
        print(f"Sending request to {endpoint}...")
        print(f"Image: {image_path}")
        print(f"Report: {report_text[:100]}..." if len(report_text) > 100 else f"Report: {report_text}")
        
        # Make the request
        response = requests.post(endpoint, files=files, data=data)
    
    # Check response
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status {response.status_code}")


def main():
    """Main function with example usage"""
    # Example usage
    if len(sys.argv) < 3:
        print("Usage: python example_client.py <image_path> <report_text>")
        print("\nExample:")
        print('  python example_client.py /path/to/xray.jpg "The lungs are clear of focal consolidation. No acute cardiopulmonary process."')
        print('\nNote: Provide the full or relative path to your X-ray image file')
        sys.exit(1)
    
    image_path = sys.argv[1]
    report_text = sys.argv[2]
    
    try:
        # Call the API
        result = verify_report(image_path, report_text)
        print(result)
        # Print results
        print("\n" + "=" * 60)
        print("Verification Results")
        print("=" * 60)
        print(f"\nTop 10 Most Relevant Report IDs:")
        for i, report_id in enumerate(result['relevant_report_ids'], 1):
            print(f"  {i}. Report ID: {report_id}")
        
        print("\n✓ Verification completed successfully")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()