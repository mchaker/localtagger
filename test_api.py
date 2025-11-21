import requests
import argparse
import sys

def test_endpoints(base_url, image_path, image_url):
    print(f"Testing API at {base_url}")
    
    # Test Data
    files = {'file': open(image_path, 'rb')}
    
    # 5. Test Main POST
    print("\n--- Testing Main POST ---")
    try:
        files['file'].seek(0)
        res = requests.post(f"{base_url}/interrogate", files=files, data={"threshold": 0.35, "use_escape": False})
        print(f"Status: {res.status_code}")
        # Truncate tags for display
        json_res = res.json()
        if isinstance(json_res, list):
            json_res = json_res[0]
        if "tag_string" in json_res:
            json_res["tag_string"] = json_res["tag_string"][:100] + "..."
        print(f"Response: {json_res}")
    except Exception as e:
        print(f"Failed: {e}")

    # 6. Test Main GET
    print("\n--- Testing Main GET ---")
    try:
        res = requests.get(f"{base_url}/interrogate", params={"url": image_url, "threshold": 0.35, "use_escape": False})
        print(f"Status: {res.status_code}")
        json_res = res.json()
        if isinstance(json_res, list):
            json_res = json_res[0]
        if "tag_string" in json_res:
            json_res["tag_string"] = json_res["tag_string"][:100] + "..."
        print(f"Response: {json_res}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--image", required=True, help="Path to local image for POST tests")
    parser.add_argument("--image-url", default="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg", help="URL of image for GET tests")
    args = parser.parse_args()
    
    test_endpoints(args.url, args.image, args.image_url)
