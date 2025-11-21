import requests
import sys

def test_batch(url, image_paths):
    print(f"Testing batch upload to {url} with {len(image_paths)} images")
    
    files = []
    for p in image_paths:
        files.append(('file', (p, open(p, 'rb'), 'image/jpeg')))
    
    try:
        # Sending multiple files with the same key 'file'
        res = requests.post(f"{url}/interrogate", files=files, data={"threshold": 0.35})
        print(f"Status: {res.status_code}")
        print(f"Response: {res.text[:500]}...") # Print first 500 chars
    except Exception as e:
        print(f"Error: {e}")
    finally:
        for _, f in files:
            f[1].close()

if __name__ == "__main__":
    test_batch("http://localhost:8000", ["test_image.jpg", "test_image_2.jpg"])
