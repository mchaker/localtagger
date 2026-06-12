import argparse

import requests


def test_batch(url, image_paths, model):
    print(f"Testing batch upload to {url} with {len(image_paths)} images (model={model or 'default'})")

    files = [("file", (p, open(p, "rb"), "image/jpeg")) for p in image_paths]
    data = {"threshold": 0.35}
    if model:
        data["model"] = model

    try:
        # Sending multiple files with the same key 'file'
        res = requests.post(f"{url}/interrogate", files=files, data=data)
        print(f"Status: {res.status_code}")
        print(f"Response: {res.text[:500]}...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        for _, f in files:
            f[1].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default=None, help="Model id from /models (optional)")
    parser.add_argument(
        "images", nargs="*", default=["test_image.jpg", "test_image_2.jpg"]
    )
    args = parser.parse_args()

    test_batch(args.url, args.images, args.model)
