import argparse

import requests


def test_endpoints(base_url, image_path, image_url, model):
    print(f"Testing API at {base_url} (model={model or 'default'})")

    # 0. Model catalog
    print("\n--- Testing GET /models ---")
    try:
        res = requests.get(f"{base_url}/models")
        print(f"Status: {res.status_code}")
        for m in res.json().get("models", []):
            print(f"  - {m['id']}: {m['label']} (family={m['family']}, gated={m['gated']})")
    except Exception as e:
        print(f"Failed: {e}")

    params = {"threshold": 0.35, "use_escape": False}
    if model:
        params["model"] = model

    # 1. Main POST (file upload)
    print("\n--- Testing Main POST ---")
    try:
        with open(image_path, "rb") as fh:
            res = requests.post(
                f"{base_url}/interrogate", files={"file": fh}, data=params
            )
        print(f"Status: {res.status_code}")
        json_res = res.json()
        if isinstance(json_res, list):
            json_res = json_res[0]
        if "tag_string" in json_res:
            json_res["tag_string"] = json_res["tag_string"][:100] + "..."
        print(f"Response: {json_res}")
    except Exception as e:
        print(f"Failed: {e}")

    # 2. Main GET (image URL)
    print("\n--- Testing Main GET ---")
    try:
        get_params = {"url": image_url, "threshold": 0.35, "use_escape": False}
        if model:
            get_params["model"] = model
        res = requests.get(f"{base_url}/interrogate", params=get_params)
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
    parser.add_argument("--model", default=None, help="Model id from /models (optional)")
    parser.add_argument(
        "--image-url",
        default="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg",
        help="URL of image for GET tests",
    )
    args = parser.parse_args()

    test_endpoints(args.url, args.image, args.image_url, args.model)
