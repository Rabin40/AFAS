import requests

SERVER = "http://127.0.0.1:5001"
SESSION = requests.Session()


def _json_or_raise(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        raise RuntimeError(
            f"Server returned non-JSON: {resp.status_code} {resp.text[:200]}"
        )


def _post_json(path: str, payload: dict, timeout: int):
    resp = SESSION.post(f"{SERVER}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return _json_or_raise(resp)


def create_user(payload: dict):
    return _post_json("/api/users/create", payload, timeout=30)


def upload_sample(user_id: int, pose: str, image_dataurl: str):
    return _post_json(
        "/api/enroll/add_sample",
        {"user_id": user_id, "pose": pose, "image": image_dataurl},
        timeout=60
    )


def find_user_by_face(image_dataurl: str):
    return _post_json(
        "/api/enroll/find_user",
        {"image": image_dataurl},
        timeout=60
    )


def scan_attendance(image_dataurl: str):
    return _post_json(
        "/api/attendance/scan_and_mark",
        {"image": image_dataurl},
        timeout=60
    )
