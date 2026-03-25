import base64

import cv2

MAX_IMAGE_WIDTH = 480
JPEG_QUALITY = 82


def resize_frame(frame, max_width=MAX_IMAGE_WIDTH):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame

    scale = max_width / float(w)
    new_size = (max_width, max(1, int(h * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def encode_frame(frame, max_width=MAX_IMAGE_WIDTH, jpeg_quality=JPEG_QUALITY):
    resized = resize_frame(frame, max_width=max_width)

    # Convert BGR to RGB before JPEG encoding.
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        return None

    return buf


def frame_to_base64(frame, max_width=MAX_IMAGE_WIDTH, jpeg_quality=JPEG_QUALITY):
    buf = encode_frame(frame, max_width=max_width, jpeg_quality=jpeg_quality)
    if buf is None:
        return None

    return base64.b64encode(buf).decode()


def frame_to_dataurl(frame, max_width=MAX_IMAGE_WIDTH, jpeg_quality=JPEG_QUALITY):
    encoded = frame_to_base64(frame, max_width=max_width, jpeg_quality=jpeg_quality)
    if encoded is None:
        return None

    return "data:image/jpeg;base64," + encoded
