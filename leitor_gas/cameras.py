import cv2


def find_cameras(max_devices: int = 8, capture_factory=None) -> list[int]:
    if capture_factory is None:
        capture_factory = cv2.VideoCapture

    available = []
    for index in range(max_devices):
        capture = None
        try:
            capture = capture_factory(index)
            if capture.isOpened():
                available.append(index)
        except cv2.error:
            continue
        finally:
            if capture is not None:
                capture.release()
    return available
