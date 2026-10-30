import cv2
import numpy as np
from PIL import Image, ImageDraw


def suggest_region_coordinates(
    regions: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    reference = min(regions, key=lambda region: region[0])
    region_width = min(width, reference[2] - reference[0])
    region_height = min(height, reference[3] - reference[1])
    x1 = max(0, reference[0] - region_width - 1)
    y1 = min(max(0, reference[1]), height - region_height)
    return x1, y1, x1 + region_width, y1 + region_height


def render_calibration_preview(
    frame: np.ndarray,
    regions: list[tuple[int, int, int, int] | None],
    max_size: tuple[int, int] = (560, 480),
) -> Image.Image:
    height, width = frame.shape[:2]
    scale = min(max_size[0] / width, max_size[1] / height)
    preview_size = (
        max(1, int(width * scale)),
        max(1, int(height * scale)),
    )
    resized = cv2.resize(frame, preview_size)
    image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    scale_x = preview_size[0] / width
    scale_y = preview_size[1] / height

    for index, region in enumerate(regions, start=1):
        if region is None:
            continue
        x1, y1, x2, y2 = region
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            continue

        left = round(x1 * scale_x)
        top = round(y1 * scale_y)
        right = round(x2 * scale_x) - 1
        bottom = round(y2 * scale_y) - 1
        draw.rectangle((left, top, right, bottom), outline="#39FF91", width=2)
        draw.text((left + 3, top + 3), str(index), fill="#39FF91")

    return image
