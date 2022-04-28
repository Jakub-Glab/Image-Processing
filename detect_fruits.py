import json
from pathlib import Path
from typing import Dict
import numpy as np

import click
import cv2
from tqdm import tqdm


def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """

    # TODO: Implement detection method.
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")

# APPLE:
    red_lower = np.array([11, 104, 255])
    red_upper = np.array([3, 255, 255])

    red_lower2 = np.array([0, 42, 0])
    red_upper2 = np.array([7, 255, 255])

    mask_red1 = cv2.inRange(hsv, red_lower, red_upper)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = mask_red1 + mask_red2
    kernel = np.ones((6, 6), np.uint8)
    kernel2 = np.ones((6, 6), np.uint8)
    kernel3 = np.ones((20, 20), np.uint8)
    opening = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
    dilation = cv2.dilate(closing, kernel3)
    contours, hierarchy = cv2.findContours(
        dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    final_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 55000:
            final_contours.append(contour)
    for i in range(len(final_contours)):
        count = cv2.drawContours(dilation, final_contours, i, (50, 250, 50), 7)

    apple = len(final_contours)
# Banana

    yellow_lower = np.array([34, 85, 255])
    yellow_upper = np.array([39, 255, 255])

    yellow_lower2 = np.array([21, 83, 139])
    yellow_upper2 = np.array([37, 255, 255])

    mask_yellow1 = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_yellow2 = cv2.inRange(hsv, yellow_lower2, yellow_upper2)
    mask_yellow = mask_yellow1 + mask_yellow2
    kernel = np.ones((2, 2), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    opening2 = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    closing2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE, kernel2)
    contours, hierarchy = cv2.findContours(
        closing2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    final_contours2 = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 30000:
            final_contours2.append(contour)
    for j in range(len(final_contours2)):
        count3 = cv2.drawContours(closing2, final_contours2, j, (50, 250, 50), 7)

    banana = len(final_contours2)

# Orange:
    orange_lower = np.array([9, 198, 153])
    orange_upper = np.array([17, 255, 255])

    orange_lower2 = np.array([23, 255, 255])
    orange_upper2 = np.array([28, 255, 255])

    mask_orange1 = cv2.inRange(hsv, orange_lower, orange_upper)
    mask_orange2 = cv2.inRange(hsv, orange_lower2, orange_upper2)
    mask_orange = mask_orange1 + mask_orange2
    kernel = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((4, 4), np.uint8)
    opening3 = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
    closing3 = cv2.morphologyEx(opening3, cv2.MORPH_CLOSE, kernel2)
    contours, hierarchy = cv2.findContours(
        closing3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    final_contours3 = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 30000:
            final_contours3.append(contour)
    for k in range(len(final_contours3)):
        count3 = cv2.drawContours(closing3, final_contours3, k, (252, 3, 115), 7)

    orange = len(final_contours3)

    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False,
                                                                                  path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
