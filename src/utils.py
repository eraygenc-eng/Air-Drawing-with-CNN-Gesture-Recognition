import cv2


def crop_hand(frame, hand_landmarks, padding=40, image_size=64):
    h, w, _ = frame.shape

    x_coords = []
    y_coords = []

    for landmark in hand_landmarks.landmark:
        x_coords.append(int(landmark.x * w))
        y_coords.append(int(landmark.y * h))

    x_min = max(min(x_coords) - padding, 0)
    y_min = max(min(y_coords) - padding, 0)
    x_max = min(max(x_coords) + padding, w)
    y_max = min(max(y_coords) + padding, h)

    hand_crop = frame[y_min:y_max, x_min:x_max]

    if hand_crop.size == 0:
        return None, None

    hand_crop = cv2.resize(hand_crop, (image_size, image_size))

    bbox = (x_min, y_min, x_max, y_max)

    return hand_crop, bbox