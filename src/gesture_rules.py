import math


def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def is_finger_up(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y


def is_thumb_open(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_mcp = landmarks[5]

    tip_distance = distance(thumb_tip, index_mcp)
    ip_distance = distance(thumb_ip, index_mcp)

    return tip_distance > ip_distance * 1.35


def detect_gesture(landmarks):
    thumb_open = is_thumb_open(landmarks)
    index_up = is_finger_up(landmarks, 8, 6)
    middle_up = is_finger_up(landmarks, 12, 10)
    ring_up = is_finger_up(landmarks, 16, 14)
    pinky_up = is_finger_up(landmarks, 20, 18)

    only_index_up = (
        index_up
        and not middle_up
        and not ring_up
        and not pinky_up
        and not thumb_open
    )

    only_thumb_up = (
        thumb_open
        and not index_up
        and not middle_up
        and not ring_up
        and not pinky_up
    )

    open_palm = (
        index_up
        and middle_up
        and ring_up
        and pinky_up
    )

    if open_palm:
        return "open_palm"

    if only_thumb_up:
        return "eraser"

    if only_index_up:
        return "drawing"

    return "idle"