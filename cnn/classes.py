CLASS_NAMES = [
    "idle",
    "index_finger",
    "open_palm",
    "thumb_only"
]

CLASS_TO_IDX = {
    class_name: idx for idx, class_name in enumerate(CLASS_NAMES)
}