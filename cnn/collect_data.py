import cv2
import os
import time
import mediapipe as mp

from cnn.classes import CLASS_NAMES


SAVE_DIR = "data/raw"
IMAGE_SIZE = 64

KEY_TO_CLASS = {
    ord("0"): "idle",
    ord("1"): "index_finger",
    ord("2"): "thumb_only",
    ord("3"): "open_palm",
}


def create_class_folders():
    for class_name in CLASS_NAMES:
        folder_path = os.path.join(SAVE_DIR, class_name)
        os.makedirs(folder_path, exist_ok=True)


def get_next_image_index(class_name):
    folder_path = os.path.join(SAVE_DIR, class_name)

    existing_files = [
        file_name for file_name in os.listdir(folder_path)
        if file_name.endswith(".jpg")
    ]

    return len(existing_files)


def crop_hand(frame, hand_landmarks, padding=40):
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

    hand_crop = cv2.resize(hand_crop, (IMAGE_SIZE, IMAGE_SIZE))

    bbox = (x_min, y_min, x_max, y_max)

    return hand_crop, bbox


def main():
    create_class_folders()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    current_class = None
    last_save_time = 0
    save_interval = 0.15

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            success, frame = cap.read()

            if not success:
                print("Camera frame could not be captured.")
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            hand_crop = None
            bbox = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                hand_crop, bbox = crop_hand(frame, hand_landmarks)

                if bbox is not None:
                    x_min, y_min, x_max, y_max = bbox

                    cv2.rectangle(
                        display_frame,
                        (x_min, y_min),
                        (x_max, y_max),
                        (0, 255, 0),
                        2
                    )

                if current_class is not None and hand_crop is not None:
                    current_time = time.time()

                    if current_time - last_save_time > save_interval:
                        image_index = get_next_image_index(current_class)

                        file_name = f"{current_class}_{image_index:04d}.jpg"
                        file_path = os.path.join(SAVE_DIR, current_class, file_name)

                        cv2.imwrite(file_path, hand_crop)

                        last_save_time = current_time

                        print(f"Saved: {file_path}")

            info_text = "0: idle | 1: index | 2: thumb | 3: open palm |  s: stop | q: quit"
            cv2.putText(
                display_frame,
                info_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            selected_text = f"Current class: {current_class if current_class else 'None'}"
            cv2.putText(
                display_frame,
                selected_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            if hand_crop is not None:
                preview = cv2.resize(hand_crop, (128, 128))
                display_frame[100:228, 20:148] = preview

            cv2.imshow("Collect Gesture Data", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                current_class = None
                print("Stopped saving.")

            if key in KEY_TO_CLASS:
                current_class = KEY_TO_CLASS[key]
                print(f"Selected class: {current_class}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()