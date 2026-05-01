import cv2

from src.hand_tracker import HandTracker
from src.gesture_rules import detect_gesture
from src.canvas import DrawingCanvas
from cnn.predict import GesturePredictor
from src.utils import crop_hand


class AirDrawingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.hand_tracker = HandTracker()
        self.canvas = DrawingCanvas()

        self.clear_ready = True

        self.use_cnn = True
        self.gesture_predictor = GesturePredictor()

    def run(self):
        while True:
            success, frame = self.cap.read()

            if not success:
                print("Camera frame could not be captured.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            clean_frame = frame.copy()

            self.canvas.initialize(frame)

            frame, results = self.hand_tracker.find_hands(frame, draw=True)

            mode_text = "No hand"

            cnn_label = "None"
            cnn_confidence = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark

                    gesture = detect_gesture(landmarks)

                    if self.use_cnn:
                        hand_crop, bbox = crop_hand(clean_frame, hand_landmarks)

                        if hand_crop is not None:
                            cnn_label, cnn_confidence = self.gesture_predictor.predict(hand_crop)

                    index_tip = landmarks[8]
                    thumb_tip = landmarks[4]

                    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                    thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

                    if gesture == "open_palm":
                        mode_text = "Open palm: Clear screen"

                        if self.clear_ready:
                            self.canvas.clear(frame)
                            self.clear_ready = False

                    else:
                        self.clear_ready = True

                        if gesture == "eraser":
                            mode_text = "Eraser"

                            cv2.circle(
                                frame,
                                (thumb_x, thumb_y),
                                self.canvas.eraser_thickness // 2,
                                (255, 255, 255),
                                2
                            )

                            self.canvas.erase(thumb_x, thumb_y)

                        elif gesture == "drawing":
                            mode_text = "Drawing"

                            cv2.circle(
                                frame,
                                (index_x, index_y),
                                8,
                                self.canvas.draw_color,
                                -1
                            )

                            self.canvas.draw(index_x, index_y)

                        else:
                            mode_text = "Idle"
                            self.canvas.reset_previous_point()

            else:
                self.canvas.reset_previous_point()

            output = self.canvas.merge_with_frame(frame)
            self.draw_ui(output, mode_text, cnn_label, cnn_confidence)

            cv2.imshow("Air Drawing", output)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                self.canvas.clear(frame)

        self.release()

    def draw_ui(self, output, mode_text, cnn_label="None", cnn_confidence=0.0):
        cv2.putText(
            output,
            f"Mode: {mode_text}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"CNN Prediction: {cnn_label} ({cnn_confidence:.2f})",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            output,
            "Index finger: draw | Thumb only: eraser | Open palm: clear | C: clear | Q: quit",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 100, 255),
            2
        )

    def release(self):
        self.cap.release()
        self.hand_tracker.close()
        cv2.destroyAllWindows()