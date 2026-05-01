import cv2
import numpy as np


class DrawingCanvas:
    def __init__(self, draw_color=(0, 0, 255), eraser_color=(0, 0, 0)):
        self.canvas = None
        self.prev_x = None
        self.prev_y = None

        self.draw_color = draw_color
        self.eraser_color = eraser_color

        self.draw_thickness = 8
        self.eraser_thickness = 40

    def initialize(self, frame):
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

    def clear(self, frame):
        self.canvas = np.zeros_like(frame)
        self.prev_x = None
        self.prev_y = None

    def reset_previous_point(self):
        self.prev_x = None
        self.prev_y = None

    def draw(self, x, y):
        if self.prev_x is None or self.prev_y is None:
            self.prev_x, self.prev_y = x, y

        cv2.line(
            self.canvas,
            (self.prev_x, self.prev_y),
            (x, y),
            self.draw_color,
            self.draw_thickness
        )

        self.prev_x, self.prev_y = x, y

    def erase(self, x, y):
        cv2.circle(
            self.canvas,
            (x, y),
            self.eraser_thickness // 2,
            self.eraser_color,
            -1
        )

        self.reset_previous_point()

    def merge_with_frame(self, frame):
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)

        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)

        output = cv2.add(frame_bg, canvas_fg)
        return output