import cv2
import mediapipe as mp
import numpy as np


class AirDrawing:
    def __init__(self):
        # initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # drawing variables
        self.drawing_points = []
        self.current_point = None
        self.canvas = None

    def process_frame(self, frame, draw_color=(0, 255, 0), erase_color=(255, 0, 0)):
        # initialize canvas if not created
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

        # process hand landmarks
        results: mp.python.solution_base.SolutionOutputs = self.hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # get index finger tip coordinates
                index_finger_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # check if middle finger is higher than the middle finger's knuckle to draw
                middle_finger_tip = hand_landmarks.landmark[12].y
                middle_finger_mcp = hand_landmarks.landmark[9].y

                # print(middle_finger_tip, middle_finger_mcp)

                # draw if index finger is higher than middle finger's knuckle
                if (
                    index_finger_tip.y < middle_finger_mcp
                    and middle_finger_tip > middle_finger_mcp
                ):
                    # we draw circle at index finger tip when drawing is on
                    cv2.circle(frame, (x, y), 10, draw_color, -1)

                    if self.current_point is None:
                        self.current_point = (x, y)
                    else:
                        # draw line on canvas
                        cv2.line(self.canvas, self.current_point, (x, y), draw_color, 4)
                        self.current_point = (x, y)
                # erasing mode
                elif (
                    abs(index_finger_tip.y - middle_finger_tip) < 0.08
                ) and index_finger_tip.y < middle_finger_mcp:
                    cv2.circle(frame, (x, y), 10, erase_color, -1)

                    if self.current_point is None:
                        self.current_point = (x, y)
                    else:
                        # draw line on canvas
                        cv2.line(self.canvas, self.current_point, (x, y), (0, 0, 0), 50)
                        self.current_point = (x, y)

                else:
                    self.current_point = None

        # add frame to canvas
        return cv2.addWeighted(frame, 1, self.canvas, 1, 1)
