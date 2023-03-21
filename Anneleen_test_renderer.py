import cv2
import numpy as np
import mediapipe_utils as mpu

# LINE_BODY and COLORS_BODY are used when drawing the skeleton in 3D.
rgb = {"right": (0, 1, 0), "left": (1, 0, 0), "middle": (1, 1, 0)}
LINES_BODY = [[9, 10], [4, 6], [1, 3],
              [12, 14], [14, 16], [16, 20], [20, 18], [18, 16],
              [12, 11], [11, 23], [23, 24], [24, 12],
              [11, 13], [13, 15], [15, 19], [19, 17], [17, 15],
              [24, 26], [26, 28], [32, 30],
              [23, 25], [25, 27], [29, 31]]

COLORS_BODY = ["middle", "right", "left",
               "right", "right", "right", "right", "right",
               "middle", "middle", "middle", "middle",
               "left", "left", "left", "left", "left",
               "right", "right", "right", "left", "left", "left"]
COLORS_BODY = [rgb[x] for x in COLORS_BODY]


class Anneleen_test_renderer:
    def __init__(self,
                 tracker,
                 output=None, number=0):
        self.tracker = tracker
        self.fram = None
        self.pause = False
        self.elapsedTime = None
        self.lslInfo = None
        self.number = number
        # Rendering flags
        self.show_rot_rect = False
        self.show_landmarks = True
        self.show_score = False
        self.show_fps = True

        self.show_xyz_zone = self.show_xyz = self.tracker.xyz

        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output, fourcc, tracker.video_fps, (tracker.img_w, tracker.img_h))

    def is_present(self, body, lm_id):
        return body.presence[lm_id] > self.tracker.presence_threshold

    def turnOffLandMarks(self):
        self.show_landmarks = False

    def turnOnLandMarks(self):
        self.show_landmarks = True

    def setElapsedTime(self, et):
        self.elapsedTime = et

    def setLSLInfo(self, lslInfo):
        self.lslInfo = lslInfo

    def draw_landmarks(self, body):
        if self.show_rot_rect:
            cv2.polylines(self.frame, [np.array(body.rect_points)], True, (0, 255, 255), 2, cv2.LINE_AA)
        if self.show_landmarks:
            list_connections = LINES_BODY
            lines = [np.array([body.landmarks[point, :2] for point in line]) for line in list_connections if
                     self.is_present(body, line[0]) and self.is_present(body, line[1])]
            cv2.polylines(self.frame, lines, False, (255, 180, 90), 10, cv2.LINE_AA)

            # for i,x_y in enumerate(body.landmarks_padded[:,:2]):
            for i, x_y in enumerate(body.landmarks[:self.tracker.nb_kps, :2]):
                if self.is_present(body, i):
                    if i > 10:
                        color = (0, 255, 0) if i % 2 == 0 else (0, 0, 255)
                    elif i == 0:
                        color = (0, 255, 255)
                    elif i in [4, 5, 6, 8, 10]:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.circle(self.frame, (x_y[0], x_y[1]), 8, color, -11)
        if self.show_score:
            h, w = self.frame.shape[:2]
            cv2.putText(self.frame, f"Landmark score: {body.lm_score:.2f}",
                        (20, h - 60),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        if self.show_xyz and body.xyz_ref:
            x0, y0 = body.xyz_ref_coords_pixel.astype(np.int)
            x0 -= 50
            y0 += 40
            cv2.rectangle(self.frame, (x0, y0), (x0 + 100, y0 + 85), (220, 220, 240), -1)
            cv2.putText(self.frame, f"X:{body.xyz[0] / 10:3.0f} cm", (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_PLAIN, 1,
                        (20, 180, 0), 2)
            cv2.putText(self.frame, f"Y:{body.xyz[1] / 10:3.0f} cm", (x0 + 10, y0 + 45), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 0), 2)
            cv2.putText(self.frame, f"Z:{body.xyz[2] / 10:3.0f} cm", (x0 + 10, y0 + 70), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 255), 2)
        if self.show_xyz_zone and body.xyz_ref:
            # Show zone on which the spatial data were calculated
            cv2.rectangle(self.frame, tuple(body.xyz_zone[0:2]), tuple(body.xyz_zone[2:4]), (180, 0, 180), 2)

    def draw(self, frame, body):
        if not self.pause:
            self.frame = frame

            cv2.putText(self.frame, str(self.elapsedTime),
                        (52, 82),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

            cv2.putText(self.frame, str(self.elapsedTime),
                        (50, 80),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            cv2.putText(self.frame, str(self.lslInfo),
                        (52, 102),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

            cv2.putText(self.frame, str(self.lslInfo),
                        (50, 100),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            if body:
                self.draw_landmarks(body)
            self.body = body
        elif self.frame is None:
            self.frame = frame
            self.body = None
        return self.frame

    def exit(self):
        if self.output:
            self.output.release()

    def waitKey(self, delay=1):
        if self.show_fps:
            self.tracker.fps.draw(self.frame, orig=(50, 50), size=1, color=(240, 180, 100))
        cv2.imshow(f"Blazepose_{self.number}", self.frame)
        if self.output:
            self.output.write(self.frame)
        key = cv2.waitKey(delay)
        if key == 32:
            # Pause on space bar
            self.pause = not self.pause
        elif key == ord('r'):
            self.show_rot_rect = not self.show_rot_rect
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('s'):
            self.show_score = not self.show_score
        elif key == ord('f'):
            self.show_fps = not self.show_fps
        elif key == ord('x'):
            if self.tracker.xyz:
                self.show_xyz = not self.show_xyz
        elif key == ord('z'):
            if self.tracker.xyz:
                self.show_xyz_zone = not self.show_xyz_zone
        return key