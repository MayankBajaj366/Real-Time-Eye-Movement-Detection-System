import cv2
import mediapipe as mp
import numpy as np
import csv
import time

# Initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmarks
LEFT_EYE_H = [33, 133]
RIGHT_EYE_H = [362, 263]

LEFT_EYE_V = [159, 145]
RIGHT_EYE_V = [386, 374]

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

cap = cv2.VideoCapture(0)

# ================= RECORDING SETUP =================
recording = False
out = None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # FIXED codec
fps = 20
frame_size = (1000, 600)

# ================= CSV DATA SAVE =================
data_file = open('eye_data.csv', 'w', newline='')
writer = csv.writer(data_file)
writer.writerow(["X", "Y"])

# Wave storage
history_len = 200
x_history = []
y_history = []

# Smoothing
alpha = 0.2
smooth_dx = 0
smooth_dy = 0

def get_iris_center(landmarks, iris_points):
    x = [landmarks[i].x for i in iris_points]
    y = [landmarks[i].y for i in iris_points]
    return np.mean(x), np.mean(y)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark

            # Iris centers
            left_iris = get_iris_center(lm, LEFT_IRIS)
            right_iris = get_iris_center(lm, RIGHT_IRIS)

            # ---------- HORIZONTAL ----------
            l_left = lm[LEFT_EYE_H[0]].x
            l_right = lm[LEFT_EYE_H[1]].x
            r_left = lm[RIGHT_EYE_H[0]].x
            r_right = lm[RIGHT_EYE_H[1]].x

            avg_x = ((left_iris[0] - l_left) / (l_right - l_left) +
                     (right_iris[0] - r_left) / (r_right - r_left)) / 2

            # ---------- VERTICAL ----------
            l_top = lm[LEFT_EYE_V[0]].y
            l_bottom = lm[LEFT_EYE_V[1]].y
            r_top = lm[RIGHT_EYE_V[0]].y
            r_bottom = lm[RIGHT_EYE_V[1]].y

            avg_y = ((left_iris[1] - l_top) / (l_bottom - l_top) +
                     (right_iris[1] - r_top) / (r_bottom - r_top)) / 2

            # Boost vertical sensitivity
            avg_y = (avg_y - 0.5) * 1.2 + 0.5

            avg_x = np.clip(avg_x, 0, 1)
            avg_y = np.clip(avg_y, 0, 1)

            # Offset from center
            dx = avg_x - 0.5
            dy = avg_y - 0.5

            # Smoothing
            smooth_dx = (1 - alpha) * smooth_dx + alpha * dx
            smooth_dy = (1 - alpha) * smooth_dy + alpha * dy

            # Dead zone
            if abs(smooth_dx) < 0.03:
                smooth_dx = 0
            if abs(smooth_dy) < 0.03:
                smooth_dy = 0

            # Amplification
            norm_x = np.clip(smooth_dx * 4, -1, 1)
            norm_y = np.clip(smooth_dy * 4, -1, 1)

            x_history.append(norm_x)
            y_history.append(norm_y)

            # Save data
            writer.writerow([norm_x, norm_y])

            if len(x_history) > history_len:
                x_history.pop(0)
                y_history.pop(0)

            # Direction detection
            direction = "CENTER"

            if abs(smooth_dx) > abs(smooth_dy):
                if smooth_dx < -0.08:
                    direction = "LEFT"
                elif smooth_dx > 0.08:
                    direction = "RIGHT"
            else:
                if smooth_dy < -0.08:
                    direction = "UP"
                elif smooth_dy > 0.08:
                    direction = "DOWN"

            cv2.putText(frame, direction, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # ================= DASHBOARD =================
    canvas = np.zeros((600, 1000, 3), dtype=np.uint8)

    # Resize camera
    frame_resized = cv2.resize(frame, (700, 600))
    canvas[0:600, 0:700] = frame_resized

    graph_h = 300
    graph_w = 300

    # -------- X GRAPH --------
    graph_x = np.zeros((graph_h, graph_w, 3), dtype=np.uint8)

    for i in range(0, graph_w, 50):
        cv2.line(graph_x, (i, 0), (i, graph_h), (40, 40, 40), 1)
    for i in range(0, graph_h, 50):
        cv2.line(graph_x, (0, i), (graph_w, i), (40, 40, 40), 1)

    cv2.line(graph_x, (0, graph_h//2), (graph_w, graph_h//2), (120, 120, 120), 1)

    for i in range(1, len(x_history)):
        cv2.line(graph_x,
                 (i-1, int(graph_h/2 - x_history[i-1]*120)),
                 (i, int(graph_h/2 - x_history[i]*120)),
                 (0, 255, 0), 2)

    cv2.putText(graph_x, "X Movement (L/R)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # -------- Y GRAPH --------
    graph_y = np.zeros((graph_h, graph_w, 3), dtype=np.uint8)

    for i in range(0, graph_w, 50):
        cv2.line(graph_y, (i, 0), (i, graph_h), (40, 40, 40), 1)
    for i in range(0, graph_h, 50):
        cv2.line(graph_y, (0, i), (graph_w, i), (40, 40, 40), 1)

    cv2.line(graph_y, (0, graph_h//2), (graph_w, graph_h//2), (120, 120, 120), 1)

    for i in range(1, len(y_history)):
        cv2.line(graph_y,
                 (i-1, int(graph_h/2 - y_history[i-1]*120)),
                 (i, int(graph_h/2 - y_history[i]*120)),
                 (255, 0, 0), 2)

    cv2.putText(graph_y, "Y Movement (U/D)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # Place graphs
    canvas[0:300, 700:1000] = graph_x
    canvas[300:600, 700:1000] = graph_y

    # Recording indicator
    if recording:
        cv2.circle(canvas, (950, 50), 10, (0, 0, 255), -1)
        cv2.putText(canvas, "REC", (900, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Show dashboard
    cv2.imshow("Eye Tracking Dashboard", canvas)

    # Save video
    if recording and out is not None:
        out.write(canvas)

    key = cv2.waitKey(1)

    # Toggle recording
    if key == ord('r'):
        recording = not recording
        if recording:
            filename = f"eye_tracking_{int(time.time())}.mp4"
            print("Recording Started:", filename)
            out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

            if not out.isOpened():
                print("ERROR: VideoWriter failed to open!")
        else:
            print("Recording Stopped")
            out.release()

    if key == 27 or key == ord('q'):
        break

# Cleanup
cap.release()

if out is not None:
    out.release()

data_file.close()
cv2.destroyAllWindows()
