import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if ret is False:
        break

    # crop the frame
    roi = frame[60:340, 200:450]
    rows, cols, _ = roi.shape

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 75, 245, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda xx: cv2.contourArea(xx), reverse=True)

    # applying filters
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_color = np.array([7, 115, 60], np.uint8)
    upper_color = np.array([15, 220, 255], np.uint8)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 21)
    contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours2) > 0:

        for c in contours2:
            # to find the max area
            cnt = max(contours2, key=lambda xx: cv2.contourArea(xx))
            epsilon = 0.0009 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            cv2.circle(roi, extRight, 5, (0, 0, 255), 2)
            cv2.circle(roi, extLeft, 5, (0, 0, 255), 2)
            cv2.circle(roi, extTop, 5, (0, 0, 255), 2)
            cv2.circle(roi, extBot, 5, (0, 0, 255), 2)

    for cnt in contours:
        # draw rectangle to the face
        (x, y, w, h) = cv2.boundingRect(cnt)

        cv2.rectangle(roi, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.line(roi, (x+int(w/2), 0), (x+int(w/2), rows), (0, 255, 0), 2)  # vertical
        cv2.line(roi, (0, y+int(h/2)), (cols, y+int(h/2)), (0, 255, 0), 2)  # horizontal
        break

    frame[60:340, 200:450] = roi
    cv2.imshow("mask", mask)
    cv2.imshow("frame", frame)

    # press q to close the program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
