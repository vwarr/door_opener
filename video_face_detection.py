import cv2 as cv

classifier = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

video = cv.VideoCapture(0)

def detect_bounds(video):
    gray = cv.cvtColor(video, cv.COLOR_RGB2BGR)
    face = classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(300, 300)
    )
    for x, y, w, h in face:
        cv.rectangle(video, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return face

while True:
    result, frame = video.read()
    if result is False:
        break
    face = detect_bounds(frame)
    cv.imshow(
        "Test", frame
    )
    if (cv.waitKey(1) & 0xFF) == ord('q'):
        break

video.release()
cv.destroyAllWindows()
