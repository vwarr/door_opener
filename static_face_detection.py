import cv2 as cv
import matplotlib.pyplot as plt

imagepath = 'known_faces/Varun.jpeg'

img = cv.imread(imagepath)

grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = classifier.detectMultiScale(
    grayscale, scaleFactor=1.1, minNeighbors=7, minSize=(400, 400)
)

for x, y, w, h in face:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

img_processed = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.figure(figsize=(20, 10))
plt.imshow(img_processed)
plt.axis('off')

plt.show()


