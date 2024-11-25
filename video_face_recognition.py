import face_recognition as fr
import numpy as np
import cv2 as cv
import os

def get_known_faces():
    known_faces = 'known_faces'
    names = os.listdir(known_faces)
    face_encodings = []

    for i, name in enumerate(names):
        face = fr.load_image_file(f'{known_faces}//{name}')
        encodings = fr.face_encodings(face)
        if len(encodings) != 0:
            face_encodings.append(encodings[0])
        names[i] = name.split(".")

    return face_encodings, names

def face_recognize(count_to_verify, video, scale, face_encodings, names) -> bool:
    # video = cv.VideoCapture(0)
    # scale = 4

    recognized_frames = 0

    for i in range(count_to_verify):
        success, image = video.read()
        
        if not success:
            break

        processed_image = cv.resize(image, (int(image.shape[1]/scale), (int(image.shape[0]/scale))))
        processed_image = cv.cvtColor(processed_image, cv.COLOR_BGR2RGB)

        locations = fr.face_locations(processed_image, model="hog")
        unknown_encodings = fr.face_encodings(processed_image, locations)

        for face_encoding, face_location in zip(unknown_encodings, locations):
            distances = fr.face_distance(face_encodings, face_encoding)
            best_match = np.argmin(distances)

            if distances[best_match] < 0.5:
                name = names[best_match][0]

                top, right, bottom, left = face_location
                                
                cv.rectangle(image, (left * scale, top * scale), (right * scale, bottom * scale), (0, 0, 255), 5)
                cv.putText(image, name, (left * scale, bottom * scale + 20), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)     

                recognized_frames += 1   
        
        cv.imshow("frame", image)
        cv.waitKey(1)

    return recognized_frames >= count_to_verify * 0.9

def motion_detect(prev_frame: cv.typing.MatLike, curr_frame: cv.typing.MatLike) -> bool:
    diff = cv.absdiff(prev_frame, curr_frame)
    diff = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)[1]

    motion_detected = cv.countNonZero(diff) > 5000
    return motion_detected







