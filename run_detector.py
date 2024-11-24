import video_face_recognition as vfr
import cv2 as cv

video = cv.VideoCapture(0)
initial_frame = video.read()[1]
prev_frame = cv.cvtColor(initial_frame, cv.COLOR_BGR2GRAY)

fps =  1000 #int(video.get(cv.CAP_PROP_FPS))
face_encodings, names = vfr.get_known_faces()
count_to_verify = 2
video_scale = 4
frame_counter = 0

while True:
    if frame_counter == 0:
        curr_frame = cv.cvtColor(video.read()[1], cv.COLOR_BGR2GRAY)
        motion_detected = vfr.motion_detect(prev_frame=prev_frame, curr_frame=curr_frame)
        if motion_detected:
            recognized = vfr.face_recognize(count_to_verify, video, video_scale, face_encodings, names)
            if recognized:
                print('found')
            else:
                print('motion detected but face not found')
        else:
            print('motion not detected')
        prev_frame = curr_frame
    frame_counter = (frame_counter + 1) % fps