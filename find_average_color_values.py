''' 
Facial Landmark Detection in Python with OpenCV
https://github.com/Danotsonof/facial-landmark-detection/blob/master/facial-landmark.py

Detection from web cam
'''

# Import Packages
import cv2
import os
import urllib.request as urlreq
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

def black_out(points_list, img):
    int32_boundaries = np.array(points_list.reshape((-1,1,2)), np.int32)

    cv2.polylines(img, [int32_boundaries], True, (0,0,255), 5)
    mask = np.ones(img.shape[:2],np.uint8)
    cv2.fillPoly(mask, [int32_boundaries], (0,0,0))
    img = cv2.bitwise_and(img, img, mask = mask)
    return img


haarcascade_clf = "haarcascades/haarcascade_frontalface_alt2.xml"

# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier(haarcascade_clf)

# save facial landmark detection model's name as LBFmodel
LBFmodel_file = "haarcascades/LFBmodel.yaml"

# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel_file)

# get image from webcam
import os
video = cv2.VideoCapture('/Users/pranaviboyalakuntla/Documents/Stanford/W_23/EE 269/identifying_heart_rate_from_video/ubfc-ppg dataset/subject1/vid.avi')

averages = {}
frame_num = 0
while(True):
    # read webcam
    # video.set(cv2.CAP_PROP_POS_FRAMES, 40)
    ret, image = video.read()
    if ret == True:
        # image = cv2.imread("opencv_video_capture.jpg")

        # convert frame to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5)

        if faces is not ():
            for (x,y,w,d) in faces:
                # Detect landmarks on "gray"
                _, landmarks = landmark_detector.fit(gray, np.array(faces))

                for landmark in landmarks:
                    for x,y in landmark[0][0:26]:
                        # display landmarks on "frame/image,"
                        # with blue colour in BGR and thickness 2
                        cv2.circle(image, (int(x), int(y)), 1, (255, 0, 0), 2)

            # keep these
            jaw = landmarks[0][0][0:17]
            eyebrow = np.flip(landmarks[0][0][17:27], axis=0)

            # black out these
            left_eye = landmarks[0][0][36:42]
            right_eye = landmarks[0][0][42:48]
            mouth = landmarks[0][0][48:60]
            
            # keep these
            landmarks_ordered = np.vstack((jaw, eyebrow))
            reshaped_landmark_boundaries = landmarks_ordered.reshape((-1, 1, 2))
            int32_boundaries = np.array(reshaped_landmark_boundaries, np.int32)

            cv2.polylines(image, [int32_boundaries], True, (0,0,255), 5)
            mask = np.zeros(image.shape[:2],np.uint8)
            cv2.fillPoly(mask, [int32_boundaries], (255,255,255))
            image = cv2.bitwise_and(image, image, mask = mask)

            # black out operations
            for pts in [left_eye, right_eye, mouth]:
                image = black_out(pts, image)

            number_of_non_black_pixels = np.sum(image != 0) 
            normalized_val = np.sum(image)/number_of_non_black_pixels
            image = 1/normalized_val * image
            colors = cv2.split(image)

            for idx, color in enumerate(colors):
                num_black_pix = np.sum(color != 0)
                avg = np.sum(color)/num_black_pix
                if idx not in averages:
                    averages[idx] = [avg]
                else:
                    averages[idx].append(avg)

            print(frame_num)
            frame_num += 1

            # number_of_non_black_pixels = np.sum(color != 0) 
            # average_r = np.sum(R)/number_of_non_black_pixels
            # average_red.append(average_r)
            
            # number_of_non_black_pixels = np.sum(G != 0) 
            # average_g = np.sum(G)/number_of_non_black_pixels
            # average_green.append(average_r)

            # number_of_non_black_pixels = np.sum(B != 0) 
            # average_b = np.sum(B)/number_of_non_black_pixels
            # average_blue.append(average_r)

    else:
        break

        # cv2.imshow("masked frame", image)
        # cv2.waitKey(0)
        
            # Show image
            # cv2.imshow("frame", (0, 0, R))
            # cv2.waitKey(0)
        # break

        # terminate the capture window
        # if cv2.waitKey(20) & 0xFF  == ord('q'):
        #     webcam_cap.release()
        #     cv2.destroyAllWindows()
        #     break
with open('test_avi.pickle', 'wb') as handle:
    pkl.dump(averages, handle)