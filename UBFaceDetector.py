'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from helper import show_image
import cv2
import numpy as np
import os
import sys
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
import face_recognition

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(input_path: str) -> dict:
    result_list = []
    images = []
    '''
    Your implementation.
    '''
    directory = input_path

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for dims in faces_rect:
            x = dims[0]
            y = dims[1]
            w = dims[2]
            h = dims[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

            images.append(img)
            result_list.append({"iname": filename, "bbox": [int(x), int(y), int(w), int(h)]})
    # show_image(images[69])
    # plt.imshow(images[0])
    return result_list


'''
K: number of clusters
'''


def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    result_list_labels = []
    images = []
    directory = input_path
    names = []
    encoded = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        #print(filename)
        names.append(filename)
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for dims in faces_rect:
            x = dims[0]
            y = dims[1]
            w = dims[2]
            h = dims[3]
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

            enc = face_recognition.face_encodings(img)
            cropped_images = img[y:y + h, x:x + w]
            resized_images = cv2.resize(cropped_images, (100, 100))
            images.append(resized_images)

        encoded.append(enc[0])
    # print(encoded)
    model = KMeans(int(K))
    model.fit(encoded)
    labels = model.labels_
    unique_labs = np.unique(labels)
    print(unique_labs)
    # list_new =[]
    print(names)

    for x in unique_labs:
        list_temp = []
        names_temp = []
        for m in range(len(list(labels))):
            if x == labels[m]:
                list_temp.append(m)
                names_temp.append(names[m])
        result_list_labels.append({"cluster no": int(x), "elements": list_temp})
        result_list.append({"cluster no": int(x), "elements": names_temp})

    print(labels)
    print(result_list)

    # Logic For Showing the cluster Images

    # for x in result_list_labels:
    #     c_i = []
    #     for m in x['elements']:
    #         c_i.append(images[m])
    #     concatenated_images = cv2.hconcat(c_i)
    #     show_image(concatenated_images, delay= 0)
    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""
