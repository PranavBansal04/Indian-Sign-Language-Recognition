import csv
import os
import cv2
import mediapipe as mp

def new_cord(old_cord,new_origin):
    orgx,orgy = new_origin
    x,y = old_cord
    return [x-orgx,y-orgy]

def distance(x1,y1,x2,y2):
    distance = ((x1-x2)**2+(y1-y2)**2)**0.5
    return round(distance,3)



mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
   static_image_mode=True,
   max_num_hands=1,
   min_detection_confidence=0.7)


relative_path = "dataset/"
FRAME_WIDTH = 540
FRAME_HEIGHT = 960



dataFile = open("dataset.csv","a",newline='')
writer = csv.writer(dataFile)
                
file2 = open("dataset2.csv","a",newline='')
w2 = csv.writer(file2)

ar = ["0x","0y","1x","1y","2x","2y","3x","3y","4x","4y","5x","5y","6x","6y","7x","7y","8x","8y","9x","9y","10x",
"10y","11x","11y","12x","12y","13x","13y","14x","14y","15x","15y","16x","16y","17x","17y","18x","18y","19x",
"19y","20x","20y"]


for j in range(1,21):
    ar.append(str(0)+"_"+str(j))

ar.append("Handedness")
ar.append("Label")
ar.append("Location")

writer.writerow(ar)
w2.writerow(ar)

for sub in os.listdir(relative_path):
    print(sub)
    if(sub in ['0','1','2','3','4','5','6','7','8','9','A']):
        continue
    os.mkdir("dataset_keypoints/"+sub)
    for files in os.listdir(relative_path+sub+"/"):
        
        path = relative_path+sub+"/"+files
        
        frame = cv2.imread(path)
        frame = cv2.resize(frame,(FRAME_WIDTH,FRAME_HEIGHT))
        image = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

   
        

        if not results.multi_hand_landmarks:
            continue

        handedness = results.multi_handedness[0].classification[0].label

        annotated_image = image.copy()

        points = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                points.append([x,y])
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
 

        cv2.imwrite("dataset_keypoints/"+sub+"/"+files, cv2.flip(annotated_image, 1))

        coordinates = []
        for i in points:
            coordinates.append(i[0])
            coordinates.append(i[1])

        org_transform = []
        new_org = points[0]
        org_transform.append(0)
        org_transform.append(0)
        for i in range(1,len(points)):
            point = points[i]
            temp = new_cord(point,new_org)
            org_transform.append(temp[0])
            org_transform.append(temp[1])
        

        distances = []
        
        p1=points[0]
        for j in range(1,len(points)):
            p2=points[j]
            distances.append(distance(p1[0],p1[1],p2[0],p2[1]))

        writer.writerow(org_transform+distances+[handedness]+[sub]+[sub+"/"+files])
        
        w2.writerow(coordinates+distances+[handedness]+[sub]+[sub+"/"+files])
            


hands.close()

dataFile.close()
file2.close()
                
            