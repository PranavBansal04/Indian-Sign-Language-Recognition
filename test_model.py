import cv2
import mediapipe as mp

import joblib 

from sklearn.preprocessing import StandardScaler

def new_cord(old_cord,new_origin):
    orgx,orgy = new_origin
    x,y = old_cord
    return [x-orgx,y-orgy]

def distance(x1,y1,x2,y2):
    distance = ((x1-x2)**2+(y1-y2)**2)**0.5
    return round(distance,3)


model = joblib.load("logistica.joblib")


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
   static_image_mode=True,
   max_num_hands=1,
   min_detection_confidence=0.7)


relative_path = "frame0.jpg"
FRAME_WIDTH = 540
FRAME_HEIGHT = 960


path = relative_path
        
frame = cv2.imread(path)
frame = cv2.resize(frame,(FRAME_WIDTH,FRAME_HEIGHT))
image = cv2.flip(frame, 1)
results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))




if not results.multi_hand_landmarks:
    print("None")

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



cv2.imwrite("test_image.png", cv2.flip(annotated_image, 1))

# coordinates = []
# for i in points:
#     coordinates.append(i[0])
#     coordinates.append(i[1])

org_transform = []
new_org = points[0]
# org_transform.append(0)
# org_transform.append(0)
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

flag = 0

if("right" in handedness):
  flag=1

# print(len(org_transform))
# print(handedness)
# print(distances)

# sc_x = StandardScaler()

xtest = org_transform+distances+[flag]
# xtest = sc_x.fit_transform(xtest) 
  

print(model.predict([xtest]))


cv2.imshow('MediaPipe Hands', annotated_image)


hands.close()

