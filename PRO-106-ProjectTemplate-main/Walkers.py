import cv2


# Create our body classifier
face_detect=cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,1.1,4)
    # Pass frame to our body classifier
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)
    
    # Extract bounding boxes for any bodies identified
    cv2.imshow("the box",frame)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
