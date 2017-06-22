from cv2 import *

'''Cascades downloaded from OpenCV repo 
https://github.com/Itseez/opencv/tree/master/data/haarcascades'''

cascade_face = CascadeClassifier('front_face.xml')
cascade_eye = CascadeClassifier('eye.xml')

captured = VideoCapture(0)

while 1:
    _, pic = captured.read()
    gray = cvtColor(pic, COLOR_BGR2GRAY)
    face = cascade_face.detectMultiScale(gray, 1.3, 5)
    #^ Find faces

    for (x,y,w,h) in face: 
        '''
        Eyes should be found in faces and not lying around outside faces
        '''
        rectangle(pic,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = pic[y:y+h, x:x+w]
        
        eyes = cascade_eye.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    imshow('pic',pic)
    k = waitKey(30) & 0xff
    if k == 27:
        break

captured.release()
destroyAllWindows()