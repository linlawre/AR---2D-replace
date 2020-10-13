import cv2
import numpy as np

cap = cv2.VideoCapture(0)
imgTarget = cv2.imread("./marker_1.png")
myVid = cv2.VideoCapture("./aaa.mp4")

def empty(a):
    pass

cv2.namedWindow("Brightness")
cv2.resizeWindow("Brightness",300,400)
cv2.createTrackbar("Brightness","Brightness",180,255,empty)

tracker = cv2.TrackerMedianFlow_create()

detection = False
frameCounter = 0
 
success, img_Video = myVid.read()
hT,wT,cT = imgTarget.shape
# find the feature for the marker
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget,None)

 
while True:
    # change the brightness
    timer = cv2.getTickCount()
    cameraBrightness = cv2.getTrackbarPos("Brightness", "Brightness")
    cap.set(10, cameraBrightness)


    sucess, imgWebcam = cap.read()
    matrix = imgWebcam.copy()
    imgWarp = imgWebcam.copy()
    imgAug = imgWebcam.copy()
    maskNew = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    # imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)
 
    if detection == False:
        # reset the counter to 0
        myVid.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter = 0
    else:
        # find the marker then start to play the video
        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, img_Video = myVid.read()
        img_Video = cv2.resize(img_Video, (wT, hT))
        img_Video = cv2.rotate(img_Video, cv2.ROTATE_90_COUNTERCLOCKWISE)
 
    
    # compare all the features if find the same feature then add to good
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good =[]
    for m,n in matches:
        if m.distance < 0.75 *n.distance:
            good.append(m)


    # more the 20 features are the same, then we find the marker
    if len(good) > 20:

        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5)
     

        # find the boundry and draw it
        pts = np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,matrix)
        img2 = cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,0,0),3)

        # wrap the video with black screen
        imgWarp = cv2.warpPerspective(img_Video,matrix, (imgWebcam.shape[1],imgWebcam.shape[0]))
        print(matrix)
        maskNew = np.zeros((imgWebcam.shape[0],imgWebcam.shape[1]),np.uint8)
        cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255))
        maskInv = cv2.bitwise_not(maskNew)
        
        # dig out the marker
        imgAug = cv2.bitwise_and(imgAug,imgAug,mask = maskInv)
        # combine the hole and video
        imgAug = cv2.bitwise_or(imgWarp,imgAug)


        
    fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
    cv2.putText(imgAug, str(int(fps)), (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)


    cv2.imshow('AR Result1', imgAug)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frameCounter +=1