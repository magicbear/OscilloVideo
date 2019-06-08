import soundcard as sc
import numpy as np
import cv2
import threading
import time
import math

terminate_program = 0

class AudioOutputThread(threading.Thread):
    def __init__(self):
        """
        初始化
        """
        threading.Thread.__init__(self)
        self.data = None

    def run(self):
        t1 = time.time()
        # speakers = sc.all_speakers()
        default_speaker = sc.default_speaker()
        while terminate_program == 0:
            if self.data is None:
                time.sleep(0.01)
                continue

            nd = self.data
            self.data = None
            print(time.time()-t1, nd.shape)
            # speakers[2].play(nd, samplerate=96000)
            default_speaker.play(nd, samplerate=96000)

at = AudioOutputThread()
at.start()

data = np.zeros((100000,2), np.float32)
p = 0
cv2.namedWindow("test")
cv2.waitKey(500)

cap = cv2.VideoCapture("test.mp4")
n = 0
fps = 30
pframe_samples = math.floor(96000 / fps)
t = time.time()
while cap.isOpened():
    res, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, (int(0.6 * img.shape[1]),int(0.6 * img.shape[0])))
    edges = cv2.Canny(img,100,200)

    ret,thresh = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_TC89_L1)

    condatas = np.array([], dtype=np.int32)
    dsize = 0
    for i in range(0, len(contours)):
        dsize += contours[i].shape[0]
        condatas=np.append(condatas, contours[i][:,0,:])
    condatas = condatas.reshape((-1,2))

    wh = np.max(img.shape)

    last_p = p

    if dsize > 0:
        for i in range(0, pframe_samples):
            data[last_p+i,0] = condatas[int(i / pframe_samples * dsize), 0] / wh * 1.2 - 0.6
            data[last_p+i,1] = 0.6 - condatas[int(i / pframe_samples * dsize), 1] / wh * 1.2
    else:
        data[last_p:last_p+pframe_samples,:] = 0
    p = last_p+pframe_samples

    if p >= pframe_samples * 2:
        at.data = np.copy(data[0:p,:])
        p = 0

    cv2.imshow("test", thresh)  

    pts = n / fps
    # print("pts => ",pts, (time.time() - t))
    pass_time = (time.time() - t)
    if pts > pass_time:
        if int((pts - pass_time)*1000) > 0:
            cv2.waitKey(int((pts - pass_time)*1000))
        # time.sleep(pts - pass_time)

    n = n + 1

terminate_program = 1