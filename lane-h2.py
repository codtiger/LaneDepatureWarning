import numpy as np
from numpy.polynomial import polynomial as poly
from scipy import signal
from matplotlib import pyplot as plt
import cv2
import datetime
import time


def imshow(I):
    # plt.imshow(I)
    # plt.show()
    # return
    
    #I = np.float32(I)
    #I /= I.max()

    
    cv2.imshow('tools_imshow', I)
    k = cv2.waitKey()
    if k & 0xFF == ord('q'):
        exit()
    
    
    

def get_threshold(I):
    w, h = I.shape
    threshold = 0.05 * w * h
    sum = w * h
    hist, _ = np.histogram(I, 256)
    cumhist=np.cumsum(hist)
    if np.any(cumhist>sum-threshold):
        return np.argmax(cumhist>sum-threshold)
    else:
        return 255
    # for i in range(256):
    #     if (sum-hist[i]) < threshold:
    #         return i
    #     sum -= hist[i]
    # return 255

def find_boundaries(a, center, left, right):
    left_bound = center
    right_bound = center
    for i in range(center-left):
        if a[center - i] < a[left_bound]:
            left_bound = center - i
    for i in range(right - center):
        if a[center+i] < a[right_bound]:
            right_bound = center+i
    return left_bound, right_bound


def ransac(I):
    I = I.T
    w, h = I.shape
    R = I.ravel() / I.sum()
    iterations = 50 # move out
    angle_ratio = 0.5
    lenght_ratio = 0.3
    max_score = -1
    points = []
    # rp = np.random.choice(len(R), 3, False, R)
    # x = (rp / h).reshape(3, 1)
    # y = (rp % h).reshape(3, 1)
    # p = np.concatenate((x, y), axis=1)
    # p = p[p[:, 1].argsort()]
    # p = np.array([(p[0][0], p[0][1]), (p[1][0], p[1][1]), (p[2][0], p[2][1])])
    #this is done without for,do not forget to delete the for
    rp=np.random.choice(len(R),(iterations,3),True,R)
    x=(rp/h).reshape(3,iterations)
    y=(rp/h).reshape(3,iterations)
    #p=np.concatenate((x,y),axis=1)
    p=np.zeros((3,iterations,2),dtype = np.int32)
    p[:,:,0],p[:,:,1]=x,y
    yargsorted=np.argsort(p[:,:,1],axis=0)
    p[:,:,0],p[:,:,1]=p[yargsorted,np.arange(iterations),0],p[yargsorted,np.arange(iterations),1]
    # yargsorted=np.argsort(pK1[:,np.arange(iterations)%2==1],axis=0)
    # yargsorted=np.repeat(yargsort,2,axis=1)
    # p=p[yargsorted,np.arange(iterations)]
    
    s = get_spline_points(I, p)
    score = s * (1 + angle_ratio * get_angle_score(p) +
        lenght_ratio * ((p[2, :, 1]-p[0, :, 1]).astype(np.float32)/h - 1))
    index = score.argmax()
    p = p[:,index,:]
    # if score > max_score:
    #     points = p
    #     max_score = score
    return np.array([(p[0,0], p[0,1]), (p[1,0], p[1,1]), (p[2,0], p[2,1])])


def get_angle_score(p):
    diffx = p[1:, :, 0] - p[:-1, :, 0]
    diffy = p[1:, :, 1] - p[:-1, :, 0]
    distance = np.sqrt(diffx ** 2 + diffy ** 2)
    cosx = diffx / distance
    sinx= diffy / distance
    cos_theta1=cosx[0] * cosx[1] + sinx[0] * sinx[1]
    #cosx = [0., 0., 0.]
    #sinx = [0., 0., 0.]
    #for i in range(2):
        #cosx[i] = (p[i+1][0] - p[i][0]) / np.sqrt((p[i+1][0] - p[i][0]) ** 2 + (p[i+1][1] - p[i][1]) ** 2)
        #sinx[i] = (p[i+1][1] - p[i][1]) / np.sqrt((p[i+1][0] - p[i][0]) ** 2 + (p[i+1][1] - p[i][1]) ** 2)
    #cos_theta1 = cosx[0] * cosx[1] + sinx[0] * sinx[1]
    return cos_theta1 - 1


def get_spline_points(I, points):
    #counts = 0
    counts=(points[1:]-points[:-1])**2
    counts=(np.sum(np.sqrt(counts[: ,:, 0]+counts[:, :, 1]), axis = 0))
    ls=list(1/counts)
    #for i in range(len(points) - 1):
        #counts += np.sqrt(((points[i]-points[i+1])**2).sum())
    t=np.array(map(lambda x:np.arange(0,1,x),ls))
    t = t.reshape((t.shape[0], 1))
    u = 1 - t
    x = (u ** 2) * points[0,:,0:1] + 2 * u * t * points[1,:,0:1] + (t ** 2) * points[2,:,0:1]
    y = (u ** 2) * points[0,:,1:2] + 2 * u * t * points[1,:,1:2] + (t ** 2) * points[2,:,1:2]
    x, y = np.squeeze(x), np.squeeze(y)
    x = map(lambda k :k.astype(np.int), x)
    y = map(lambda k :k.astype(np.int), y)
    score = np.array(map (lambda x,y:I[x,y].sum(),x,y))
    #score = I[x, y].sum()
    return score


def draw_spline(I, points, color=(150, 0, 200)):
    counts = 0
    for i in range(len(points) - 1):
        counts += np.sqrt(((points[i] - points[i + 1]) ** 2).sum())
    t = np.arange(0, 1, 1 / counts)
    t = t.reshape((t.shape[0], 1))
    u = 1 - t
    x = (u ** 2) * points[0][0] + 2 * u * t * points[1][0] + (t ** 2) * points[2][0]
    y = (u ** 2) * points[0][1] + 2 * u * t * points[1][1] + (t ** 2) * points[2][1]
    x = x.astype(np.int)
    y = y.astype(np.int)
    I[y, x] = color


cap = cv2.VideoCapture('testdrive15.mp4')

src_pts = np.array([(416, 576), (927, 576), (654, 426), (776, 426)]).astype(np.float32)
dst_pts = np.array([(416, 500 * 1.5 - 200), (927, 500 * 1.5 - 200), (416, 426 * 1.0 - 350), (927, 426 * 1.0 - 350)]).astype(np.float32)
src_pts = src_pts / 1.6
dst_pts= dst_pts / 1.6
open_element = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 20))
close_element = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
pmat = cv2.getPerspectiveTransform(src=src_pts, dst=dst_pts)
ipmat = np.linalg.inv(pmat)

l = []

ds_rate = 2
m = 1920
n = 1080

frame0 = np.zeros((m,n,3), dtype=np.uint8);

m /= ds_rate
n /= ds_rate

gray = np.zeros((m,n), dtype=np.uint8);



while cap.isOpened():
    start1 = time.time() 

    _, frame = cap.read() # read grayscale, read certrain resoution

    

    #frame = cv2.resize(frame, (800, 450), cv2.INTER_NEAREST) # use NN, integer dowsample rate
    frame = frame[::ds_rate,::ds_rate]

    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # check gray=frame[:,:,0]
    
    # cv2.imshow('Frame', frame)




    


    gray = cv2.warpPerspective(gray, M=pmat, dsize=(n,m), flags=cv2.INTER_NEAREST)

    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    sx = cv2.Sobel(gray, cv2.CV_16S, 2, 0, ksize = 1) # give dst as argument

    
    
    
    
    t0 = time.time()

    
    
    
    
    l.append(time.time() - t0)
    if len(l) == 100:
        break
    continue



    #cv2.Sobel(gray, cv2.CV_16S, 2, 0, ksize = 5,dst=sx)
    # sx = lineFilter(gray)
    sx = np.abs(sx) # use inplace abs
    #sx = sx/ sx.max() # sx = sx*255/max
    # see if max could be computed out of loop
    # cv2.imshow('normalized', sx)

    
    sx = cv2.morphologyEx(sx, cv2.MORPH_CLOSE, close_element)
    sx = cv2.morphologyEx(sx, cv2.MORPH_OPEN, open_element)
    # cv2.imshow('morpholo', sx)
    # C = sx.copy() # remove this
    sx = (sx * 255).astype(np.uint8) # not needed anymore
    thresh = get_threshold(sx) #?
    C[sx < thresh] = 0 # use cv2.threshold
    # cv2.imshow('C', C)    
    end = time.time()
    print "the threshholding:{}".format((end-start).total_seconds())
    
    
    s = C.sum(axis=0)
    g = signal.gaussian(35, std=13)
    # g = np.ones(20)
    y = np.convolve(s, g / g.sum(), mode='valid') # use integer g*255/g.max()
    # see if filtering an be done faster
    ext = signal.argrelextrema(y, np.greater, order=55)[0]
    exth = [0]
    img = cv2.cvtColor(C.astype(np.float32), cv2.COLOR_GRAY2BGR) # not needed
    for e in ext:
        if y[e] > 15:
            exth.append(e)
            cv2.line(img, (e+35/2, 1), (e+35/2, 700), (0, 255, 0), 2)
    exth.append(y.shape[0])
    boundaries = []
    pts = [0., 0.]
    start = time.time() 
    for i in range(1, len(exth)-1):
        l, r = find_boundaries(y, exth[i], exth[i-1], exth[i+1])
        l += 35/2
        r += 35/2
        points = ransac(C[:, l:r])
        dp = []
        for p in points:
            xp, yp = p
            xp += l
            dp.append(np.array([xp, yp]))
            cv2.circle(img, (xp, yp), 3, (255, 0, 0), 2)
        draw_spline(img, dp)
        boundaries.append(l)
        boundaries.append(r)
    end = time.time()
    print (end - start).total_seconds()
    # print (end - start).total_seconds()
    for m in boundaries:
        cv2.line(img, (m, 1), (m, 700), (0, 0, 255), 2)
    end = time.time()
    # cv2.imshow('Borders', img)
    key = cv2.waitKey(33)    
    if key == ord(' '):
    	plt.plot(s)
    	plt.plot(y)
    	plt.show()
    if key == ord('q'):
        break
        


print np.array(l).mean()
