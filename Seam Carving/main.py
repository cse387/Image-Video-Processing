from EnergyFunction import EF,calc_energy,optimal_vertical_seam,show_seam,minimum_seam,carving_horizontal,carving_vertical
from Reduce_Height import *
from Reduce_Width import *
import cv2 as cv
import numpy as np
import sys
import math
from matplotlib import pyplot as plt, cm
imagea=cv.imread('Lefkada.jpg',1)
r=25
c=45
n,m,_=imagea.shape
Mx=horizontal_energy_map(imagea)
My=vertical_energy_map(imagea)
T=np.zeros((r+1,c+1))
def sx(M,i,j,sum1):
    sum1+=M[i,j]
    if(i==0):
        return sum1
    elif(j==0):
        idx=np.argmin(M[i,j:j+2])
        return sx(M,i-1,idx,sum1)
    else:
        idx=j-1+np.argmin(M[i,j-1:j+2])
        return sx(M,i-1,idx,sum1)
My=np.rot90(My,3,(0,1))
'''
sx_sum=[]
sy_sum=[]
for i in range(1,r+1):
    min_idx_y=np.argmin(Mx[n-i-1,:])
    sx_sum.append(sx(Mx,n-i-1,min_idx_y,0))
for j in range(1,c+1):
    min_idx_y=np.argmin(My[m-j-1,:])
    sy_sum.append(sx(My,m-j-1,min_idx_y,0))
    '''
for i in range(1,r+1):
    for j in range(1,c+1):
       min_idx_y=np.argmin(Mx[n-i-1,:])
       sx_sum=sx(Mx,n-i-1,min_idx_y,0)
       min_idx_y=np.argmin(My[m-j-1,:])
       sy_sum=sx(My,m-j-1,min_idx_y,0)
       
       #print('sx',sx_sum,'\nsy',sy_sum)
       T[i,j]=min(T[i-1,j]+min(Mx[n-i-1,:]),T[i,j-1]+min(My[:,m-j-1]))
       #T[i,j]=min(T[i-1,j]+sx_sum,T[i,j-1]+sy_sum)
        #prwto M apo horizontal kai deutero apo vertical seam 
seq=[]
def backTrackofT(T,i,j,seq):
    if(i==0 and j==0):
        return 
    if(T[i-1,j]<=T[i,j-1]):
        seq.append(1)
        return backTrackofT(T,i-1,j,seq)
    else:
        seq.append(0)
        return backTrackofT(T,i,j-1,seq)
backTrackofT(T,r,c,seq)
#evaluation
sys.exit()
gray=EF(imagea,sigma)
evaluate=sum(sum(gray))/gray.size
from skimage.feature import hog
gray=cv.cvtColor(imagea,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)


fd, hog_image = hog(imagea, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
print("hog shape",hog_image.shape)
img=imagea
#cv2.IMREAD_UNCHANGED
image=np.rot90(imagea,1,(0,1))
plt.figure()
plt.imshow(image)

#res = cv.resize(image,None,fx=3, fy=3, interpolation = cv.INTER_CUBIC)
#plt.imshow(res)
#plt.show()
#mag, angle = cv.cartToPolar(Ix, Iy, angleInDegrees=True)

gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
#Blurring image
sigma=0.25
plt.figure()
plt.imshow(EF(gray,sigma))

'''
plt.imshow(calc_energy(image))
plt.show()
blur=cv.GaussianBlur(image,(5,5),0)
plt.figure()
plt.imshow(blur)
plt.show()
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)

'''
B=optimal_vertical_seam(EF(gray,sigma))
#B=np.stack([B]*3,axis=2)
plt.figure()
plt.imshow(show_seam(image,B))
#image=image[B].reshape((617,411,3))
image=np.rot90(image,3,(0,1))


plt.figure()
plt.imshow(carving_vertical(imagea))
plt.figure()
plt.imshow(carving_horizontal(imagea))
plt.show()
#for i in  range(200):
    #imagea=carving_horizontal(imagea)
    #imagea=carving_vertical(imagea)
#res=cv.resize(img,(img.shape[1],img.shape[0]-200), interpolation = cv.INTER_CUBIC)
'''
cv.imwrite('diseyout.png',imagea)

 hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    '''


