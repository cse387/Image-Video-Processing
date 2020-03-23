from scipy.ndimage import filters , convolve
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.feature import hog
import time
import sys
sys.setrecursionlimit(10000)
sigma=0.5
ksize=68
sys.setrecursionlimit(10000)
sigma=6
'''
filters.gaussian_filter(gray, (sigma,sigma), (0,1), imx)
filters.gaussian_filter(gray, (sigma,sigma), (1,0), imy)
'''

def Gaussian_Derivatives(gray,sigma):
    #create gaussian kernel 
    B = np.array([[[-1,-1],[-1,0],[-1,1]],
                [[ 0,-1],[ 0,0],[ 0,1]],
                [[ 1,-1],[ 1,0],[ 1,1]]])
    pi = math.pi
    D = -(B[:,:,0]**2+B[:,:,1]**2)/(2*(sigma**2))
    D = np.exp(D)/(2*(sigma**2)*pi)
    #create the derivatives
    bx = -2*B[:,:,0]*D/(2*(sigma**2))
    B = np.rot90(B,1,(0,1))
    by = -2*B[:,:,1]*D/(2*(sigma**2))
    Ix = convolve(gray,bx)#convolution kernel with the image
    Iy = convolve(gray,by)
    return Ix,Iy

def EF(gray,sigma):
    #create gaussian kernel 
    B = np.array([[[-1,-1],[-1,0],[-1,1]],
            [[ 0,-1],[ 0,0],[ 0,1]],
            [[ 1,-1],[ 1,0],[ 1,1]]])
    pi = math.pi
    D = -(B[:,:,0]**2+B[:,:,1]**2)/(2*sigma**2)
    D = np.exp(D)/(2*(sigma)**2*pi)
    #create the derivatives
    bx = -2*B[:,:,0]*D/(2*(sigma)**2)
    by = -2*B[:,:,1]*D/(2*(sigma)**2)
    '''
    Gx=np.zeros(gray.shape)
    Gy=np.zeros(gray.shape)
    for k in range(0,gray.shape[0]-3):
        for l in range(0,gray.shape[1]-3):
            sumx=np.zeros(3)
            sumy=np.zeros(3)
            for i in range(0,3):
                for j in range(0,3):
                    sumx[i]+=gray[k+i,l+j]*bx[0+i,0+j]
                    sumy[i]+=gray[k+i,l+j]*by[0+i,0+j]                        
            Gx[k,l]=sum(sumx)
            Gy[k,l]=sum(sumy)
    
    Ix=np.fabs(gray*Gx)
    Iy=np.fabs(gray*Gy)
    '''
    Ix = np.fabs(convolve(gray,bx))#convolution kernel with the image
    Iy = np.fabs(convolve(gray,by))
    return np.add(Ix,Iy)

def Prewitt(gray,ksize):
    #create Prewitt kernel 3x3
    '''
    Dx=np.array([[-1,0,1],
                [-1,0,1],
                [-1,0,1]])
    Dy=np.rot90(Dx,3,(0,1))
    '''
    #create Prewitt kernel 5x5
    Dx = np.array([[-2,-1,0,1,2],
                 [-2,-1,0,1,2],
                 [-2,-1,0,1,2],
                 [-2,-1,0,1,2],
                 [-2,-1,0,1,2]])
    Dy = np.rot90(Dx,3,(0,1))
    Dx = Dx/ksize
    Dy = Dy/ksize
    Ix = np.fabs(convolve(gray,Dx))
    Iy = np.fabs(convolve(gray,Dy))
    
    return np.add(Ix,Iy)

def Scharr(gray,ksize):
    #create Scharr kernel 3x3
    '''
    Dx=np.array([[-1,0,1],
                [-3,0,3],
                [-1,0,1]])
    Dy=np.rot90(Dx,3,(0,1))
    '''
    #create Scharr kernel 5x5
    Dx = np.array([[-2,-1,0,1,2],
                [-2,-4,0,4,2],
                [-2,-4,0,4,2],
                [-2,-4,0,4,2],
                [-2,-1,0,1,2]])
    Dy = np.rot90(Dx,3,(0,1))
    Dx = Dx/ksize
    Dy = Dy/ksize
    Ix = np.fabs(convolve(gray,Dx))
    Iy = np.fabs(convolve(gray,Dy))
    
    return np.add(Ix,Iy)

def Hog_energy(gray,sigma):
    Ix = np.zeros(gray.shape)
    Iy = np.zeros(gray.shape)
    filters.gaussian_filter(gray, sigma, (1,0), Ix)
    filters.gaussian_filter(gray, sigma, (0,1), Iy)
    Ix = np.fabs(Ix)
    Iy = np.fabs(Iy)
    return np.add(Ix,Iy)/HOG(gray)

#compute the HOG of image and return the max value
def HOG(gray):
    normalizedGray = np.zeros((800, 800))
    cv.normalize(gray, normalizedGray, 0, 255, cv.NORM_MINMAX)
    gray = cv.resize(gray,(gray.shape[1]-gray.shape[1]%16,gray.shape[0]-gray.shape[0]%16), interpolation = cv.INTER_CUBIC)
    _, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)
    return abs(hog_image.max())

def optimal_vertical_seam(e):
    M=e[:]
    r,c=e.shape
    mask = np.ones(e.shape,dtype=np.bool)
    for i in range(1,r):
        for j in range(0,c):
            if(j == 0):
                M[i,j] = e[i,j]+min(M[i-1,j:j+2])
            else:
                M[i,j] = e[i,j]+min(M[i-1,j-1:j+2])
    backtrack(M,mask,r-1,np.argmin(M[r-1,:]))    
    return mask

def ReduceWidth(image,weight):
    global totalmask
    totalmask = np.ones((image.shape[0],image.shape[1]),dtype=np.bool)
    for i in range(weight):
        image = carving_vertical(image)
    return image

def totalmaskvertical():
    return totalmask

def show_seam_vertical(imgDisplayed):
    r,c,_ = imgDisplayed.shape
    for i in range(r):
        for j in range(c):
            if(totalmask[i][j] == False):
               imgDisplayed[i][j] = [0,0,208]
    return imgDisplayed

def carving_vertical(image):
    r,c,p = image.shape
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    energy = EF(gray,sigma)
    #energy=Prewitt(gray,ksize)
    #energy=Scharr(gray,ksize)
    #energy=Hog_energy(gray,sigma)
    
    mask = optimal_vertical_seam(energy)
    mask = np.stack([mask]*3,axis=2)
    image = image[mask].reshape((r,c-1,p))
    return image

def vertical_energy_map(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    energy = EF(gray,sigma)
    #energy=Prewitt(gray,ksize)
    #energy=Scharr(gray,ksize)
    #energy=Hog_energy(gray,sigma)

    M = energy[:]
    r,c = energy.shape
    mask = np.ones(energy.shape,dtype=np.bool)
    for i in range(1,r):
        for j in range(0,c):
            if(j == 0):
                M[i,j] = energy[i,j]+min(M[i-1,j:j+2])
            else:
                M[i,j] = energy[i,j]+min(M[i-1,j-1:j+2])
    return M

def energy_avg(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    e = EF(gray,sigma)
    return sum(sum(e))/e.size

def backtrack(M,mask,i,j):
    mask[i,j] = False
    totalmask[i,j] = False
    if(i == 0):
        return
    elif(j == 0):
        idx = np.argmin(M[i,j:j+2])
        backtrack(M,mask,i-1,idx)
    else:
        idx = j - 1 + np.argmin(M[i,j-1:j+2])
        backtrack(M,mask,i-1,idx)
        
def backTrackofT(T,i,j,seq):
        if(i == 0 and j == 0):
            return 
        if(T[i-1,j] < T[i,j-1]):
            seq.append(1)
            return backTrackofT(T,i-1,j,seq)
        else:
            seq.append(0)
            return backTrackofT(T,i,j-1,seq)

def main():
    t1 = time.clock()
    image = cv.imread('image_1.jpg',1)
    imageout = ReduceWidth(image,1)
    print("time",time.clock()-t1)
    plt.figure()
    plt.imshow(imageout)
    plt.title('Reduced_Width by 100 pixels austin.jpg \n shape '+str(imageout.shape))
    plt.figure()
    plt.imshow(show_seam_vertical(image))
    plt.title('vertical seams display \n initial shape '+str(image.shape))
    plt.show()
    gray = cv.cvtColor(imageout,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    e = EF(gray,0.25)
    print("eval==",sum(sum(e))/e.size)
    
if __name__ == "__main__":
    main()
