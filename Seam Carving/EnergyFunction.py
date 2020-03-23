from scipy.ndimage import filters , convolve
import multiprocessing
import time
import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt

def EF(gray,sigma):
    Ix = np.zeros(gray.shape)
    Iy = np.zeros(gray.shape)
    filters.gaussian_filter(gray, sigma, (0,1), Ix)
    filters.gaussian_filter(gray, sigma, (1,0), Iy)
    Ix = np.fabs(Ix)
    Iy = np.fabs(Iy)
    return np.add(Ix,Iy)
'''
def proc(j,M,e):
    if(j==0):
        M[i,j]=e[i,j]+min(M[i-1,j:j+2])
    else:
        M[i,j]=e[i,j]+min(M[i-1,j-1:j+2])
    '''

def optimal_vertical_seam(e):
    M = e[:]
    r,c = e.shape
    mask = np.ones(e.shape,dtype=np.bool)
    num_cores = multiprocessing.cpu_count()
    for i in range(1,r):
        #Parallel(n_jobs=num_cores)(delayed(proc)(j)
        for j in range(c):
            if(j == 0):
                M[i,j] = e[i,j] + min(M[i-1,j:j+2])
            else:
                M[i,j] = e[i,j] + min(M[i-1,j-1:j+2])
    backtrack(M,mask,r-1,np.argmin(M[r-1,:]))
    return mask

def carving_vertical(image):
    r,c,p = image.shape
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    sigma = 0.25
    energy = EF(gray,sigma)
    mask = optimal_vertical_seam(energy)
    mask = np.stack([mask]*3,axis=2)
    image = image[mask].reshape((r,c-1,p))
    return image

def backtrack(M,mask,i,j):
    mask[i,j] = False
    if(i == 0):
        return
    elif(j == 0):
        idx = np.argmin(M[i,j:j+2])
        backtrack(M,mask,i-1,idx)
    else:
        idx = j - 1 + np.argmin(M[i,j-1:j+2])
        backtrack(M,mask,i-1,idx)
        
def EF1(gray,sigma):
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
    Ix = np.fabs(convolve(gray,bx))#convolution cernel with the image
    Iy = np.fabs(convolve(gray,by))
    return np.add(Ix,Iy)

def carving_horizontal(image):
    image=np.rot90(image,1,(0,1))
    image=carving_vertical(image)
    image=np.rot90(image,3,(0,1))
    return image

def show_seam(img,mask):
    r,c,_=img.shape
    imgDisplayed=img[:]
    for i in range(r):
        for j in range(c):
            if(mask[i][j]==False):
               imgDisplayed[i][j]=[0,0,208]
    return imgDisplayed

def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index -1
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack


def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map

def main():
    t1 = time.clock()
    image = cv.imread('austin.jpg')
    img = image
    r,c,p = image.shape
    for i in range(500):
        image = carving_vertical(image)
    print("time",time.clock()-t1)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    energy = EF(gray,0.25)
    print("eval",sum(sum(energy))/energy.size)
    plt.imshow(image)
    plt.show()
    
if __name__=="__main__":
    main()
    


