from Reduce_Width import ReduceWidth,show_seam_vertical,vertical_energy_map,totalmaskvertical
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#class Reduce_Height:
    #def __init__(self,image,height):
    #    self.image=image
    #    self.height=height
def ReduceHeight(image,height):
    image=np.rot90(image,1,(0,1))
    image=ReduceWidth(image,height)
    image=np.rot90(image,3,(0,1))
    return image
def show_seam_horizontal(imgDisplayed):
    imgDisplayed=np.rot90(imgDisplayed,1,(0,1))
    imgDisplayed=show_seam_vertical(imgDisplayed)
    imgDisplayed=np.rot90(imgDisplayed,3,(0,1))
    return imgDisplayed
def horizontal_energy_map(image):
    image=np.rot90(image,1,(0,1))
    M=vertical_energy_map(image)
    M=np.rot90(M,3,(0,1))
    return M
def totalmaskhorizontal():
    mask=totalmaskvertical()
    return np.rot90(mask,1,(0,1))
def main():
    image=cv.imread('disney.jpg',1)
    imageout=ReduceHeight(image,100)
    plt.figure()
    plt.imshow(imageout)
    plt.title('Reduced_Height by 100 pixels disney.jpg \n shape '+str(imageout.shape))
    plt.figure()
    plt.imshow(show_seam_horizontal(image))
    plt.title('horizontal seams display \n initial shape '+str(image.shape))
    plt.show()
if __name__ == "__main__":
    main()
