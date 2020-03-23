from Reduce_Width import *
from Reduce_Height import *
import cv2 as cv
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import random
from matplotlib import pyplot as plt, cm
import numpy as np
class Script:
    
    def Script_1(self,image_austin,image_disney):
        image_austin_out = ReduceWidth(image_austin,100)
        seam_vertical = show_seam_vertical(copy.deepcopy(image_austin))
        image_disney_out = ReduceHeight(image_disney,100)
        seam_horizontal = show_seam_horizontal(copy.deepcopy(image_disney))
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(image_austin)
        plt.title('initial austin.jpg \n shape '+str(image_austin.shape))
        plt.subplot(1,2,2)
        plt.imshow(image_austin_out)
        plt.title('Reduced_Width by 100 pixels austin.jpg \n shape '+str(image_austin_out.shape))
        plt.figure()
        plt.imshow(seam_vertical)
        plt.title('vertical seams display')
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(image_disney)
        plt.title('initial disney.jpg \n shape '+str(image_disney.shape))
        plt.subplot(1,2,2)
        plt.imshow(image_disney_out)
        plt.title('Reduced_Height by 100 pixels disney.jpg \n shape '+str(image_disney_out.shape))
        plt.figure()
        
        plt.imshow(seam_horizontal)
        plt.title('horizontal seams display')
        plt.show()
        
    def Script_2(self,image_austin):
        gray = cv.cvtColor(image_austin,cv.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        
        energy = EF(gray,0.5)
        #energy=Prewitt(gray,50)
        #energy=Scharr(gray,68)
        #energy=Hog_energy(gray,0.5)
        
        plt.figure()
        plt.imshow(energy)
        plt.title('energy function display \n austin.jpg')
        plt.figure()
        verticalenergymap=vertical_energy_map(copy.deepcopy(image_austin))
        plt.imshow(verticalenergymap)
        plt.title('minimum energy map for vertical direction')
        plt.figure()
        horizontalenergymap=horizontal_energy_map(copy.deepcopy(image_austin))
        plt.imshow(horizontalenergymap)
        plt.title('minimum energy map for horizontal direction')

        plt.figure()
        ax = plt.subplot(111)
        im = ax.imshow(verticalenergymap)
        plt.title('colorbar for minimum vertical energy map')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.figure()
        ax = plt.subplot(111)
        im = ax.imshow(horizontalenergymap)
        plt.title('colorbar for minimum horizontal energy map')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        

        plt.figure()
        ax = plt.subplot(111)
        im = ax.imshow(energy)
        plt.title('colorbar for enegy output')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
        
    def Script_3(self,image_austin):
        
        image = ReduceWidth(copy.deepcopy(image_austin),1)
        maskvertical = totalmaskvertical()
        image = ReduceHeight(image,1)
        maskhorizontal = totalmaskhorizontal()
              
        r,c,_ = image_austin.shape
        for i in range(r):
            for j in range(c-1):
                if(maskvertical[i][j] == False or maskhorizontal[i][j] == False):
                   image_austin[i][j] = [208,0,0]
        plt.figure()
        plt.imshow(image_austin)
        plt.title('first horizontal and vertical seams ')
        plt.show()
        
    def Script_4(self,name,rows,columns):
        image = cv.imread(name,1)
        plt.figure()
        image_out = ReduceWidth(copy.deepcopy(image),columns)
        image_out = ReduceHeight(image_out,rows)
        plt.imshow(image)
        plt.title("original "+name+" with size\n"+str(image.shape))
        plt.xlabel('energy average '+str(energy_avg(image)))
        plt.figure()
        plt.imshow(image_out)
        plt.title("content-aware resized "+name+" with output size\n"+str(image_out.shape))
        plt.xlabel('energy average '+str(energy_avg(image_out)))
        res = cv.resize(image,(image.shape[1]-columns,image.shape[0]-rows), interpolation = cv.INTER_CUBIC)
        plt.figure()
        plt.imshow(res)
        plt.title("no-content resized "+name+" with output size\n"+str(res.shape))
        plt.xlabel('energy average '+str(energy_avg(res)))
        plt.show()
        
    def Script_5(self,name,rows,columns):
        image = cv.imread(name,1)
        image_out = copy.deepcopy(image)
        plt.figure()
        seq=[]
        counter_rows = 0
        counter_cols = 0
        for i in range(rows+columns):
            r = random.randint(0,1)
            if(counter_cols >= columns):
                r = 1
            elif(counter_rows >= rows):
                r = 0
            seq.append(r)
            if(r == 1):
                image_out = ReduceHeight(image_out,1)
                counter_rows += 1
            else:
                image_out = ReduceWidth(image_out,1)
                counter_cols += 1
        plt.imshow(image)
        plt.title("original "+name+" with size\n"+str(image.shape))
        plt.xlabel('energy average '+str(energy_avg(image)))
        plt.figure()
        plt.imshow(image_out)
        plt.title("random carving steps content-aware \n resized "+name+" with output size\n"+str(image_out.shape))
        plt.xlabel('energy average '+str(energy_avg(image_out)))
        print('sequence of removals',seq)
        plt.show()

    
        
    def Script_6(self,rows,columns):
        image =  cv.imread('forest.jpg',1)
        image_out = copy.deepcopy(image)
        name =  'forest.jpg'
        n,m,_ = image.shape
        Mx = horizontal_energy_map(image)
        My = vertical_energy_map(image)
        T = np.zeros((rows+1,columns+1))
        for i in range(1,rows+1):
            for j in range(1,columns+1):
                T[i,j] = min(T[i-1,j]+Mx[n-i-1,m-j],T[i,j-1]+My[n-i,m-j-1])
        seq = []
        backTrackofT(T,rows,columns,seq)
        for r in seq:
            if(r == 1):
                image_out = ReduceHeight(image_out,1)
            else:
                image_out = ReduceWidth(image_out,1)
        plt.imshow(image)
        plt.title("original "+name+" with size\n"+str(image.shape))
        plt.xlabel('energy average '+str(energy_avg(image)))
        plt.figure()
        plt.imshow(image_out)
        plt.title("optimal carving steps content-aware \n resized "+name+" with output size\n"+str(image_out.shape))
        plt.xlabel('energy average '+str(energy_avg(image_out)))
        print('sequence of removals',seq)
        plt.show()
        
    
        
        
def main():
    s=Script()

    in_put = str(input("Press a to run partA or b for running partB\n"))
    if(in_put == 'a'):
        image_austin = cv.imread('austin.jpg',1)
        image_disney = cv.imread('disney.jpg',1)
        s.Script_1(copy.deepcopy(image_austin),copy.deepcopy(image_disney))
        s.Script_2(copy.deepcopy(image_austin))
        s.Script_2(copy.deepcopy(image_disney))
        s.Script_3(image_austin)
        s.Script_3(image_disney)
    elif(in_put == 'b'):
        images_test = [['beach.jpg',94,159],['forest.jpg',251,330],['NewYork.jpr',135,204],
        ['Lefkada.jpg',358,284]]
        for test in images_test:
            s.Script_4(test[0],test[1],test[2])
            s.Script_5(test[0],test[1],test[2])

if __name__ == "__main__":
    main()
