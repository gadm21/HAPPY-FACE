""" 
This is an implementation of the paper found here: http://www.cs.cmu.edu/~htong/pdf/ICME04_tong.pdf
"""

import os
import sys
import pywt
from PIL import Image as im
import numpy as np
import cv2

def find_local_maximum(Emap, scale):

    dimx, dimy = (i / scale for i in Emap.shape)
    #print '\tdimx', dimx
    #print '\tdimy', dimx

    Emax = []
    vert = 1

    ## why 2
    dim_offset = 0

    for j in range(0, int(dimx - dim_offset)):

        horz = 1;
        Emax.append([])

        for k in range(0, int(dimy - dim_offset)):        

            max1 = np.max(Emap[vert:vert + (scale - 1), horz:horz + (scale - 1)])
            Emax[j].append(max1)
            horz = horz + scale 

        vert = vert + scale
            
    return Emax


def algorithm(image):

    ## shape == image resolution
    # image = image.resize((309,309), im.ANTIALIAS)

    # width, height = image.size
    # resize = min(width,height)

    # image = image.resize((resize,resize), im.ANTIALIAS)

    #image.show()




    x = np.asarray(image)




    # print(x)
    # size = min(x.shape[0],x.shape[1])
    # x = resizeImage(size,size,x)
    # print(x.shape)
    #print 'x.shape', x.shape 

    ## why 16 why -1
    # crop_x, crop_y = ((i / 16) * 16 - 1 for i in x.shape)
    # crop_x, crop_y = (int(i / 16) * 16 - 1 for i in x.shape)
    crop_x = (int(x.shape[0]/16)*16)
    crop_y = (int(x.shape[1]/16)*16)
    #oka change


    cropped = x[0:int(crop_x), 0:int(crop_y)]
    # print('cropped',cropped.shape)

    #print 'cropped.shape', cropped.shape

    ## Step1: Harr Discrete Wavelet Transform decomposition level 3 (masw needs more time)
    wavelet = 'haar'
    LL1, (LH1, HL1, HH1) = pywt.dwt2(cropped, wavelet)
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, wavelet)
    LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, wavelet)

    #                         -----------------
    #                         |       |       |
    #                         | A(LL) | H(LH) |
    #                         |       |       |
    # (A, (H, V, D))  <--->   -----------------
    #                         |       |       |
    #                         | V(HL) | D(HH) |
    #                         |       |       |
    #                         -----------------

    ## Step2: construct edge map in each scale
    Emap1 = np.sqrt(np.square(LH1) + np.square(HL1) + np.square(HH1))
    Emap2 = np.sqrt(np.square(LH2) + np.square(HL2) + np.square(HH2))
    Emap3 = np.sqrt(np.square(LH3) + np.square(HL3) + np.square(HH3))

    ## Step3: Partition the edge maps and find local maxima in each window
    Emax1 = find_local_maximum(Emap1, 8)
    Emax2 = find_local_maximum(Emap2, 4)
    Emax3 = find_local_maximum(Emap3, 2)

    # print(Emap1.shape,Emap2.shape,Emap3.shape)
    # print(np.asarray(Emax1).shape,np.asarray(Emax2).shape,np.asarray(Emax3).shape)


    return Emax1, Emax2, Emax3


def ruleset(Emax1, Emax2, Emax3, thresh):

    N_edge = 0 ## edge point
    N_da = 0 ## dirac astep
    N_rg = 0 ## roof gstep
    N_brg = 0 ## 

    dim_offset = 0
    dimx, dimy = len(Emax3) + dim_offset, len(Emax3) + dim_offset
    #print '\tdimx', dimx
    #print '\tdimy', dimx

    #oka code below
    dimx = len(Emax3)
    dimy = len(Emax3[0])

    EdgeMap = []

    for j in range(0, dimx - dim_offset):

        EdgeMap.append([])

        for k in range(0, dimy - dim_offset):

            ## Rule 1: (j, k) is edge point
            if (Emax1[j][k] > thresh) or (Emax2[j][k] > thresh) or (Emax3[j][k] > thresh):

                EdgeMap[j].append(1)
                N_edge = N_edge + 1
                rg = 0

                ## Rule 2: Dirac structure, Astep structure
                if (Emax1[j][k] > Emax2[j][k]) and (Emax2[j][k] > Emax3[j][k]):

                    N_da = N_da + 1

                ## Rule 3: Gstep structure Roof structure
                elif (Emax1[j][k] < Emax2[j][k]) and (Emax2[j][k] < Emax3[j][k]):

                    N_rg = N_rg + 1
                    rg = 1
                    
                ## Rule 4: Roof structure not sure if consistent with table
                elif (Emax2[j][k] > Emax1[j][k]) and (Emax2[j][k] > Emax3[j][k]):                
                #elif (Emax2[j][k] > Emax3[j][k]) and (Emax3[j][k] > Emax1[j][k]):

                    N_rg = N_rg + 1
                    rg = 1

                ## Rule 5:
                if rg and (Emax1[j][k] < thresh):
                    
                    N_brg = N_brg + 1

            ## (j, k) is non-edge point
            else:

                EdgeMap[j].append(0)

    per = N_da/float(N_edge)
    BlurExtent = N_brg/float(N_rg)

    return per, BlurExtent

