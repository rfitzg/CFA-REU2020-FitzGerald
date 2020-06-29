#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


detector_arr = np.zeros((500, 500))
print(detector_arr)


# In[3]:


pix_edge_length = 15 #microns
pix_half_length = (pix_edge_length / 2)
num_of_pix = 500
num_of_pix_x = 500
num_of_pix_y = 500
electrons_per_photon = 1500 #electrons


# In[4]:


def bottom_left_of_pix():
    pix_corner_i = i * pix_edge_length
    pix_corner_j = j * pix_edge_length


# In[5]:


#i = the row value of pixel
#j = the column value of pixel
#horz_d= horizontal distance from bottom left
#vert_d= vertical distance from bottom left
def center_of_pix(pix_edge_length, i, j):
    horz_d = (i * pix_edge_length) + pix_half_length
    vert_d = (j * pix_edge_length) + pix_half_length
    #print("pixel: [" + str(i) + ", " + str(j) + "], center of pixel: [" + str(horz_d) + ", " + str(vert_d) + "] microns.")
for i in range(len(detector_arr)):
    for j in range(len(detector_arr)):
        center_of_pix(pix_edge_length, i, j)   


# In[6]:


import random
#the random.random() makes a random number between 0 & 1, so we multiply it to get the range we want
photon_sphere_x = random.random() * pix_edge_length * num_of_pix_x #microns
photon_sphere_y = random.random() * pix_edge_length * num_of_pix_y #microns
photon_sphere_z = 15 #microns
print("photon hit at [x,y] = [" + str(photon_sphere_x) + ", " + str(photon_sphere_y) + "] microns")


# In[7]:


#photon_sphere_x and photon_sphere_y is the center point of the sphere
diameter_electron_cloud = 10 #microns
radius_electron_cloud = diameter_electron_cloud / 2 #microns


# In[8]:


detector_arr_x = np.arange(num_of_pix_x) * pix_edge_length + pix_half_length
detector_arr_y = np.arange(num_of_pix_y) * pix_edge_length + pix_half_length
#remember multiple pixels have the same center value, but this array contains all of those possibilities


# In[9]:


#redefining variables as arrays to be more precise
photon_sphere_microns = [photon_sphere_x, photon_sphere_y, photon_sphere_z]
photon_sphere_pix = [photon_sphere_x / pix_edge_length, photon_sphere_y / pix_edge_length]
print("sphere position  in microns " + str(photon_sphere_microns))
print("sphere position in pixel " + str(photon_sphere_pix))


# In[10]:


closest_pix_x = (np.where(np.abs(detector_arr_x - photon_sphere_x) == np.min(np.abs(detector_arr_x - photon_sphere_x))))[0]
closest_pix_x_microns = closest_pix_x [0] * pix_edge_length + pix_half_length
closest_pix_y = (np.where(np.abs(detector_arr_y - photon_sphere_y) == np.min(np.abs(detector_arr_y - photon_sphere_y))))[0]
closest_pix_y_microns = closest_pix_y [0] * pix_edge_length + pix_half_length
closest_pix_arr = [closest_pix_x[0], closest_pix_y[0]]
closest_pix_microns = [closest_pix_x_microns, closest_pix_y_microns]
print("closest pixel to sphere: " + str(closest_pix_arr))
print("closest pixel's center in microns: " + str(closest_pix_microns))


# In[11]:


#measure the distance from closest_pix_microns to photon_sphere_microns in x and y directions
#all variables here are measured in microns
dist_sphere_pix_x = np.abs(closest_pix_x_microns - photon_sphere_x)
dist_sphere_pix_y = np.abs(closest_pix_y_microns - photon_sphere_y)
total_dist_sphere_pix = np.sqrt((dist_sphere_pix_x ** 2) + (dist_sphere_pix_y ** 2))
print("distnace in x: " + str(dist_sphere_pix_x) + " microns, " + "distance in y: " + str(dist_sphere_pix_y) + " microns")
print("total distance from center of sphere to center of pixel: " + str(total_dist_sphere_pix) + " microns")


# In[12]:


#setting up for numerical integration
subpix_edge_length = 1 #microns
subpix_half_length = subpix_edge_length / 2 #microns
num_of_subpix = 45
third_subpix = num_of_subpix / 3
two_thirds_subpix = num_of_subpix * (2/3)
subpix_arr = np.zeros ((num_of_subpix, num_of_subpix, num_of_subpix))
subpix_x_cen = np.arange(num_of_subpix) * subpix_edge_length + (detector_arr_x[closest_pix_x - 1] - pix_half_length) + subpix_edge_length
subpix_y_cen = np.arange(num_of_subpix) * subpix_edge_length + (detector_arr_y[closest_pix_y -1] - pix_half_length) + subpix_edge_length   


# In[13]:


subpix_x_cen


# 

# In[14]:


detector_arr_x[closest_pix_x]


# In[15]:


subpix_y_cen


# In[16]:


subpix_z_cen = np.arange(num_of_subpix) * subpix_edge_length


# In[17]:


subpix_z_cen = np.arange(num_of_subpix) * subpix_edge_length


# In[18]:


for i in range(num_of_subpix):
    for j in range(num_of_subpix):
        for k in range(num_of_subpix):
            dist_cen_of_photon = np.sqrt((sm_x_cen[i]-photon_sphere_x)**2 +                                                 (sm_y_cen[j]-photon_sphere_y)**2 +                                                 (sm_z_cen[k]-photon_sphere_z)**2)
            if dist_cen_of_photon < diameter_electron_cloud / 2:
                subpix_arr[i,j,k] = 1.


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[21]:


img = np.sum(sm_arr, axis=2)
imgplot = plt.imshow(img)
#figure be self explanitory
#add legends etc. 
#voxel = 3D pixel
#add dashed lines to show pixel boarder


# In[22]:


np.sum(img)


# In[23]:


#LOOP through the different pixels by varying the input
pct_pix1 = np.sum(img[0:int(sm_third_pixels), 0:int(sm_third_pixels)]) / np.sum(img)
pct_pix2 = np.sum(img[0:int(sm_third_pixels), int(sm_third_pixels):int(sm_two_thirds_pixels)]) / np.sum(img)
pct_pix3 = np.sum(img[0:int(sm_third_pixels), int(sm_two_thirds_pixels):int(sm_num_of_pixels)]) / np.sum(img)
pct_pix4 = np.sum(img[int(sm_third_pixels):int(sm_two_thirds_pixels), 0:int(sm_third_pixels)]) / np.sum(img)
pct_pix5 = np.sum(img[int(sm_third_pixels):int(sm_two_thirds_pixels), int(sm_third_pixels):int(sm_two_thirds_pixels)]) / np.sum(img)
pct_pix6 = np.sum(img[int(sm_third_pixels):int(sm_two_thirds_pixels), int(sm_two_thirds_pixels):int(sm_num_of_pixels)]) / np.sum(img)
pct_pix7 = np.sum(img[int(sm_two_thirds_pixels):int(sm_num_of_pixels), 0:int(sm_third_pixels)]) / np.sum(img)
pct_pix8 = np.sum(img[int(sm_two_thirds_pixels):int(sm_num_of_pixels), int(sm_third_pixels):int(tsm_two_thirds_pixels)]) / np.sum(img)
pct_pix9 = np.sum(img[int(sm_two_thirds_pixels):int(sm_num_of_pixels), int(sm_two_thirds_pixels):int(sm_num_of_pixels)]) / np.sum(img)


# In[24]:


#not sure how to make the above into a loop instead of typing it out the way I did 


# In[25]:


#LOOP through this print function?? if not too complex
print("% in first pixel: " + str(percent_pixel_1 * 100) + "%")
print("% in second pixel: " + str(percent_pixel_2 * 100) + "%")
print("% in third pixel: " + str(percent_pixel_3 * 100) + "%")
print("% in fourth pixel: " + str(percent_pixel_4 * 100) + "%")
print("% in fifth pixel: " + str(percent_pixel_5 * 100) + "%")
print("% in sixth pixel: " + str(percent_pixel_6 * 100) + "%")
print("% in seventh pixel: " + str(percent_pixel_7 * 100) + "%")
print("% in eigth pixel: " + str(percent_pixel_8 * 100) + "%")
print("% in ninth pixel: " + str(percent_pixel_9 * 100) + "%")


# In[26]:


detector_arr[closest_pixel_x,closest_pixel_y] += percent_pixel_1 * electrons_per_photon


# In[39]:


#LOOP?
electrons_pixel_1 = round(electrons_per_photon * percent_pixel_1)
electrons_pixel_2 = round(electrons_per_photon * percent_pixel_2)
electrons_pixel_3 = round(electrons_per_photon * percent_pixel_3)
electrons_pixel_4 = round(electrons_per_photon * percent_pixel_4)
electrons_pixel_5 = round(electrons_per_photon * percent_pixel_5)
electrons_pixel_6 = round(electrons_per_photon * percent_pixel_6)
electrons_pixel_7 = round(electrons_per_photon * percent_pixel_7)
electrons_pixel_8 = round(electrons_per_photon * percent_pixel_8)
electrons_pixel_9 = round(electrons_per_photon * percent_pixel_9)


# In[40]:


#can make if else statement for this, so you don't have to check the zeros
print('1. ' + str(detector_arr[closest_pixel_x - 1, closest_pixel_y - 1]))
print('2. ' + str(detector_arr[closest_pixel_x, closest_pixel_y - 1]))
print('3. ' + str(detector_arr[closest_pixel_x + 1, closest_pixel_y - 1]))
print('4. ' + str(detector_arr[closest_pixel_x - 1, closest_pixel_y]))
print('5. ' + str(detector_arr[closest_pixel_x, closest_pixel_y]))
print('6. ' + str(detector_arr[closest_pixel_x + 1, closest_pixel_y]))
print('7. ' + str(detector_arr[closest_pixel_x - 1, closest_pixel_y + 1]))
print('8. ' + str(detector_arr[closest_pixel_x, closest_pixel_y + 1]))
print('9. ' + str(detector_arr[closest_pixel_x + 1, closest_pixel_y + 1]))


# In[29]:


detector_arr[closest_pixel_x - 1, closest_pixel_y - 1] += electrons_pixel_1
detector_arr[closest_pixel_x, closest_pixel_y - 1] += electrons_pixel_2
detector_arr[closest_pixel_x + 1, closest_pixel_y - 1] += electrons_pixel_3
detector_arr[closest_pixel_x - 1, closest_pixel_y] += electrons_pixel_4
detector_arr[closest_pixel_x, closest_pixel_y] += electrons_pixel_5
detector_arr[closest_pixel_x + 1, closest_pixel_y] += electrons_pixel_6
detector_arr[closest_pixel_x - 1, closest_pixel_y + 1] += electrons_pixel_7
detector_arr[closest_pixel_x, closest_pixel_y + 1] += electrons_pixel_8
detector_arr[closest_pixel_x + 1, closest_pixel_y + 1] += electrons_pixel_9


# In[30]:


print('1. ' + str(detector_arr[closest_pix_x - 1, closest_pix_y - 1]))
print('2. ' + str(detector_arr[closest_pix_x, closest_pix_y - 1]))
print('3. ' + str(detector_arr[closest_pix_x + 1, closest_pix_y - 1]))
print('4. ' + str(detector_arr[closest_pix_x - 1, closest_pix_y]))
print('5. ' + str(detector_arr[closest_pix_x, closest_pix_y]))
print('6. ' + str(detector_arr[closest_pix_x + 1, closest_pix_y]))
print('7. ' + str(detector_arr[closest_pix_x - 1, closest_pix_y + 1]))
print('8. ' + str(detector_arr[closest_pix_x, closest_pix_y + 1]))
print('9. ' + str(detector_arr[closest_pix_x + 1, closest_pix_y + 1]))


# In[31]:


#!!! sometimes this "check" doesnt work: ask if theres a better numpy function to be precise
check_tot_electrons = np.sum(detector_arr)
if (check_tot_electrons == electrons_per_photon):
    print('No rounding error. total number of electrons = ' + str(electrons_per_photon))
else:
    print('***ROUNDING ERROR: total number of electrons is != ' + str(electrons_per_photon) + ". It is = " + str(check_tot_electrons))


# In[32]:


#digitize array using np.digitize
#make array values integers
#how would you find the gain if you weren't given it?
#what kind of errors are introduced?
#how many times to measure a single photon to get rid of error?
#numpy.digitize(x, bins, right=False(optional))
#numpy.histogram(a, bins=10, range=None, normed=None, weights=None, density=None)

#assume 2 electrons/DN
#gain = (# of electrons per pixel) / (# of counts per pixel)
#stdev (photons)  =  sqrt(number of photons)


# In[33]:


gain =  2 #electrons/digital(data) number
#here the data in detector_arr is still float it will become integer when digitized below
detector_arr = np.round(detector_arr * (1 / gain))


# In[ ]:





# In[ ]:





# In[34]:


#NOTES:
#could you figure out, gain? figure out gain if the # of electrons per photon is known and output is known
#How well would you know it?  

#What kind of errors are introduced by the charge sharing and digitization? 
        # we already have a rounding error before digitization, but after digitizing it can cause errors 
        # because we have less information... we only know which pixels are "on"
#How many times do you need to measure this single photon signal to beat down those errors?

#The gain value is set by the electronics that read out the CCD chip. 
#Gain is expressed in units of electrons per count. 
#For example, a gain of 1.8 e-/count means that the camera produces 1 count for every 1.8 recorded electrons.
#Of course, we cannot split electrons into fractional parts, as in the case for a gain of 1.8 e-/count. 
#What this number means is that 4/5 of the time 1 count is produced from 2 electrons, and 1/5 of the time 1 count is produced from 1 electron. 
#This number is an average conversion ratio, based on changing large numbers of electrons into large numbers of counts.
#Note: This use of the term "gain" is in the opposite sense to the way a circuit designer would use the term since, 
#in electronic design, gain is considered to be an increase in the number of output units compared with the number of input units.


# In[35]:


imgplot = plt.imshow(detector_arr[int(closest_pix_x[0]-5):int(closest_pix_x[0] + 5), int(closest_pix_y[0]) - 5:closest_pix_y[0] + 5])


# In[36]:


print(np.max(detector_arr))


# In[37]:


print(np.sum(detector_arr))


# In[38]:


if (np.sum(detector_arr) == 750):
    print('no problem')
else:
    print('error: sum of detector did not result in expected value')


# In[ ]:




