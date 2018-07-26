import numpy as np
import h5py
import random

def image_cropping(img, patch_h, patch_w, center_x, center_y, N_patches):
    # Create ouput patches data
    patches = np.empty((N_patches, patch_h, patch_w, img.shape[2]))
    labels = np.empty(N_patches)
    
    img_h = img.shape[0]
    img_w = img.shape[1]
    
    for i in range(N_patches):
        x = random.randint(0 + int(patch_h/2), img_w - int(patch_w)/2)
        y = random.randint(0 + int(patch_h/2), img_h - int(patch_w)/2)
        y_min = y - int(patch_h/2)
        y_max = y + int(patch_h/2)
        
        x_min = x - int(patch_w/2)
        x_max = x + int(patch_w/2)
        patches[i] = img[y_min:y_max, x_min:x_max, :]
        # Check the patches contains the center of the stent
        has_stent = 0
        for c in range(len(center_x)):
            sx = center_x[c]
            sy = center_y[c]
            if sx >= x_min and sx <= x_max and sy >= y_min and sy <= y_max:
                has_stent = 1
                break
        labels[i] = has_stent
    
    return patches, labels


def image_test_crop(img, patch_h, patch_w, center_x, center_y, strides):
    # Create ouput patches data

    img_h = img.shape[0]
    img_w = img.shape[1]
    
    n_height = int((img_h - patch_h)/strides + 1)
    n_width = int((img_w - patch_w)/strides + 1)
    N_patches = n_height * n_width
    
    print("Number of patches:" + str(N_patches))
    
    patches = np.empty((N_patches, patch_h, patch_w, img.shape[2]))
    labels = np.zeros(N_patches)

   

    for i in range(n_height):
        for j in range(n_width):
            y_min = i
            y_max = i + patch_h

            x_min = j
            x_max = j + patch_w
            
            patches[i* n_height + j] = img[y_min:y_max, x_min:x_max, :]
            
            for c in range(len(center_x)):
                sx = center_x[c]
                sy = center_y[c]
                if sx >= x_min and sx <= x_max and sy >= y_min and sy <= y_max:
                    labels[i * n_height + j] = 1
                
    return patches, labels

def image_sliding_window(img, patch_h, patch_w, strides):
    # Create ouput patches data
    
    img_h = img.shape[0]
    img_w = img.shape[1]
    
    n_height = int((img_h - patch_h)/strides + 1)
    n_width = int((img_w - patch_w)/strides + 1)
    N_patches = n_height * n_width
    
    print("Number of patches:" + str(N_patches))
    
    patches = np.empty((N_patches, patch_h, patch_w, img.shape[2]))

    for i in range(n_height):
        for j in range(n_width):
            y_min = i
            y_max = i + patch_h

            x_min = j
            x_max = j + patch_w
            
            patches[i* n_height + j] = img[y_min:y_max, x_min:x_max, :]
                
    return patches


def recombine_test(img, patch_h, patch_w, strides, labels):
    # Create ouput patches data
        
    img_h = img.shape[0]
    img_w = img.shape[1]
    
    n_height = int((img_h - patch_h)/strides + 1)
    n_width = int((img_w - patch_w)/strides + 1)
    N_patches = n_height * n_width
    
    print("Number of patches:" + str(N_patches))
    
    recombine_img = np.zeros((img_h,img_w,3))
    overlap_count = np.zeros_like(recombine_img)
    #patches = np.empty((N_patches, patch_h, patch_w, img.shape[2]))
   
    for i in range(n_height):
        for j in range(n_width):
            y_min = i
            y_max = i + patch_h

            x_min = j
            x_max = j + patch_w
            
            recombine_img[y_min:y_max, x_min:x_max,:] += labels[i * n_height + j]
            overlap_count[y_min:y_max, x_min:x_max,:] += 1
                
    return recombine_img/overlap_count



def load_hdf5(infile):
    with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
        return f["image"][()]
    
def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)