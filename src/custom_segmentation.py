"""
Custom segmentation functions for image processing.
"""

import numpy as np 

def otsu_threshold(image, n_thresholds=256):
    """ 
    Compute Otsu's threshold for a grayscale image.
    """
    
    assert len(image.shape) == 2, "Image must be grayscale"

    # create an array possible thresholds
    color_range = np.max(image) - np.min(image)
    assert color_range > 0, "Image must have multiple intensity values"
    step = color_range/n_thresholds
    thresholds = np.linspace(np.min(image)+step, np.max(image)-step, n_thresholds)
    
    image = image.flatten()
    least_variance = np.inf
    
    # iterate through each threshold and calculate the within-class variance
    for thresh in thresholds:
        
        # compute intensity variance of background and foreground pixels
        bg_pixels = image[image < thresh]
        bg_var = np.var(bg_pixels)

        fg_pixels = image[image >= thresh]
        fg_var = np.var(fg_pixels)

        # calculate within-class variance
        intra_class_variance = (len(fg_pixels)*fg_var + len(bg_pixels)*bg_var) / len(image)

        # update threshold if variance is lower
        if least_variance > intra_class_variance:
            least_variance = intra_class_variance
            least_variance_threshold = thresh
            
    return least_variance_threshold