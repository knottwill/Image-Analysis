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

def find_regions(binary_image, forground_only=True):
    """
    Identifies all homogeneous regions in a binary image of 0's and 1's using flood fill algorithm.

    Parameters:
    ----------
    binary_image : numpy.ndarray
        A binary image of 0's and 1's.
    forground_only : bool
        If True, only forground regions are labeled. If False, both forground and background regions are labeled.

    Returns:
    -------
    labelled_image : numpy.ndarray
        An integer image where each region is labeled with a unique integer.
    n_regions : int
        The number of regions identified in the image.
    """
    labelled_image = np.zeros_like(binary_image, dtype=int)
    
    def flood_fill(x, y, label):
        """ Flood fill algorithm with start point (x, y) and target label."""
        stack = [(x, y)] # stack of points to visit 
        while stack:
            cx, cy = stack.pop() # current point

            # If the current point is unlabeled and has the same value as the start point
            # then label it and add its neighbors to the stack
            if labelled_image[cx, cy] == 0 and binary_image[cx, cy] == binary_image[x, y]:
                labelled_image[cx, cy] = label
                if cx > 0: stack.append((cx-1, cy))
                if cx < binary_image.shape[0] - 1: stack.append((cx+1, cy))
                if cy > 0: stack.append((cx, cy-1))
                if cy < binary_image.shape[1] - 1: stack.append((cx, cy+1))
    
    # Loop over all pixels in the image, labeling each region
    current_label = 1
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if labelled_image[i, j] == 0: # If the pixel is not labeled
            # If the pixel is part of the forground or we are labeling background as well
                if binary_image[i, j] == 1 or not forground_only:
                    flood_fill(i, j, current_label)
                    current_label += 1
    
    n_regions = current_label - 1
    return labelled_image, n_regions