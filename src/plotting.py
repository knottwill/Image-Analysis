from skimage import measure
import numpy as np

def plot_image_mask_overlay(image, mask, ax, dim_factor=0.3, cmap=None, border_color='yellow'):
    """
    Plot an image with a segmentation mask overlaid.
    """
    dimmed_image = image * dim_factor
    result_image = np.where(mask==1, image, dimmed_image)
    ax.imshow(result_image, cmap=cmap)
    contours = measure.find_contours(mask, 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=border_color)