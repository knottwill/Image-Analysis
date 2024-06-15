from skimage import measure
import numpy as np

def plot_image_mask_overlay(image, mask, ax, dim_factor=0.3, cmap=None, border_color='yellow', border_width=1):
    """
    Plot an image with a segmentation mask overlaid.
    """
    dimmed_image = image * dim_factor

    if len(image.shape) == 2:
        result_image = np.where(mask==1, image, dimmed_image)
        ax.imshow(result_image, cmap=cmap)
    else:
        ax.imshow(image)
        ax.imshow(mask, cmap=cmap, alpha=dim_factor)
    contours = measure.find_contours(mask, 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=border_width, color=border_color)