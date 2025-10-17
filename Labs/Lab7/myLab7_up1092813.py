from cv2 import imread, IMREAD_GRAYSCALE
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from time import time
import numpy as np

# Question-1: Implement the OpenCV matchTemplate and the minMaxLoc.
def myMatchTemplate(input_image: np.ndarray,
                    template_image: np.ndarray,
                    method: str = 'TM_SQDIFF') -> np.ndarray:
    '''
    Η δικιά μου υλοποίηση της συνάρτησης matchTemplate από την OpenCV!

    Args:
        input_image:    Η εικόνα στην οποία θα γίνει η αναζήτηση.
        template_image: Η εικόνα/τμήμα που θα αναζητηθεί.
        method:         Ο αλγόριθμος που θα χρησιμοποιηθεί για την αναζήτηση.
    
    Returns:
        result: Η εικόνα/heatmap που προκύπτει από την αναζήτηση.
    '''
    methods = [
        'TM_SQDIFF', 'TM_SQDIFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED'
    ]
    #https://docs.opencv.org/3.4/df/dfb/group__imgproc__object.html

    if method not in methods:
        raise ValueError(f'Δεν υποστηρίζεται η μέθοδος {method}!');

    (h_temp, w_temp) = template_image.shape
    (h_img, w_img)   = input_image.shape

    h_result = h_img - h_temp + 1
    w_result = w_img - w_temp + 1

    result = np.zeros((h_result, w_result), dtype = np.float32)
    for y in range(h_result):
        for x in range(w_result):
            img_crop = input_image[y:y + h_temp, x:x + w_temp]

            if method == 'TM_SQDIFF':
                result[y, x] = np.sum(
                    (img_crop - template_image) * (img_crop - template_image)
                )
            # Question-2: Try different distance functions or
            # correlation metrics on the cv2 matchTemplate.
            elif method == 'TM_SQDIFF_NORMED':
                num = np.sum(
                    (img_crop - template_image) * (img_crop - template_image)
                )
                den = np.sqrt(
                    np.sum(template_image * template_image) * np.sum(img_crop * img_crop)
                )
                result[y, x] = num / den
            elif method == 'TM_CCORR':
                result[y, x] = np.sum(img_crop * template_image)
            elif method == 'TM_CCORR_NORMED':
                num = np.sum(img_crop * template_image)
                den = np.sqrt(
                    np.sum(template_image * template_image) * \
                    np.sum(img_crop * img_crop)
                )
                result[y, x] = num / den
            
    return result;

def myMinMaxLoc(input_image: np.ndarray) -> tuple:
    '''
    Η δικιά μου υλοποίηση της συνάρτησης minMaxLoc από την OpenCV!
    
    - ΠΡΟΣΟΧΗ!
    Καλύτερη αντιστοιχία για TM_SQDIFF και TM_SQDIFF_NORMED,
    είναι η ελάχιστη τιμή και η θέση της ελάχιστης τιμής!

    Args:
        input_image: Η εικόνα στην οποία θα γίνει η αναζήτηση.
    
    Returns:
        min_val: Η ελάχιστη τιμή της εικόνας.
        max_val: Η μέγιστη τιμή της εικόνας.
        min_loc: Η θέση της ελάχιστης τιμής.
        max_loc: Η θέση της μέγιστης τιμής.
    '''
    min_val = np.min(input_image)
    max_val = np.max(input_image)

    # https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html
    min_loc = np.unravel_index(
        np.argmin(input_image), input_image.shape
    )[::-1]
    max_loc = np.unravel_index(
        np.argmax(input_image), input_image.shape
    )[::-1]
    # np.argmin/max -> index της ελάχιστης/μέγιστης τιμής στην εικόνα
    # np.unravel_index -> μετατροπή του index σε (x, y) συντεταγμένες

    return (min_val, max_val, min_loc, max_loc);

def myLocateObject(input_image: np.ndarray,
                   template_image: np.ndarray,
                   method: str = 'TM_SQDIFF',
                   block_size: int = 50) -> plt.Figure:
    '''
    Εμφάνιση της εικόνας με το τετράγωνο αναζήτησης και το heatmap της.
    '''
    print(f'--- < {method} > ---')
    res = myMatchTemplate(input_image, template_image, method)

    # --- Find the location of the best match ---
    (min_val, max_val, min_loc, max_loc) = myMinMaxLoc(res)
    print(f'Ελάχιστη τιμή: {min_val} - Θέση ελάχιστης τιμής: {min_loc}')
    print(f'Μέγιστη τιμή: {max_val} - Θέση μέγιστης τιμής: {max_loc}')

    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize = (10, 5))

    # --- Heatmap ---
    im1 = ax1.imshow(res)
    ax1.set_title('Heatmap')
    fig.colorbar(im1, ax = ax1)

    # --- Εικόνα με το τετράγωνο αναζήτησης ---
    ax2.imshow(input_image, cmap = 'gray')
    location = min_loc if method in [
        'TM_SQDIFF', 'TM_SQDIFF_NORMED'
    ] else max_loc
    rect = Rectangle(
        location,
        block_size,
        block_size,
        linewidth = 1,
        edgecolor = 'r',
        facecolor = 'none'
    )
    ax2.add_patch(rect)
    ax2.set_title('Τετράγωνο αναζήτησης')
    
    fig.suptitle(f'Αναζήτηση με {method}')
    plt.tight_layout()

    return fig;

def main():
    test_images = [
        {'filename': 'SuperMarioBros_resized.jpg', 'params': [367, 29, 50]},
        {'filename': 'left.ppm', 'params': [165, 135, 50]}
    ]

    for test_img in test_images:
        filename = test_img['filename']
        (
            template_row_start,
            template_col_start,
            block_size
        ) = test_img['params']
        
        print(f'\nΕικόνα: {filename}')
        img = imread(f'sample_data/{filename}', IMREAD_GRAYSCALE)
        img = img.astype(np.float32) # Για να είναι συμβατό με την υλοποίηση!

        template = img[template_row_start:template_row_start + block_size,
                    template_col_start:template_col_start + block_size]

        methods = [
            'TM_SQDIFF', 'TM_SQDIFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED'
        ]
        for method in methods:
            start = time()
            fig = myLocateObject(
                img,
                template,
                method,
                block_size
            )
            print(f'Χρόνος εκτέλεσης: {time() - start:.4f} sec')
            plt.show()

    return;

if __name__ == '__main__':
    main()
