import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(img, seeds, threshold = 5):
    ''' Γυρνάει την μάσκα της περιοχής βάση των seeds που δώσαμε.
    img -> Γκρι εικόνα '''

    (height, width) = img.shape
    segmented = np.zeros((height, width), np.uint8)
    visited = np.zeros_like(segmented, dtype = bool) # Όλα false αρχικά!
    
    region_intensity = np.mean([img[point] for point in seeds])
    region_pixels = list(seeds)

    neighbors = [(-1, -1), (-1, 0), (-1, +1),
                 ( 0, -1),          ( 0, +1),
                 (+1, -1), (+1, 0), (+1, +1)]

    (count, sum_intensity) = (0, 0)
    while region_pixels:
        (x, y) = region_pixels.pop()
        if visited[x, y]:
            continue;

        visited[x, y] = True
        intensity = int(img[x, y])

        # Δυναμική ενημέρωση του μέσου όρου!
        sum_intensity += intensity
        count += 1
        region_intensity = sum_intensity / count

        if abs(intensity - region_intensity) <= threshold:
            segmented[x, y] = 255 # Άσπρο pixel

            for (dx, dy) in neighbors:
                (xn, yn) = (x + dx, y + dy)
                if (0 <= xn < height) and (0 <= yn < width) and (not visited[xn, yn]):
                    region_pixels.append((xn, yn))

    return segmented;

def apply_grabcut(image, mask):
    ''' Γυρνάει την τελική μάσκα μετά το GrabCut.
    image -> Έγχρωμη εικόνα (BGR format)
    mask  -> Το αποτέλεσμα του region growing '''

    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        None,
        bgModel,
        fgModel,
        10, # Όσα περισσότερα iterations, τόσο καλύτερα!
        cv2.GC_INIT_WITH_MASK
    )

    # Final binary mask: 1 or 3 = foreground
    grabcut_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
    
    return grabcut_mask;

def my_road_detection(image, image_color):
    ''' Η έγχρωμη εικόνα περνά by reference,
    το αποτέλεσμα περνιέται απευθείας στην image_color! '''

    (height, width) = image.shape

    seed_points = [
        (height - 1, width//2) # Το μόνο σίγουρο seed...
        # (height - height//4, width//2)
    ]
    seg_result = region_growing(image, seeds = seed_points, threshold = 8)
    seg_result = cv2.dilate(seg_result, None, iterations = 4)

    temp = np.where(seg_result == 255, cv2.GC_FGD, cv2.GC_PR_BGD).astype('uint8')
    # Εάν seg_result == 255 -> True, τότε GC_FGD, αλλιώς GC_PR_BGD (πιθανό background)
    grabcut_mask = apply_grabcut(image_color, temp)

    ' https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html '
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(grabcut_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    ' https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html '
    (contours, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key = cv2.contourArea) # Μεγαλύτερο contour σε επιφάνεια!
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness = cv2.FILLED)

    return (final_mask, largest_contour);

def find_mask_vertices(largest_contour):
    '''
    Βρίσκει τις κορυφές της περιμέτρου της μάσκας!
    Επιστρέφει λίστα με σημεία [(x, y), (x, y), ...]
    '''

    peri = cv2.arcLength(largest_contour, True) # Υπολογισμός περιμέτρου
    epsilon = 0.01 * peri # Το 0.01 δίνει συνήθως πάνω από 4 σημεία!

    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    vertices = [tuple(pt[0]) for pt in approx]

    return vertices;

def main():
    base_dir = os.path.dirname(__file__)

    for i in range(6, 10):
        temp = os.path.join(
            base_dir,
            'data_road',
            'testing',
            'image_2',
            f'um_00000{i}.png'
        )
        img_file = os.path.abspath(temp)

        # Φορτώνουμε γκρι και έγχρωμη εικόνα!
        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        image_color = cv2.imread(img_file, cv2.IMREAD_COLOR) # Σε BGR format!
        
        (road_mask, largest_contour) = my_road_detection(image, image_color)
        corners = find_mask_vertices(largest_contour)

        '''
        - Βρες το context hull των corner points
        - Βρες έναν τρόπο να το χωρίζεις σε 2 τμήματα που αντιστοιχούν στα lanes
        '''

        # # Ζωγραφικήηη
        # plt.figure(figsize = (10, 6))

        # # Πρέπει να γίνει μετατροπή σε RGB format, για να εμφανιστεί σωστά στο matplotlib!
        # plt.imshow(cv2.cvtColor(end, cv2.COLOR_BGR2RGB))

        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()
        # plt.close()

    return;

if __name__ == "__main__":
    main()
