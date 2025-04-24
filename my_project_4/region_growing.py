import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(img, seed, threshold = 5):
    (height, width) = img.shape
    segmented = np.zeros((height, width), np.uint8)
    visited = np.zeros_like(segmented, dtype = bool) # Όλα false αρχικά!
    
    region_intensity = img[seed]
    region_pixels = [seed]

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
    ''' Γυρνάει την τελική μάσκα μετά το GrabCut '''

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

def main():
    base_dir = os.path.dirname(__file__)

    for i in range(5, 10):
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
        image_color = cv2.imread(img_file)
        (height, width) = image.shape

        seed_point = (height - 1, width//2) # Το μόνο σίγουρο seed...
        seg_result = region_growing(image, seed = seed_point, threshold = 12)

        temp = np.where(seg_result == 255, cv2.GC_FGD, cv2.GC_PR_BGD).astype('uint8')
        grabcut_mask = apply_grabcut(image_color, temp)

        ' https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html '
        final_mask = cv2.morphologyEx(grabcut_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        # result = cv2.bitwise_and(image_color, image_color, mask = final_mask)

        ' https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html '
        (contours, _) = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_color, contours, -1, (0, 255, 0), 3)

        # ' https://stackoverflow.com/questions/72265055/how-does-approxpolydp-and-epsilon-parameter-work '
        # contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if contours:
        #     largest_contour = max(contours, key = cv2.contourArea)

        # epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        # approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # # Draw the polygon on image
        # cv2.drawContours(image_color, [approx], -1, (255, 255, 255), 2)

        # # If it's a 4-corner shape (like a road rectangle), print corners
        # if len(approx) == 4:
        #     for idx, point in enumerate(approx):
        #         pt = tuple(point[0])
        #         cv2.circle(image_color, pt, 6, (0, 255, 255), -1)
        #         cv2.putText(image_color, f"{idx+1}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        #         print(f"Corner {idx+1}: {pt}")
        # else:
        #     print(f"Approximated shape has {len(approx)} corners.")

        # Ζωγραφικήηη
        plt.figure(figsize = (10, 6))

        plt.imshow(image_color)
        plt.title('Περίγραμμα')

        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return;

if __name__ == "__main__":
    main()
