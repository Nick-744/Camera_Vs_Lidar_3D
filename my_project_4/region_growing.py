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
    (height, width) = image.shape

    seed_points = [
        (height - 1, width//2) # Το μόνο σίγουρο seed...
        # (height - height//4, width//2)
    ]
    seg_result = region_growing(image, seeds = seed_points, threshold = 8)
    seg_result = cv2.dilate(seg_result, None, iterations = 3)

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

def CH_graham_scan(points: np.ndarray): # Χορηγία του Lab 2!
    if len(points) < 3:
        raise ValueError("No convex hull can be defined with less than 3 points!");
    
    def orientation(p, q, r):
        return np.cross(q - p, r - q);

    # Initial step - sorting according to angle
    p0_idx = np.argmin(points[:, 1])
    p0 = points[p0_idx]

    # Calculate the polar angle of each point with respect to p0
    angles = np.arctan2(points[:, 1] - p0[1], points[:, 0] - p0[0])

    # Sort the points by angle
    sorted_idx = np.argsort(angles)
    sorted_points = points[sorted_idx]

    stack = [0, 1] # Η λίστα πρχ δείκτες! Σίγουρα το 0 και 1, σκέψου το γιατί!
    ids = np.arange(2, sorted_points.shape[0])

    # Iterating over the rest of the points
    for id in ids:
        # Pop elements from the stack until graham condition is satisfied
        while (len(stack) > 1) and not orientation(sorted_points[stack[-2]],
                                                   sorted_points[stack[-1]],
                                                   sorted_points[id]) > 0:
            stack.pop()

        #append current point
        stack.append(id)

    return sorted_points[np.array(stack)];

def lane_separation(image, road_mask):
    ''' image -> Πρέπει να είναι γκρι εικόνα!
    road_mask -> Μάσκα με 0 & 255
    Επιστρέφει 2 σημεία που ορίζουν την ευθεία που χωρίζει τα 2 lanes! '''

    edges = cv2.Canny(image, 100, 150)
    edges = cv2.bitwise_and(edges, road_mask)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 40)

    mask_center_x = int(np.mean(np.column_stack(np.where(road_mask > 0))[:, 1]))
    best_line = min(
        lines, 
        key = lambda line: abs((np.cos(line[0][1]) * line[0][0]) - mask_center_x)
    )

    (rho, theta) = best_line[0]
    (a, b) = (np.cos(theta), np.sin(theta))
    (x0, y0) = (a*rho, b*rho)

    # Υπολογισμός των 2 σημείων που ορίζουν την ευθεία
    (x1, y1) = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    (x2, y2) = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

    return ((x1, y1), (x2, y2));

######################################################################################################
# Ανέλυσε τον κώδικα από εδώ και κάτω!
############################################################################### --- Lane splitting ---

def split_convex_hull(convex_hull, middle_line, intersections):
    '''
    Δέχεται:
    - convex_hull: λίστα σημείων [(x,y), ...]
    - middle_line: ((x1,y1), (x2,y2))
    - intersections: λίστα με 2 πλευρές [(A1, B1), (A2, B2)] που τέμνουν την ευθεία

    Επιστρέφει:
    - lane1_points: σημεία για το πρώτο lane
    - lane2_points: σημεία για το δεύτερο lane
    '''

    # Υπολογισμός των ακριβών σημείων τομής
    p_intersection1 = compute_intersection(intersections[0][0], intersections[0][1], middle_line[0], middle_line[1])
    p_intersection2 = compute_intersection(intersections[1][0], intersections[1][1], middle_line[0], middle_line[1])

    # Βρίσκουμε σε ποια σημεία του convex hull συμβαίνουν τα intersections
    idx1 = find_index_of_point(convex_hull, intersections[0][0])
    idx2 = find_index_of_point(convex_hull, intersections[1][0])

    # lane1 = από idx1 μέχρι idx2 (κυκλικά), και προσθέτουμε τα intersections
    lane1 = []
    lane1.append(p_intersection1)
    i = idx1
    while i != idx2:
        lane1.append(tuple(convex_hull[i]))
        i = (i + 1) % len(convex_hull)
    lane1.append(tuple(convex_hull[idx2]))
    lane1.append(p_intersection2)

    # lane2 = υπόλοιπο
    lane2 = []
    lane2.append(p_intersection2)
    i = idx2
    while i != idx1:
        lane2.append(tuple(convex_hull[i]))
        i = (i + 1) % len(convex_hull)
    lane2.append(tuple(convex_hull[idx1]))
    lane2.append(p_intersection1)

    return lane1, lane2;

# --- Helpers ---
def ccw(A, B, C):
    ''' Επιστρέφει True αν τα σημεία A, B, C είναι counter-clockwise. '''
    return ((C[1] - A[1]) * (B[0] - A[0])) > ((B[1] - A[1]) * (C[0] - A[0]));

def intersect(A, B, C, D):
    ''' Επιστρέφει True αν το τμήμα AB τέμνει το τμήμα CD. '''
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D));

def compute_intersection(A, B, C, D):
    ''' Βρίσκει το σημείο τομής των ευθειών AB και CD. '''
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    D = np.array(D, dtype=float)

    s = np.vstack([A, B, C, D])        
    h = np.hstack((s, np.ones((4, 1)))) # homogeneous

    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])

    x, y, z = np.cross(l1, l2)
    if z == 0:
        return None  # Παράλληλες
    return (x/z, y/z);

def draw_lanes(image_color, lane1, lane2):
    overlay = image_color.copy()

    lane1 = np.array(lane1, dtype=np.int32)
    lane2 = np.array(lane2, dtype=np.int32)

    # Fill πάνω στο overlay
    cv2.fillPoly(overlay, [lane1], color=(255, 0, 0))  # Μπλε γεμάτο
    cv2.fillPoly(overlay, [lane2], color=(0, 255, 0))  # Πράσινο γεμάτο

    alpha = 0.4  # 0.0 = τελείως διάφανο, 1.0 = τελείως αδιαφανές

    # Συνδυασμός με το αρχικό image
    cv2.addWeighted(overlay, alpha, image_color, 1 - alpha, 0, dst=image_color)

    # Προαιρετικά, κάνε και περίγραμμα για να φαίνεται καλύτερα
    cv2.polylines(image_color, [lane1], isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.polylines(image_color, [lane2], isClosed=True, color=(0, 0, 0), thickness=2)

def find_index_of_point(hull, point):
    '''
    hull: numpy array [[x,y], [x,y], ...]
    point: tuple (x,y)
    Επιστρέφει το index στο hull που αντιστοιχεί στο point
    '''
    for idx, pt in enumerate(hull):
        if np.allclose(pt, point):  # Χρησιμοποιούμε np.allclose για float σύγκριση
            return idx
    raise ValueError("Point not found in convex hull!")

def main():
    base_dir = os.path.dirname(__file__)

    for i in range(0, 10):
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
        convex_hull = CH_graham_scan(np.array(corners))
        middle_white_line = lane_separation(image, road_mask)

        intersections = []
        for i in range(len(convex_hull)):
            A = convex_hull[i]
            B = convex_hull[(i+1) % len(convex_hull)] # κυκλικά

            if intersect(A, B, middle_white_line[0], middle_white_line[1]):
                intersections.append((A, B))

        lane1, lane2 = split_convex_hull(convex_hull, middle_white_line, intersections)
        draw_lanes(image_color, lane1, lane2)

        # Ζωγραφικήηη
        plt.figure(figsize = (10, 6))

        # Πρέπει να γίνει μετατροπή σε RGB format, για να εμφανιστεί σωστά στο matplotlib!
        plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))

        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()

    return;

if __name__ == "__main__":
    main()
