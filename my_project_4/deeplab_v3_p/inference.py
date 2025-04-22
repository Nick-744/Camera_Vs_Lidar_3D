# Για να μην έχουμε σφάλμα με το OpenMP!!! Χρειάζεται μόνο στο αρχείο που εκτελείται 1ο!
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

import torch
import cv2
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath
import numpy as np

def paint(image, mask_result): # Ζωγραφίζουμεεε!
    # (fig, ax) = plt.subplots(1, 2)
    # ax[0].imshow(image)
    # ax[0].set_title("Original")
    # ax[1].imshow(mask_result, cmap = "jet", alpha = 0.7)
    # ax[1].set_title('Μάσκες')
    # plt.show()

    color_map = {
        0: [255, 0, 0], # Κόκκινο
        1: [0, 0, 255], # Μπλε
        2: [0, 255, 0], # Πρασίνο
    }

    # Δημιουργία RGB μάσκας
    color_mask = np.zeros_like(image)
    for (k, v) in color_map.items():
        color_mask[mask_result == k] = v # NumPy magic!

    overlay = cv2.addWeighted(image, 0.7, color_mask, 0.5, 0)

    plt.imshow(overlay)
    plt.title('Χαρτογράφηση των κατηγοριών')
    plt.axis("off")
    plt.show()

    return;

from transformer import transform
import my_model

def main():
    device = torch.device('cpu')
    base_path = dirname(abspath(__file__))

    # Φορτώνουμε το μοντέλο που εκπαιδεύσαμε με το train.py!
    model = my_model.get_model(device)
    model.load_state_dict(torch.load('my_road_model.pth', map_location = 'cpu', weights_only = True))
    model.eval()

    image_path = abspath(join(base_path, '..', 'KITTI', 'data_road', 'testing', 'image_2')) # Για test, υπάρχει από:
                                                                                            # um_000000.png  -> um_000095.png
                                                                                            # umm_000000.png -> umm_000093.png
                                                                                            # uu_000000.png  -> uu_000099.png
    for i in range(10, 15):
        image = cv2.cvtColor(cv2.imread(join(image_path, f'um_0000{i}.png')), cv2.COLOR_BGR2RGB)
        img = cv2.resize(image, my_model.model_input_size, interpolation = cv2.INTER_LINEAR)

        input_tensor = transform(img).unsqueeze(0)
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            mask_result = model(input_tensor)['out'][0].argmax(0).numpy()
            mask_resized = cv2.resize(mask_result, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_NEAREST)

        paint(image, mask_resized)

    return;

if __name__ == '__main__':
    main()
