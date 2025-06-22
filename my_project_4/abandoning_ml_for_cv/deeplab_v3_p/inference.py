# Για να μην έχουμε σφάλμα με το OpenMP!!!
# Χρειάζεται μόνο στο αρχείο που εκτελείται 1ο!
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

import cv2
import torch
import numpy as np
from time import time
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

def paint(image, mask_result): # Ζωγραφίζουμεεε!
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

    plt.figure(figsize = (14, 6))
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
    my_trained_model = abspath(join(
        base_path, 'my_road_model.pth'
    ))
    model = my_model.get_model(device)
    model.load_state_dict(torch.load(
        my_trained_model, map_location = 'cpu', weights_only = True
    ))
    model.eval()

    category    = 'um'
    images_type = 'testing'
    images_type = 'training'

    image_path = abspath(join(
        base_path, '..', '..',
        'KITTI', 'data_road', images_type, 'image_2'
    )) # Για test, υπάρχει από:
    # um_000000.png  -> um_000095.png
    # umm_000000.png -> umm_000093.png
    # uu_000000.png  -> uu_000099.png
    for i in range(0, 94):
        temp = (f'{category}_0000{i}.png' if i > 9 \
                else f'{category}_00000{i}.png')
        image = cv2.cvtColor(
            cv2.imread(join(image_path, temp)), cv2.COLOR_BGR2RGB
        )
        img = cv2.resize(
            image, my_model.model_input_size, interpolation = cv2.INTER_LINEAR
        )

        start = time()
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            mask_result = model(input_tensor)['out'][0].argmax(0).numpy()
            mask_resized = cv2.resize(
                mask_result,
                (image.shape[1], image.shape[0]),
                interpolation = cv2.INTER_NEAREST
            )
        print(f'Επεξεργασία εικόνας {temp} σε {time() - start:.2f} δευτ.')

        paint(image, mask_resized)

    return;

if __name__ == '__main__':
    main()
