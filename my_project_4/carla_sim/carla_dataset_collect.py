import numpy as np
import glob, time
import carla
import cv2
from PIL import Image
from random import choice
from sys import version_info, path
from os import makedirs, name
from os.path import join

import threading             # Λειτουργηκά Συστήματα!
sync_lock = threading.Lock() # Για την sync_buffer

# === Carla paths ===
path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    version_info.major,
    version_info.minor,
    'win-amd64' if name == 'nt' else 'linux-x86_64'))[0]
)

# === Output folders ===
SAVE_RGB_PATH = '_out/rgb'
SAVE_MASK_PATH = '_out/masks'
makedirs(SAVE_RGB_PATH, exist_ok = True)
makedirs(SAVE_MASK_PATH, exist_ok = True)

# === Global States ===
sync_buffer = {}  # key = frame_id, value = {'rgb': ..., 'mask': ...}
latest_rgb = None # Το αποθηκεύουμε για προβολή!
latest_mask = None

# === Color Mapping ===
def lane_segmentation_colormap(array):
    color_mask = np.zeros((array.shape[0], array.shape[1], 3), dtype = np.uint8)

    color_mask[(array != 1) & (array != 24)] = [255, 0, 0] # Red -> other
    color_mask[(array == 1) | (array == 24)] = [0, 0, 255] # Blue -> road

    return color_mask;

def save_rgb(image):
    global latest_rgb, sync_buffer

    frame_id = image.frame
    array = np.frombuffer(image.raw_data, dtype = np.uint8)
    array = array.reshape((image.height, image.width, 4))
    bgr_image = array[:, :, :3]
    latest_rgb = bgr_image.copy()

    with sync_lock:
        if frame_id not in sync_buffer:
            sync_buffer[frame_id] = {}
        sync_buffer[frame_id]['rgb'] = bgr_image

        check_and_save(frame_id)

    return;

def save_segmentation(image):
    global latest_mask, sync_buffer

    frame_id = image.frame
    array = np.frombuffer(image.raw_data, dtype = np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, 2]
    color_mask = lane_segmentation_colormap(array)
    latest_mask = color_mask.copy()

    with sync_lock:
        if frame_id not in sync_buffer:
            sync_buffer[frame_id] = {}
        sync_buffer[frame_id]['mask'] = color_mask

        check_and_save(frame_id)

    return;

def check_and_save(frame_id):
    with sync_lock:
        if 'rgb' in sync_buffer[frame_id] and 'mask' in sync_buffer[frame_id]:
            rgb = sync_buffer[frame_id]['rgb'][:, :, ::-1] # BGR -> RGB, αλλιώς έχουνε μπλε hue
            mask = sync_buffer[frame_id]['mask']

            Image.fromarray(rgb).save(join(SAVE_RGB_PATH, f"{frame_id:06d}.png"))
            Image.fromarray(mask).save(join(SAVE_MASK_PATH, f"{frame_id:06d}.png"))

            print(f"Saved frame {frame_id}")
            del sync_buffer[frame_id]
    
    return;

SAVE_INTERVAL = 2. # Αποθήκευση frame κάθε 2 δευτερόλεπτα
last_save_time = 0

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Συνχρονισμένη λειτουργία, ώστε τα mask και rgb frames ταυτίζονται!
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    ego_bp = choice(blueprint_library.filter('vehicle.tesla.model3'))
    spawn = choice(world.get_map().get_spawn_points())
    ego = world.spawn_actor(ego_bp, spawn)
    ego.set_autopilot(True)

    cam_transform = carla.Transform(carla.Location(x = 1.5, z = 1.8))
    (IMAGE_WIDTH, IMAGE_HEIGHT) = ('1280', '720')

    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', IMAGE_WIDTH)
    rgb_bp.set_attribute('image_size_y', IMAGE_HEIGHT)
    rgb_bp.set_attribute('fov', '90')

    seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    seg_bp.set_attribute('image_size_x', IMAGE_WIDTH)
    seg_bp.set_attribute('image_size_y', IMAGE_HEIGHT)
    seg_bp.set_attribute('fov', '90')

    rgb_cam = world.spawn_actor(rgb_bp, cam_transform, attach_to = ego)
    seg_cam = world.spawn_actor(seg_bp, cam_transform, attach_to = ego)
    rgb_cam.listen(save_rgb)
    seg_cam.listen(save_segmentation)

    print(f"Καταγραφή RGB & Segmentation frames... (Press Ctrl+C to stop)")

    try:
        last_save_time = time.time()
        while True:
            world.tick()
            now = time.time()

            if (latest_rgb is not None) and (latest_mask is not None):
                combined = np.hstack((latest_rgb, latest_mask))
                cv2.imshow("RGB & Segmentation", combined)
                cv2.waitKey(1)

            if now - last_save_time >= SAVE_INTERVAL:
                last_save_time = now
                if sync_buffer:
                    frame_id = sorted(sync_buffer.keys())[0]
                    check_and_save(frame_id)
            
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass;
    finally:
        cv2.destroyAllWindows()
        rgb_cam.stop()
        seg_cam.stop()
        rgb_cam.destroy()
        seg_cam.destroy()
        ego.destroy()

        # Reset world settings
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        
        print('Τα δεδομένα αποθηκεύτηκαν στις διαδρομές:')
        print(f"RGB: {SAVE_RGB_PATH}")
        print(f"Segmentation: {SAVE_MASK_PATH}")
        print(f"Συνολικός αριθμός frames: {len(sync_buffer)}\n")
        print("-- Τερματισμός Προγράμματος --")

    return;

if __name__ == '__main__':
    main()
