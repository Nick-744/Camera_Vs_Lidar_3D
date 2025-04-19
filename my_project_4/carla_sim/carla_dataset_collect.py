import glob, os, sys, time, random, queue, threading
import numpy as np
import cv2
from PIL import Image

# === Carla path ===
try:
    carla_egg_glob = glob.glob(
        os.path.join('..', '..', '..', 'WindowsNoEditor', 'PythonAPI', 'carla', 'dist', 
            'carla-*%d.%d-%s.egg' % (
                sys.version_info.major,
                sys.version_info.minor,
                'win-amd64' if os.name == 'nt' else 'linux-x86_64'
            )
        )
    )
    if carla_egg_glob:
        sys.path.append(carla_egg_glob[0])
except IndexError:
    pass

import carla

# === Output Folders ===
SAVE_RGB_PATH = '_out/rgb'
SAVE_MASK_PATH = '_out/masks'
os.makedirs(SAVE_RGB_PATH, exist_ok=True)
os.makedirs(SAVE_MASK_PATH, exist_ok=True)

# === Global State ===
latest_rgb = None
latest_mask = None
last_save_time = 0
SAVE_INTERVAL = 2.0  # seconds

# === Color Mapping ===
def lane_segmentation_colormap(array):
    color_mask = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
    color_mask[(array != 1) & (array != 24)] = [255, 0, 0]   # Red → Other
    color_mask[(array == 1) | (array == 24)] = [0, 0, 255]   # Blue → Road
    return color_mask

# === Async Save ===
def async_save(path, array):
    threading.Thread(target=lambda: Image.fromarray(array).save(path), daemon=True).start()

# === Save Callbacks ===
def save_rgb(image):
    global latest_rgb
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    latest_rgb = array.copy()

def save_segmentation(image):
    global latest_mask
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, 2]
    latest_mask = lane_segmentation_colormap(array)

def main():
    global last_save_time

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Set Synchronous Mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Spawn ego vehicle
    ego_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
    spawn = random.choice(world.get_map().get_spawn_points())
    ego = world.spawn_actor(ego_bp, spawn)
    ego.set_autopilot(True)

    cam_transform = carla.Transform(carla.Location(x=1.5, z=1.8))
    (WIDTH, HEIGHT) = ('1280', '720')

    # RGB camera
    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', WIDTH)
    rgb_bp.set_attribute('image_size_y', HEIGHT)
    rgb_bp.set_attribute('fov', '90')
    rgb_bp.set_attribute('sensor_tick', '0.0')  # manual trigger

    # Segmentation camera
    seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    seg_bp.set_attribute('image_size_x', WIDTH)
    seg_bp.set_attribute('image_size_y', HEIGHT)
    seg_bp.set_attribute('fov', '90')
    seg_bp.set_attribute('sensor_tick', '0.0')  # manual trigger

    # Spawn cameras
    rgb_cam = world.spawn_actor(rgb_bp, cam_transform, attach_to=ego)
    seg_cam = world.spawn_actor(seg_bp, cam_transform, attach_to=ego)

    rgb_queue = queue.Queue()
    seg_queue = queue.Queue()
    rgb_cam.listen(lambda image: rgb_queue.put(image))
    seg_cam.listen(lambda image: seg_queue.put(image))

    print("[INFO] Capturing synchronized frames every 2 seconds... Press Ctrl+C to stop.")

    try:
        while True:
            world.tick()
            rgb_image = rgb_queue.get()
            seg_image = seg_queue.get()

            save_rgb(rgb_image)
            save_segmentation(seg_image)

            # Display side-by-side
            if (latest_rgb is not None) and (latest_mask is not None):
                now = time.time()
                if now - last_save_time >= SAVE_INTERVAL:
                    last_save_time = now
                    frame_id = rgb_image.frame

                    rgb_path = os.path.join(SAVE_RGB_PATH, f"{frame_id:06d}.png")
                    mask_path = os.path.join(SAVE_MASK_PATH, f"{frame_id:06d}.png")

                    async_save(rgb_path, latest_rgb[:, :, ::-1])  # BGR → RGB
                    async_save(mask_path, latest_mask)
                    print(f"[✓] Saved frame {frame_id}")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rgb_cam.stop()
        seg_cam.stop()
        rgb_cam.destroy()
        seg_cam.destroy()
        ego.destroy()

        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        print("[✓] Done. Images saved in:")
        print(f"    RGB: {SAVE_RGB_PATH}")
        print(f"    Masks: {SAVE_MASK_PATH}")

if __name__ == '__main__':
    main()
