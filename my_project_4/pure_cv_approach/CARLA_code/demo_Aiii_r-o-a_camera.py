# demo_Aiii_r-o-a_camera.py
# r-o-a: Road, Obstacles and Arrow detection using only stereo camera setup!

import os
import cv2
import sys
import glob
import numpy as np
from datetime import datetime

# --- Imports ---
from carla_helpers import (
    get_kitti_calibration,
    setup_camera,
    overlay_mask,
    setup_CARLA
)

part_A_object_detection_module_path = os.path.abspath(
    os.path.join('..', 'Aii_object_detection')
)
sys.path.append(part_A_object_detection_module_path)
from Aii_obj_detection_current import (
    YOLODetector, crop_bottom_half, draw_bboxes
)

# Import για το διάνυσμα κίνησης!
parrent_path = os.path.abspath(os.path.join('..'))
sys.path.append(parrent_path)
from Aiii_arrowMove import draw_arrow_right_half

# --- CARLA egg setup
try:
    carla_egg_path = glob.glob(os.path.abspath(
        os.path.join('..', '..',
            'CARLA_0.9.11', 'WindowsNoEditor',
            'PythonAPI', 'carla', 'dist', 'carla-*%d.%d-%s.egg' % (
                sys.version_info.major, sys.version_info.minor,
                'win-amd64' if os.name == 'nt' else 'linux-x86_64'
            )
        )
    ))[0]
    sys.path.append(carla_egg_path)
except:
    print('Πρόβλημα εύρεσης του αρχείου CARLA egg.')
    sys.exit(1);
import carla

# --- Image buffer ---
latest_images = {'left': None, 'right': None}

# --- Callbacks ---
def _cam_callback(buffer_name: str) -> callable:
    ''' Constructor wrapper για δημιουργία image callback '''
    def _cb(image):
        array = np.frombuffer(image.raw_data, dtype = np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        latest_images[buffer_name] = array
    
    return _cb;

def control_what_to_show(key:       int,
                        show_road:  bool,
                        show_boxes: bool,
                        show_arrow: bool) -> tuple:
    ''' Ελέγχει τι θα εμφανιστεί στην οθόνη '''
    if key == ord('1'):
        print() # Καλύτερη εμφάνιση!
        if show_road:
            print("Απενεργοποίηση: Road Mask")
        else:
            print("Ενεργοποίηση: Road Mask")
        show_road = not show_road
    elif key == ord('2'):
        print() # Καλύτερη εμφάνιση!
        if show_boxes:
            print("Απενεργοποίηση: Bounding Boxes")
        else:
            print("Ενεργοποίηση: Bounding Boxes")
        show_boxes = not show_boxes
    elif key == ord('3'):
        print() # Καλύτερη εμφάνιση!
        if show_arrow:
            print("Απενεργοποίηση: Arrow")
        else:
            print("Ενεργοποίηση: Arrow")
        show_arrow = not show_arrow

    return (show_road, show_boxes, show_arrow);

def main():
    (world, original_settings) = setup_CARLA()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp  = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle     = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)

    # --- Stereo cameras
    (WIDTH, HEIGHT, FOV) = (600, 600, 90)
    # https://www.cvlibs.net/datasets/kitti/setup.php
    camera_2 = setup_camera(
        world, blueprint_library, vehicle,
        WIDTH, HEIGHT, FOV
    )
    baseline = 0.54 # Η απόσταση μεταξύ των 2 καμερών!
    camera_3 = setup_camera(
        world,
        blueprint_library,
        vehicle,
        WIDTH, HEIGHT, FOV,
        y_arg = baseline
    )

    camera_2.listen(_cam_callback('left'))
    camera_3.listen(_cam_callback('right'))
    
    (calib, _, _) = get_kitti_calibration(
        WIDTH = WIDTH, HEIGHT = HEIGHT, FOV = FOV,
        baseline = baseline,
    )

    yolo_detector = YOLODetector(source = 'pip')
    show_road  = True
    show_boxes = True
    show_arrow = True

    print('Το setup ολοκληρώθηκε!')

    dt0 = datetime.now()
    try:
        while True:
            world.tick()

            if latest_images['left'] is None or \
                latest_images['right'] is None:
                continue;

            # .copy() -> Assignment destination is read-only
            left_color = latest_images['left'].copy()
            left_gray  = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(
                latest_images['right'], cv2.COLOR_BGR2GRAY
            )

            left_gray_cropped  = crop_bottom_half(left_gray)
            right_gray_cropped = crop_bottom_half(right_gray)

            (boxes, _, road_mask_cleaned) = yolo_detector.detect(
                left_color,
                left_gray_cropped,
                right_gray_cropped,
                calib,
                numDisparities = 80
            )

            # --- Ζωγραφικηηηή ---
            if show_road:
                if np.sum(road_mask_cleaned) != 0: # Fail safe...
                    vis = overlay_mask( # Μπλε χρώμα δρόμος!
                        left_color, road_mask_cleaned, (255, 0, 0)
                    )
            else:
                vis = left_color

            if show_boxes:
                draw_bboxes(vis, boxes)
            
            if show_arrow:
                vis = draw_arrow_right_half(
                    vis,
                    road_mask_cleaned,
                    boxes,
                    line_len  = 400,
                    full_road = True,
                    rj_filter = True
                )

            cv2.imshow('Dash Camera', vis)

            # --- Έλεγχος της επεξεργασίας/εμφάνισης ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break;
            (
                show_road,
                show_boxes,
                show_arrow
            ) = control_what_to_show(
                key,
                show_road, show_boxes, show_arrow
            )

            # FPS
            dt1 = datetime.now()
            fps = 1. / (dt1 - dt0).total_seconds()
            print(f'\rFPS: {fps:.2f}', end = '')
            dt0 = dt1
            
    except KeyboardInterrupt:
        print('\nΔιακοπή')
    finally:
        world.apply_settings(original_settings)
        camera_2.destroy()
        camera_3.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print('\nΕπιτυχής εκκαθάριση!')

    return;

if __name__ == '__main__':
    main()
