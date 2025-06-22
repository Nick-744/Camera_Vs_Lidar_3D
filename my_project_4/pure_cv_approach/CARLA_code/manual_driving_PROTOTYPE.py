# manual_driving_PROTOTYPE.py - Δεν έχει δοκιμαστεί σχεδόν καθόλου, άμα στουκάρεις,
#                               πιθανός να κρασάρει όλο!!!

# --- CARLA egg setup
import os, sys, glob
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

from datetime import datetime
import open3d as o3d
import numpy as np
import weakref
import pygame
import cv2

from pygame.locals import (
    K_w, K_s, K_a, K_d,
    K_UP, K_DOWN, K_LEFT, K_RIGHT, K_SPACE, K_q, K_ESCAPE
)

# --- Imports ---
from carla_helpers import (
    get_kitti_calibration, setup_camera, setup_CARLA
)

# Προσθήκη path για την υλοποίηση των ερωτημάτων του part B!
part_B_module_path = os.path.abspath(os.path.join('..'))
sys.path.append(part_B_module_path)
from Biii_LiDAR_arrowMove import prepare_processed_pcd

# --- Image buffer
latest_rgb       = {'frame': None}
raw_lidar_points = None

# --- Callbacks
def lidar_callback(point_cloud: carla.LidarMeasurement) -> None:
    global raw_lidar_points

    data = np.frombuffer(
        point_cloud.raw_data, dtype = np.float32
    ).reshape(-1, 4)
    raw_lidar_points = data[:, :3]

    return;

def camera_callback(image: carla.Image) -> None:
    array = np.frombuffer(image.raw_data, dtype = np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    latest_rgb['frame'] = array

    return;

# --- Setups
def setup_lidar_sensor(
    world:             carla.World,
    blueprint_library: carla.BlueprintLibrary,
    vehicle:           carla.Vehicle
) -> carla.Sensor:
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100.0')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '1000000')
    lidar_transform = carla.Transform(carla.Location(x = 0.2, z = 1.8))
    lidar = world.spawn_actor(
        lidar_bp, lidar_transform, attach_to = vehicle
    )
    
    return lidar;

class CameraManager:
    def __init__(self, parent_actor, gamma_correction=2.2):
        self.sensor           = None
        self.surface          = None
        self._parent          = parent_actor
        self.gamma_correction = gamma_correction
        self.recording        = False
        
        # Camera blueprint
        world          = self._parent.get_world()
        bp_library     = world.get_blueprint_library()
        self.camera_bp = bp_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', '800')
        self.camera_bp.set_attribute('image_size_y', '600')
        self.camera_bp.set_attribute('fov', '110')
        
        # Camera transform
        self.camera_transform = carla.Transform(
            carla.Location(x = -5.5, z = 2.8),
            carla.Rotation(pitch = -15)
        )
        
        weak_self   = weakref.ref(self)
        self.sensor = world.spawn_actor(
            self.camera_bp,
            self.camera_transform,
            attach_to=self._parent
        )
        self.sensor.listen(
            lambda image: CameraManager._parse_image(weak_self, image)
        )

        return;

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return;
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1] # BGR -> RGB
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        return;

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
        
        return;

    def destroy(self):
        self.sensor.destroy()

        return;

class MinimalController:
    def __init__(self, vehicle):
        self.vehicle     = vehicle
        self.control     = carla.VehicleControl()
        self.steer_cache = 0.

        # Setup joystick
        pygame.joystick.init()
        self.joystick = None

        # Wait a moment for controller detection
        pygame.time.wait(100)

        # Έχει τεσταριστεί μόνο με PS4 controller!!!
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f'OK! Connected to joystick: {self.joystick.get_name()}\n')
        else:
            print('[X] No joystick detected. Make sure your PS4 controller is connected!')

        return;

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True;
            elif event.type == pygame.KEYUP and event.key == K_ESCAPE:
                return True;
            elif event.type == pygame.JOYDEVICEADDED:
                print('-> Controller connected!')
                if self.joystick is None:
                    self.joystick = pygame.joystick.Joystick(0)
                    self.joystick.init()
            elif event.type == pygame.JOYDEVICEREMOVED:
                print('-> Controller disconnected!')
                self.joystick = None

        # Reset control values
        self.control.throttle   = 0.
        self.control.brake      = 0.
        self.control.steer      = 0.
        self.control.reverse    = False
        self.control.hand_brake = False

        # --- Gamepad input (PS4 controller)
        if self.joystick and self.joystick.get_init():
            try:
                # PS4 controller axis mapping:
                # Axis 0: Left stick X (steering)
                # Axis 1: Left stick Y 
                # Axis 2: L2 trigger (brake) - ranges from -1 to 1
                # Axis 3: Right stick X
                # Axis 4: Right stick Y
                # Axis 5: R2 trigger (throttle) - ranges from -1 to 1
                
                # Get axis values with deadzone
                def apply_deadzone(value, deadzone=0.1):
                    return value if abs(value) > deadzone else 0.0
                
                # Steering (left stick X-axis)
                axis_steer = apply_deadzone(self.joystick.get_axis(0))
                
                # Throttle (R2 trigger) - convert from -1,1 to 0,1
                axis_throttle = (self.joystick.get_axis(5) + 1) / 2
                
                # Brake (L2 trigger) - convert from -1,1 to 0,1  
                axis_brake = (self.joystick.get_axis(2) + 1) / 2
                
                # PS4 button mapping:
                # Button 0:  X (Cross)
                # Button 1:  Circle
                # Button 2:  Square  
                # Button 3:  Triangle
                # Button 4:  L1
                # Button 5:  R1
                # Button 6:  L2 (digital)
                # Button 7:  R2 (digital)
                # Button 8:  Share
                # Button 9:  Options
                # Button 10: PS button
                # Button 11: Left stick press
                # Button 12: Right stick press
                
                button_reverse   = self.joystick.get_button(0) # X button
                button_handbrake = self.joystick.get_button(1) # Circle button
                
                # Apply values with proper clamping
                self.control.steer      = float(np.clip(axis_steer,   -1., 1.))
                self.control.throttle   = float(np.clip(axis_throttle, 0., 1.))
                # Το χειριστήριό μου είναι χαλασμένο...
                # self.control.brake      = float(np.clip(axis_brake,    0., 1.))
                self.control.reverse    = bool(button_reverse)
                self.control.hand_brake = bool(button_handbrake)
            except Exception as e:
                print(f'[X] Controller error: {e}')
                self.joystick = None
        
        else:
            # --- Keyboard control fallback ---
            keys = pygame.key.get_pressed()
            
            # Throttle/Brake
            if keys[K_w] or keys[K_UP]:
                self.control.throttle = 1.
            if keys[K_s] or keys[K_DOWN]:
                self.control.brake = 1.
            
            # Steering with smooth transition
            steer_left  = keys[K_a] or keys[K_LEFT]
            steer_right = keys[K_d] or keys[K_RIGHT]
            
            if steer_left:
                self.steer_cache -= 0.03
            elif steer_right:
                self.steer_cache += 0.03
            else:
                # Gradually return to center
                if self.steer_cache > 0:
                    self.steer_cache = max(0, self.steer_cache - 0.05)
                elif self.steer_cache < 0:
                    self.steer_cache = min(0, self.steer_cache + 0.05)
            
            self.steer_cache   = max(-1., min(1., self.steer_cache))
            self.control.steer = round(self.steer_cache, 2)
            
            # Other controls
            self.control.hand_brake = keys[K_SPACE]
            self.control.reverse    = keys[K_q]

        # Apply control to vehicle
        self.vehicle.apply_control(self.control)

        return False;

def main():
    (world, original_settings) = setup_CARLA()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp        = blueprint_library.filter('model3')[0]
    spawn_point       = world.get_map().get_spawn_points()[0]
    vehicle           = world.spawn_actor(vehicle_bp, spawn_point)

    (WIDTH, HEIGHT, FOV) = (600, 600, 90)
    camera = setup_camera(
        world,
        blueprint_library,
        vehicle,
        WIDTH, HEIGHT, FOV
    )
    lidar = setup_lidar_sensor(world, blueprint_library, vehicle)

    camera.listen(camera_callback)
    lidar.listen(lidar_callback)

    (_, P2, Tr_velo_to_cam) = get_kitti_calibration(
        WIDTH = WIDTH, HEIGHT = HEIGHT, FOV = FOV,
        camera = camera, lidar = vehicle
    )

    # --- Διαμόρφωση της προβολής του pcd ---
    theta = np.radians(90)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    Rz    = np.array([ # Rotation matrix
        [cos_t, -sin_t, 0, 0],
        [sin_t,  cos_t, 0, 0],
        [0,      0,     1, 0],
        [0,      0,     0, 1]
    ])
    flip = np.array([ # Transformation matrix
        [1,  0, 0, 0],
        [0, -1, 0, 0], # Flip Y-axis
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])
    T = Rz @ flip

    # Manual control setup
    pygame.init()
    display = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('CARLA Manual Drive with Road Detection')

    # Create pygame camera manager for manual control display
    camera_manager = CameraManager(vehicle)
    controller     = MinimalController(vehicle)

    print('Το setup ολοκληρώθηκε!')

    # --- Main loop ---
    dt0 = datetime.now()
    try:
        pcd = o3d.geometry.PointCloud() # Empty pcd
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name = 'LiDAR Viewer - Part B',
            width = WIDTH, height = HEIGHT
        )
        geometry_added = False
        clock          = pygame.time.Clock()
        
        while True:
            clock.tick(60)
            
            # Handle manual control input
            if controller.parse_events():
                break;
            
            world.tick()

            if (latest_rgb['frame'] is None) or \
                (raw_lidar_points is None):
                # Render pygame camera feed
                camera_manager.render(display)
                pygame.display.flip()

                continue;

            # .copy() -> Assignment destination is read-only
            display_cv = latest_rgb['frame'].copy()

            # Ενημέρωση του pcd με τα νέα δεδομένα LiDAR
            new_pcd = prepare_processed_pcd(
                display_cv,
                raw_lidar_points,
                P2, Tr_velo_to_cam,
                max_length = 6.,
                origin     = np.array([3., 0., 0.]),
            )
            new_pcd.transform(T)
            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors

            # Αρχικοποίηση της προβολής του LiDAR
            if not geometry_added:
                vis.add_geometry(pcd)
                geometry_added = True
            else:
                vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Render pygame camera feed
            camera_manager.render(display)
            pygame.display.flip()

            cv2.imshow('Dash Camera', display_cv)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break;

            # FPS
            dt1 = datetime.now()
            fps = 1. / (dt1 - dt0).total_seconds()
            print(f'\rFPS: {fps:.2f}', end = '')
            dt0 = dt1
    except KeyboardInterrupt:
        print('\nΔιακοπή')
    finally:
        world.apply_settings(original_settings)
        lidar.destroy()
        camera.destroy()
        camera_manager.destroy()
        vehicle.destroy()
        vis.destroy_window()
        cv2.destroyAllWindows()
        pygame.quit()
        print('\nΕπιτυχής εκκαθάριση!')

    return;

if __name__ == '__main__':
    main()
