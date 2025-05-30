# ========================
# Carla Configuration
# ========================
import glob
import os
import sys

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
    pass;

import carla
from carla import VehicleLightState as vls

# ========================
# Imports
# ========================
import time
import random
import threading
import math
import numpy as np
from PIL import Image
import cv2

# ========================
# Configuration
# ========================
SAVE_RGB_PATH = '_out/rgb'
SAVE_MASK_PATH = '_out/masks'
SAVE_INTERVAL = 1.
CAMERA_RESOLUTION = (1280, 720)
VEHICLES_NUM = 30
WALKERS_NUM = 25
TRAFFIC_MANAGER_PORT = 8000
SEED = 23

# ========================
# Core Classes
# ========================
class DynamicWeather:
    def __init__(self, world):
        self.weather = world.get_weather()
        self._sun = Sun(self.weather.sun_azimuth_angle, self.weather.sun_altitude_angle)
        self._storm = Storm(self.weather.precipitation)

        return;

    def tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

        return;

class Sun:
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

        return;

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self._t)) - 20

        return;

class Storm:
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

        return;

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

        return;

def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum));

# ========================
# Traffic Management
# ========================
class TrafficGenerator:
    def __init__(self, client, world):
        self.client = client
        self.world = world
        self.traffic_manager = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_hybrid_physics_radius(80.0)
        self.vehicle_actors = []
        self.walker_actors = []
        self.all_id = []

        self.percentage_pedestrians_running = 0.2
        self.percentage_pedestrians_crossing = 0.5

        return;

    def _get_blueprints(self, filter_pattern, generation):
        bps = self.world.get_blueprint_library().filter(filter_pattern)

        if generation.lower() == "all":
            return bps;

        # If the filter returns only one bp, we assume that this one needed
        # and therefore, we ignore the generation
        if len(bps) == 1:
            return bps;

        try:
            int_generation = int(generation)
            # Check if generation is in available generations
            if int_generation in [1, 2, 3]:
                bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
                return bps;
            else:
                print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                return [];
        except:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return [];

    def configure_traffic_manager(self):
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.global_percentage_speed_difference(10.0)
        if SEED is not None:
            self.traffic_manager.set_random_device_seed(SEED)
        
        return;

    def spawn_vehicles(self):
        vehicle_blueprints = self._get_blueprints('vehicle.*', 'All')
        spawn_points = self.world.get_map().get_spawn_points()

        if VEHICLES_NUM > len(spawn_points):
            print(f"Warning: Requested {VEHICLES_NUM} vehicles but only {len(spawn_points)} spawn points available.")
            num_vehicles = len(spawn_points)
        else:
            num_vehicles = VEHICLES_NUM
            random.shuffle(spawn_points)

        batch = []
        hero = False  # CVC default: one hero if enabled (we skip that for now)

        for n, transform in enumerate(spawn_points[:num_vehicles]):
            bp = random.choice(vehicle_blueprints)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            if bp.has_attribute('driver_id'):
                driver_id = random.choice(bp.get_attribute('driver_id').recommended_values)
                bp.set_attribute('driver_id', driver_id)
            bp.set_attribute('role_name', 'hero' if hero else 'autopilot')
            hero = False  # Only one hero allowed, so disable after first

            batch.append(carla.command.SpawnActor(bp, transform)
                         .then(carla.command.SetAutopilot(carla.command.FutureActor, True, self.traffic_manager.get_port())))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                print(f"Vehicle spawn error: {response.error}")
            else:
                self.vehicle_actors.append(response.actor_id)

        return;

    def spawn_pedestrians(self):
        walker_blueprints = self._get_blueprints('walker.pedestrian.*', 'All')
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        if SEED is not None:
            self.world.set_pedestrians_seed(SEED)
            random.seed(SEED)

        spawn_points = []
        while len(spawn_points) < WALKERS_NUM:
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_points.append(carla.Transform(location=loc))

        walker_batch = []
        walker_speeds = []

        for transform in spawn_points:
            bp = random.choice(walker_blueprints)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')

            if bp.has_attribute('speed'):
                if random.random() > self.percentage_pedestrians_running:
                    walker_speeds.append(bp.get_attribute('speed').recommended_values[1]) # walk
                else:
                    walker_speeds.append(bp.get_attribute('speed').recommended_values[2]) # run
            else:
                walker_speeds.append(0.0)

            walker_batch.append(carla.command.SpawnActor(bp, transform))

        results = self.client.apply_batch_sync(walker_batch, True)
        valid_walkers = []
        walker_speeds2 = []

        for i, res in enumerate(results):
            if res.error:
                print(f"Walker spawn error: {res.error}")
            else:
                valid_walkers.append(res.actor_id)
                walker_speeds2.append(walker_speeds[i])
        walker_speeds = walker_speeds2

        # 3. spawn the walker controllers ------------------------------
        controller_batch = [
            carla.command.SpawnActor(controller_bp, carla.Transform(), walker_id)
            for walker_id in valid_walkers
        ]
        controller_results = self.client.apply_batch_sync(controller_batch, True)

        for i, res in enumerate(controller_results):
            if res.error:
                print(f"Controller spawn error: {res.error}")
            else:
                self.all_id.append(res.actor_id)
                self.all_id.append(valid_walkers[i])

        all_actors = self.world.get_actors(self.all_id)

        # >>> safety tick -------------------------------------------------
        # make sure every new controller has at least one transform
        self.world.tick() # use world.wait_for_tick() if you run asynchronous
        # <<< -------------------------------------------------------------

        self.world.set_pedestrians_cross_factor(self.percentage_pedestrians_crossing)

        # 4. start controllers and send them off --------------------------
        for i in range(0, len(self.all_id), 2):
            controller = all_actors[i]          # controller
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(float(walker_speeds[i // 2]))

        return;

    def cleanup(self):
        print("\nDestroying traffic actors...")

        for i in range(0, len(self.all_id), 2):
            actor = self.world.get_actor(self.all_id[i])
            if actor:
                actor.stop()

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicle_actors])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])
        return;

# ========================
# Sensor & Data Handling
# ========================
class DataCollector:
    def __init__(self, world, ego_vehicle):
        self.world = world
        self.last_save = time.time()

        self.latest_rgb = None
        self.latest_mask = None

        cam_transform = carla.Transform(carla.Location(x=1.5, z=1.8))
        blueprint_library = world.get_blueprint_library()
        
        # Camera setup
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(CAMERA_RESOLUTION[0]))
        cam_bp.set_attribute('image_size_y', str(CAMERA_RESOLUTION[1]))
        cam_bp.set_attribute('sensor_tick', '0.1') # manual trigger

        seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(CAMERA_RESOLUTION[0]))
        seg_bp.set_attribute('image_size_y', str(CAMERA_RESOLUTION[1]))
        seg_bp.set_attribute('sensor_tick', '0.1') # manual trigger

        # Spawn sensors
        self.rgb_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)
        self.seg_cam = world.spawn_actor(seg_bp, cam_transform, attach_to=ego_vehicle)
        
        self.rgb_image = None
        self.seg_image = None

        self.rgb_cam.listen(lambda image: setattr(self, 'rgb_image', image))
        self.seg_cam.listen(lambda image: setattr(self, 'seg_image', image))

        return;

    def process_segmentation(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, 2]
        color_mask = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
        color_mask[(array != 1) & (array != 24)] = [255, 0, 0] # υπόλοιπες κατηγορίες -> κόκκινο
        color_mask[(array == 1) | (array == 24)] = [0, 0, 255] # δρόμος -> μπλε
        
        return color_mask;

    def save_data(self):
        now = time.time()

        # 1 — always refresh the in‑memory preview buffers
        rgb = self.rgb_image
        seg = self.seg_image
        if (rgb is not None) and (seg is not None) and (rgb.frame == seg.frame):
            rgb_array = np.frombuffer(rgb.raw_data, dtype=np.uint8) \
                .reshape((rgb.height, rgb.width, 4))[:, :, :3]

            self.latest_rgb  = rgb_array
            self.latest_mask = self.process_segmentation(seg)

            # 2 — only touch the disk on SAVE_INTERVAL
            if now - self.last_save >= SAVE_INTERVAL:
                frame_id = rgb.frame
                threading.Thread(
                    target=Image.fromarray(rgb_array[:, :, ::-1]).save,
                    args=(f"{SAVE_RGB_PATH}/{frame_id:06d}.png",),
                    daemon=True
                ).start()
                threading.Thread(
                    target=Image.fromarray(self.latest_mask).save,
                    args=(f"{SAVE_MASK_PATH}/{frame_id:06d}.png",),
                    daemon=True
                ).start()
                
                self.last_save = now
        
        return;

    def preview(self):
        if (self.latest_rgb is not None) and (self.latest_mask is not None):
            combined = np.hstack((self.latest_rgb, self.latest_mask))
            combined = cv2.resize(combined, (1280, 360))
            cv2.imshow("RGB + Segmentation", combined)
            cv2.waitKey(1)
        
        return;

    def cleanup(self):
        self.rgb_cam.stop()
        self.seg_cam.stop()
        self.rgb_cam.destroy()
        self.seg_cam.destroy()
        cv2.destroyAllWindows()

        return;

# ========================
# Main Simulation
# ========================
def main():
    # Initialize environment
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Load world - select town
    # https://carla.readthedocs.io/en/latest/core_map/
    town_names = [
        'Town01',
        'Town02',
        'Town03',
        'Town04',
        'Town05',
        'Town06',
        'Town07',
        'Town10',
        'Town11',
        'Town12'
    ]
    print("\nAvailable towns:")
    print(", ".join(town_names))
    town = input("Enter the town you want to load (e.g., Town01): ").strip()
    if town not in town_names:
        print(f"Invalid town '{town}'. Falling back to default '{town_names[0]}'")
        town = town_names[0]
    print(f"Loading {town}...")
    world = client.load_world(town)
    
    # Set synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = True # Να σώσουμε τίποτα από performance!
    world.apply_settings(settings)

    try:
        weather = DynamicWeather(world)

        traffic = TrafficGenerator(client, world)
        traffic.configure_traffic_manager()
        traffic.spawn_vehicles()
        traffic.spawn_pedestrians()

        # All existing vehicles after you spawned traffic
        vehicles = world.get_actors().filter("vehicle.*")
        free_points = [p for p in world.get_map().get_spawn_points()
                       if not any(v.get_location().distance(p.location) < 1.0 for v in vehicles)]

        # Spawn ego vehicle
        ego_bp = random.choice(world.get_blueprint_library().filter('vehicle.tesla.model3'))
        ego    = world.spawn_actor(ego_bp, random.choice(free_points))
        ego.set_autopilot(True)
        ego_current_location = ego.get_location()
        collector = DataCollector(world, ego)

        # Main loop
        while True:
            world.tick()

            weather.tick(settings.fixed_delta_seconds)
            world.set_weather(weather.weather)

            temp_location = ego.get_location() # Φανάριαααααααα!
            if (temp_location.x != ego_current_location.x) and \
                (temp_location.y != ego_current_location.y):
                collector.save_data()
            ego_current_location = temp_location

            collector.preview()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic.cleanup()
        collector.cleanup()
        ego.destroy()

        print("Simulation cleaned up")

    return;

if __name__ == '__main__':
    os.makedirs(SAVE_RGB_PATH, exist_ok=True)
    os.makedirs(SAVE_MASK_PATH, exist_ok=True)
    main()
