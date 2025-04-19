import glob
import os
import sys
import time
import random
import queue
import threading
import math
import numpy as np
from PIL import Image
import cv2
import carla
from carla import VehicleLightState as vls

# ========================
# Configuration
# ========================
SAVE_RGB_PATH = '_out/rgb'
SAVE_MASK_PATH = '_out/masks'
SAVE_INTERVAL = 2.0
CAMERA_RESOLUTION = (1280, 720)
TOWN_MAP = 'Town01'
VEHICLES_NUM = 30
WALKERS_NUM = 10
TRAFFIC_MANAGER_PORT = 8000
SEED = 42

# ========================
# Core Classes
# ========================
class DynamicWeather:
    def __init__(self, world):
        self.weather = world.get_weather()
        self._sun = Sun(self.weather.sun_azimuth_angle, self.weather.sun_altitude_angle)
        self._storm = Storm(self.weather.precipitation)

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
        return self.weather

class Sun:
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self._t)) - 20

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

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        self.puddles = clamp(self._t - 10.0, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))

# ========================
# Traffic Management
# ========================
class TrafficGenerator:
    def __init__(self, client, world):
        self.client = client
        self.world = world
        self.traffic_manager = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
        self.vehicle_actors = []
        self.walker_actors = []
        self.all_id = []

    def _get_blueprints(self, filter_pattern, generation):
        blueprints = self.world.get_blueprint_library().filter(filter_pattern)
        if generation.lower() == "all":
            return blueprints
        return [x for x in blueprints if int(x.get_attribute('generation')) == int(generation)]

    def configure_traffic_manager(self):
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.global_percentage_speed_difference(30.0)
        if SEED is not None:
            self.traffic_manager.set_random_device_seed(SEED)

    def spawn_vehicles(self):
        vehicle_blueprints = self._get_blueprints('vehicle.*', 'All')
        spawn_points = self.world.get_map().get_spawn_points()
        
        batch = []
        for transform in random.sample(spawn_points, VEHICLES_NUM):
            bp = random.choice(vehicle_blueprints)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            batch.append(carla.command.SpawnActor(bp, transform)
                         .then(carla.command.SetAutopilot(carla.command.FutureActor, True)))

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                print(f"Vehicle spawn error: {response.error}")
            else:
                self.vehicle_actors.append(response.actor_id)

    def spawn_pedestrians(self):
        walker_blueprints = self._get_blueprints('walker.pedestrian.*', '2')
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        # Spawn walkers
        spawn_points = [carla.Transform(location=self.world.get_random_location_from_navigation()) 
                       for _ in range(WALKERS_NUM)]
        
        batch = [carla.command.SpawnActor(random.choice(walker_blueprints), sp) for sp in spawn_points]
        results = self.client.apply_batch_sync(batch, True)
        
        walker_speeds = [1.4 + random.random()*0.5 for _ in range(len(results))]  # 1.4-1.9 m/s
        
        # Spawn controllers
        controller_batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), walker_id)
                           for walker_id in [r.actor_id for r in results if not r.error]]
        
        # Configure controllers
        for response in self.client.apply_batch_sync(controller_batch):
            if response.error:
                continue
            self.all_id.append(response.actor_id)
            controller = self.world.get_actor(response.actor_id)
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(walker_speeds.pop(0))

    def shorten_traffic_lights(self):
        for light in self.world.get_actors().filter('traffic.traffic_light'):
            light.set_green_time(3.0)
            light.set_yellow_time(0.5)
            light.set_red_time(4.0)

    def cleanup(self):
        print("\nDestroying traffic actors...")
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicle_actors])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

# ========================
# Sensor & Data Handling
# ========================
class DataCollector:
    def __init__(self, world, ego_vehicle):
        self.world = world
        self.last_save = time.time()

        self.rgb_queue = queue.Queue()
        self.seg_queue = queue.Queue()
        self.latest_rgb = None
        self.latest_mask = None

        cam_transform = carla.Transform(carla.Location(x=1.5, z=1.8))
        blueprint_library = world.get_blueprint_library()
        
        # Camera setup
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(CAMERA_RESOLUTION[0]))
        cam_bp.set_attribute('image_size_y', str(CAMERA_RESOLUTION[1]))
        cam_bp.set_attribute('sensor_tick', '0.0') # manual trigger

        seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(CAMERA_RESOLUTION[0]))
        seg_bp.set_attribute('image_size_y', str(CAMERA_RESOLUTION[1]))
        seg_bp.set_attribute('sensor_tick', '0.0') # manual trigger

        # Spawn sensors
        self.rgb_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)
        self.seg_cam = world.spawn_actor(seg_bp, cam_transform, attach_to=ego_vehicle)
        
        # Register listeners
        self.rgb_cam.listen(self.rgb_queue.put)
        self.seg_cam.listen(self.seg_queue.put)

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
        if now - self.last_save >= SAVE_INTERVAL:
            if not self.rgb_queue.empty() and not self.seg_queue.empty():
                rgb_image = self.rgb_queue.get()
                seg_image = self.seg_queue.get()
                
                if rgb_image.frame == seg_image.frame:
                    rgb_array = np.frombuffer(rgb_image.raw_data, dtype=np.uint8)
                    rgb_array = rgb_array.reshape((rgb_image.height, rgb_image.width, 4))[:, :, :3]
                    self.latest_rgb = rgb_array

                    self.latest_mask = self.process_segmentation(seg_image)
                    
                    # Save asynchronously
                    frame_id = rgb_image.frame
                    threading.Thread(target=Image.fromarray(rgb_array[:, :, ::-1]).save,
                                     args=(f"{SAVE_RGB_PATH}/{frame_id:06d}.png",), daemon=True).start()
                    threading.Thread(target=Image.fromarray(self.latest_mask).save,
                                     args=(f"{SAVE_MASK_PATH}/{frame_id:06d}.png",), daemon=True).start()
                    
                    print(f"Saved frame {rgb_image.frame}")
                    self.last_save = now
        
        return;

    def preview(self):
        if self.latest_rgb is not None and self.latest_mask is not None:
            combined = np.hstack((self.latest_rgb, self.latest_mask))
            combined = cv2.resize(combined, (720, 203))
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
    world = client.load_world(TOWN_MAP)
    
    # Set synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = True # Μπας και σωθεί τίποτα στο performance!
    world.apply_settings(settings)

    try:
        # Spawn ego vehicle
        ego_bp = random.choice(world.get_blueprint_library().filter('vehicle.tesla.model3'))
        ego = world.spawn_actor(ego_bp, random.choice(world.get_map().get_spawn_points()))
        ego.set_autopilot(True)

        # Initialize systems
        weather = DynamicWeather(world)
        traffic = TrafficGenerator(client, world)
        collector = DataCollector(world, ego)

        # Configure traffic
        traffic.configure_traffic_manager()
        traffic.spawn_vehicles()
        traffic.spawn_pedestrians()
        traffic.shorten_traffic_lights()

        # Main loop
        while True:
            world.tick()
            weather.tick(0.05)
            world.set_weather(weather.weather)

            collector.save_data()
            collector.preview()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        settings.synchronous_mode = False
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
