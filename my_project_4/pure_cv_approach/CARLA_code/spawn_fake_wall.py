import os
import sys
import glob
import numpy as np

from carla_helpers import setup_CARLA

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
    sys.exit(1)
import carla

def main():
    wall_actor = None
    try:
        world, original_settings = setup_CARLA(render=True)
        blueprint_library = world.get_blueprint_library()

        # Spawn the vehicle
        vehicle_bp = blueprint_library.filter('model3')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)

        # Spawn a fake wall ahead of the car
        wall_bp = blueprint_library.find('static.prop.streetbarrier')

        vehicle_location = spawn_point.location
        vehicle_yaw = spawn_point.rotation.yaw
        distance_ahead = 15

        dx = distance_ahead * np.cos(np.radians(vehicle_yaw))
        dy = distance_ahead * np.sin(np.radians(vehicle_yaw))

        wall_location = carla.Location(
            x = vehicle_location.x + dx,
            y = vehicle_location.y + dy,
            z = vehicle_location.z
        )

        wall_transform = carla.Transform(
            location = wall_location,
            rotation = carla.Rotation(pitch=0, yaw=vehicle_yaw + 180, roll=0)
        )

        wall_actor = world.spawn_actor(wall_bp, wall_transform)
        print(f"[INFO] Wall spawned at: x={wall_location.x:.1f}, y={wall_location.y:.1f}, z={wall_location.z:.1f}")

        spectator = world.get_spectator()
        print("[INFO] Simulation running. Press Ctrl+C to stop.")
        while True:
            world.tick()
            loc = spectator.get_location()
            print(f'\rSpectator position: x={loc.x:.1f}, y={loc.y:.1f}, z={loc.z:.1f}', end='')

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        if wall_actor is not None:
            wall_actor.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()
        if 'original_settings' in locals():
            world.apply_settings(original_settings)
        print("\n[INFO] Clean exit.")

if __name__ == '__main__':
    main()
