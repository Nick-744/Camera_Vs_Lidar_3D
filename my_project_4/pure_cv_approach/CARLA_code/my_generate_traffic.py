#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

'''Simplified traffic‐generation script for CARLA 0.9.11
   – defaults to 45 vehicles and 15 pedestrians, no argparse parsing.'''

import glob
import os
import sys
import time

# --- Locate the CARLA 0.9.11 egg ----------------------------------------------
try:
    # Try the 'standard' ../carla/dist path
    egg_path = glob.glob(os.path.abspath(
        os.path.join('..', 'carla', 'dist', 'carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'
        ))
    ))[0]
    sys.path.append(egg_path)
except IndexError:
    try:
        # Fallback: CARLA_0.9.11 WindowsNoEditor (or LinuxNoEditor) install tree
        egg_path = glob.glob(os.path.abspath(
            os.path.join('..', '..',
                'CARLA_0.9.11', 'WindowsNoEditor', 'PythonAPI', 'carla', 'dist',
                'carla-*%d.%d-%s.egg' % (
                    sys.version_info.major,
                    sys.version_info.minor,
                    'win-amd64' if os.name == 'nt' else 'linux-x86_64'
                )
            )
        ))[0]
        sys.path.append(egg_path)
    except IndexError:
        # If neither path exists, continue; import will fail if CARLA isn't found.
        pass

import carla
from carla import VehicleLightState as vls

import logging
from numpy import random

def get_actor_blueprints(world, filter, generation):
    '''
    Retrieve all blueprint objects matching 'filter', then (if generation!='all')
    filter by 'generation' attribute when present.
    '''
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == 'all':
        return bps

    # If only one blueprint matches, ignore generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        filtered = []
        for bp in bps:
            if bp.has_attribute('generation') and int(bp.get_attribute('generation')) == int_generation:
                filtered.append(bp)
        if not filtered:
            print(f"   Warning! No '{filter}' blueprints with generation={int_generation} found.")
        return filtered
    except ValueError:
        print('   Warning! Actor Generation is not valid. No actor will be spawned.')
        return []

def main():
    # --- Hardcoded configuration -----------------------------------------------
    HOST               = '127.0.0.1'
    PORT               = 2000
    NUMBER_OF_VEHICLES = 45
    NUMBER_OF_WALKERS  = 15
    SAFE_MODE          = False   # If True, only spawn 'car' base_type vehicles
    FILTER_VEHICLES    = 'vehicle.*'
    GENERATION_VEH     = 'All'
    FILTER_WALKERS     = 'walker.pedestrian.*'
    GENERATION_WALK    = 'All'
    TM_PORT            = 8000
    ASYNCHRONOUS       = False
    HYBRID_MODE        = False
    RANDOM_SEED        = None    # e.g. 42 for deterministic Traffic Manager
    PEDESTRIAN_SEED    = None    # e.g. 17 for deterministic walker spawning
    CAR_LIGHTS_ON      = False
    HERO_MODE          = False   # If True, the very first spawned vehicle is 'hero'
    RESPAWN_DORMANT    = False
    NO_RENDERING       = False

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list  = []
    all_id        = []

    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    synchronous_master = False

    # Seed the RNGs
    seed_value = RANDOM_SEED if RANDOM_SEED is not None else int(time.time())
    random.seed(seed_value)

    try:
        world = client.get_world()

        # --- Shorten all traffic‐light durations -------------------------
        # Grab every traffic light in the map and override its cycle times:
        for tl in world.get_actors().filter('traffic.traffic_light*'):
            try:
                tl.set_green_time(2.5)
                tl.set_yellow_time(1.)
                tl.set_red_time(2.5)
            except Exception as e:
                # Some maps/versions may not support all setters; ignore if it fails:
                logging.debug(f"Could not set times on TL {tl.id}: {e}")

        # --- Configure Traffic Manager ---------------------------------------
        traffic_manager = client.get_trafficmanager(TM_PORT)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if RESPAWN_DORMANT:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if HYBRID_MODE:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if RANDOM_SEED is not None:
            traffic_manager.set_random_device_seed(RANDOM_SEED)

        settings = world.get_settings()
        if not ASYNCHRONOUS:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False
        else:
            print('Asynchronous mode is enabled. You may experience timing issues.')

        if NO_RENDERING:
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        # --- Vehicle blueprints -----------------------------------------------
        blueprints = get_actor_blueprints(world, FILTER_VEHICLES, GENERATION_VEH)
        if not blueprints:
            raise ValueError("Couldn't find any vehicle blueprints with those filters.")
        if SAFE_MODE:
            # Only keep 'car' base_type if SAFE_MODE is on
            blueprints = [bp for bp in blueprints if bp.get_attribute('base_type') == 'car']
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        # --- Walker blueprints -------------------------------------------------
        blueprints_walkers = get_actor_blueprints(world, FILTER_WALKERS, GENERATION_WALK)
        if not blueprints_walkers:
            raise ValueError("Couldn't find any walker blueprints with those filters.")

        # --- Determine spawn points -------------------------------------------
        spawn_points = world.get_map().get_spawn_points()
        n_spawn_points = len(spawn_points)

        # Shuffle if we have more spawn points than vehicles
        if NUMBER_OF_VEHICLES < n_spawn_points:
            random.shuffle(spawn_points)
        elif NUMBER_OF_VEHICLES > n_spawn_points:
            msg = f'Requested {NUMBER_OF_VEHICLES} vehicles but found only {n_spawn_points} spawn points.'
            logging.warning(msg)
            NUMBER_OF_VEHICLES = n_spawn_points

        # Aliases for batch commands
        SpawnActor   = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor  = carla.command.FutureActor

        # --- Spawn Vehicles ---------------------------------------------------
        batch = []
        is_hero_assigned = HERO_MODE
        for idx, transform in enumerate(spawn_points):
            if idx >= NUMBER_OF_VEHICLES:
                break

            bp = random.choice(blueprints)

            # Random color if available
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)

            # Random driver if available
            if bp.has_attribute('driver_id'):
                driver_id = random.choice(bp.get_attribute('driver_id').recommended_values)
                bp.set_attribute('driver_id', driver_id)

            # Assign role_name
            if is_hero_assigned:
                bp.set_attribute('role_name', 'hero')
                is_hero_assigned = False
            else:
                bp.set_attribute('role_name', 'autopilot')

            batch.append(
                SpawnActor(bp, transform)
                     .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
            )

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # Turn on vehicle lights if requested
        if CAR_LIGHTS_ON:
            actors = world.get_actors(vehicles_list)
            for veh in actors:
                traffic_manager.update_vehicle_lights(veh, True)

        # --- Spawn Walkers (Pedestrians) --------------------------------------
        percentagePedestriansRunning  = 0.0  # 0% will run
        percentagePedestriansCrossing = 0.0  # 0% will cross roads
        if PEDESTRIAN_SEED is not None:
            world.set_pedestrians_seed(PEDESTRIAN_SEED)
            random.seed(PEDESTRIAN_SEED)

        walker_spawn_points = []
        for _ in range(NUMBER_OF_WALKERS):
            spawn_t = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                spawn_t.location = loc
                walker_spawn_points.append(spawn_t)

        # 2) Spawn walker actors
        batch = []
        walker_speeds = []
        for w_sp in walker_spawn_points:
            walker_bp = random.choice(blueprints_walkers)

            # Make walker vulnerable
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')

            # Assign a random walking/running speed
            if walker_bp.has_attribute('speed'):
                if random.random() > percentagePedestriansRunning:
                    walker_speeds.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    walker_speeds.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                walker_speeds.append(0.0)

            batch.append(SpawnActor(walker_bp, w_sp))

        results = client.apply_batch_sync(batch, True)
        walker_speeds_filtered = []
        for i, res in enumerate(results):
            if res.error:
                logging.error(res.error)
            else:
                walkers_list.append({'id': res.actor_id})
                walker_speeds_filtered.append(walker_speeds[i])
        walker_speeds = walker_speeds_filtered

        # 3) Spawn walker controllers
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]['id']))

        results = client.apply_batch_sync(batch, True)
        for i, res in enumerate(results):
            if res.error:
                logging.error(res.error)
            else:
                walkers_list[i]['con'] = res.actor_id

        # 4) Collect all walker controller + walker IDs for cleanup
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]['con'])
            all_id.append(walkers_list[i]['id'])
        all_actors = world.get_actors(all_id)

        # Wait one tick so transforms propagate
        if ASYNCHRONOUS or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        # 5) Start walker controllers and assign random destinations
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            controller = all_actors[i]
            walker     = all_actors[i+1]
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(float(walker_speeds[int(i/2)]))

        print(f'Spawned {len(vehicles_list)} vehicles and {len(walkers_list)} walkers. Press Ctrl+C to exit.')

        # Example: globally reduce all speeds by 30%
        traffic_manager.global_percentage_speed_difference(30.0)

        # --- Main loop ---------------------------------------------------------
        while True:
            if not ASYNCHRONOUS and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

    finally:
        # Restore world settings if we were the synchronous master
        if not ASYNCHRONOUS and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode    = False
            settings.no_rendering_mode   = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        # Destroy all vehicles
        print(f'\nDestroying {len(vehicles_list)} vehicles...')
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # Stop all walker controllers
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        # Destroy all walkers (controllers + pedestrian actors)
        print(f'\nDestroying {len(walkers_list)} walkers...')
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone.')
