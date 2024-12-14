from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from utils import import_train_configuration, set_sumo, set_train_path

if __name__ == "__main__":
    # Load training configuration
    config = import_train_configuration(config_file='training_settings.ini')
    # Configure SUMO command
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    # Set up training output directory
    path = set_train_path(config['models_path_name'])

    # Initialize components
    model = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )
    memory = Memory(config['memory_size_max'], config['memory_size_min'])
    traffic_gen = TrafficGenerator(config['max_steps'], config['n_cars_generated'])
    simulation = Simulation(
        model,
        memory,
        traffic_gen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs']
    )

    # Training loop
    episode = 0
    start_time = datetime.datetime.now()
    while episode < config['total_episodes']:
        print(f"\n----- Episode {episode + 1} of {config['total_episodes']}")
        epsilon = 1.0 - (episode / config['total_episodes'])  # Epsilon-greedy policy
        sim_time, train_time = simulation.run(episode, epsilon)  # Run simulation and training
        print(f"Simulation time: {sim_time}s - Training time: {train_time}s - Total: {round(sim_time + train_time, 1)}s")
        episode += 1

    # Finalize training
    print("\n----- Start time:", start_time)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)
    model.save_model(path)

    # Save training configuration
    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))