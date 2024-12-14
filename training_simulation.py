import traci
import numpy as np
import random
import timeit
import os

# Traffic light phase definitions from environment.net.xml
PHASE_NS_GREEN = 0  # Action 0, Code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # Action 1, Code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # Action 2, Code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # Action 3, Code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs

    def run(self, episode, epsilon):
        """
        Executes a simulation episode and triggers training afterward.
        """
        start_time = timeit.default_timer()

        # Prepare the simulation route file and start SUMO
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # Initialize variables
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        while self._step < self._max_steps:
            # Get the current state of the intersection
            current_state = self._get_state()

            # Compute reward based on changes in cumulative waiting time
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # Store the experience in memory
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # Select the next action based on the current state
            action = self._choose_action(current_state, epsilon)

            # Transition to a yellow phase if the action changes
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # Execute the selected green phase
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # Update state, action, and wait metrics
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # Track only negative rewards
            if reward < 0:
                self._sum_neg_reward += reward

        # Save episode statistics
        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()

        simulation_time = round(timeit.default_timer() - start_time, 1)
        print("Training...")
        start_time = timeit.default_timer()

        for _ in range(self._training_epochs):
            self._replay()

        training_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time, training_time

    def _simulate(self, steps_todo):
        """
        Perform simulation steps while collecting statistics.
        """
        steps_todo = min(steps_todo, self._max_steps - self._step)
        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length

    def _collect_waiting_times(self):
        """
        Compute total waiting time for cars on incoming roads.
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self._waiting_times[car_id] = wait_time
            elif car_id in self._waiting_times:
                del self._waiting_times[car_id]
        return sum(self._waiting_times.values())

    def _choose_action(self, state, epsilon):
        """
        Select an action using an epsilon-greedy policy.
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1)
        return np.argmax(self._Model.predict_one(state))

    def _set_yellow_phase(self, old_action):
        """
        Set the yellow light phase based on the previous action.
        """
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Activate the corresponding green phase for the selected action.
        """
        green_phases = [PHASE_NS_GREEN, PHASE_NSL_GREEN, PHASE_EW_GREEN, PHASE_EWL_GREEN]
        traci.trafficlight.setPhase("TL", green_phases[action_number])

    def _get_queue_length(self):
        """
        Calculate the total number of stopped vehicles at the intersection.
        """
        halt_counts = [
            traci.edge.getLastStepHaltingNumber(road)
            for road in ["N2TL", "S2TL", "E2TL", "W2TL"]
        ]
        return sum(halt_counts)

    def _get_state(self):
        """
        Retrieve the intersection state as a binary occupancy vector.
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = 750 - traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)

            lane_cell = next(
                (i for i, dist in enumerate([7, 14, 21, 28, 40, 60, 100, 160, 400, 750]) if lane_pos < dist),
                9,
            )

            lane_mapping = {
                "W2TL_0": 0, "W2TL_1": 0, "W2TL_2": 0, "W2TL_3": 1,
                "N2TL_0": 2, "N2TL_1": 2, "N2TL_2": 2, "N2TL_3": 3,
                "E2TL_0": 4, "E2TL_1": 4, "E2TL_2": 4, "E2TL_3": 5,
                "S2TL_0": 6, "S2TL_1": 6, "S2TL_2": 6, "S2TL_3": 7,
            }
            lane_group = lane_mapping.get(lane_id, -1)

            if 0 <= lane_group <= 7:
                car_position = int(f"{lane_group}{lane_cell}")
                state[car_position] = 1

        return state

    def _replay(self):
        """
        Update the model using a batch of experiences from memory.
        """
        batch = self._Memory.get_samples(self._Model.batch_size)
        if batch:
            states = np.array([sample[0] for sample in batch])
            next_states = np.array([sample[3] for sample in batch])
            q_s_a = self._Model.predict_batch(states)
            q_s_a_d = self._Model.predict_batch(next_states)

            x, y = np.zeros((len(batch), self._num_states)), np.zeros((len(batch), self._num_actions))
            for i, (state, action, reward, next_state) in enumerate(batch):
                q_update = reward + self._gamma * np.amax(q_s_a_d[i])
                q_s_a[i][action] = q_update
                x[i], y[i] = state, q_s_a[i]

            self._Model.train_batch(x, y)

    def _save_episode_stats(self):
        """
        Record statistics for the current episode.
        """
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store
