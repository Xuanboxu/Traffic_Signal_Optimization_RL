import traci
import numpy as np
import random
import timeit
import os

# Phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # Action 0: Code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # Action 1: Code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # Action 2: Code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # Action 3: Code 11
PHASE_EWL_YELLOW = 7


class TrafficSimulation:
    def __init__(self, model, traffic_gen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self.model = model
        self.traffic_gen = traffic_gen
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.num_states = num_states
        self.num_actions = num_actions

        self.step = 0
        self.total_wait_time = 0
        self.reward_episode = []
        self.queue_length_episode = []
        self.waiting_times = {}

    def run(self, episode):
        """
        Run the simulation for one episode.
        """
        start_time = timeit.default_timer()
        self.traffic_gen.generate_routefile(seed=episode)
        traci.start(self.sumo_cmd)
        print("Simulation running...")

        self.step = 0
        old_total_wait = 0
        old_action = 0

        while self.step < self.max_steps:
            current_state = self._get_state()
            current_total_wait = self._collect_waiting_times()
            self.total_wait_time += current_total_wait
            reward = old_total_wait - current_total_wait

            action = self._choose_action(current_state)

            self._set_yellow_phase(old_action)
            self._simulate(self.yellow_duration)

            self._set_green_phase(action)
            self._simulate(self.green_duration)

            old_action = action
            old_total_wait = current_total_wait

            self.reward_episode.append(reward)

        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time, self.total_wait_time

    def _simulate(self, steps_todo):
        """
        Advance the simulation by a specified number of steps.
        """
        if (self.step + steps_todo) >= self.max_steps:
            steps_todo = self.max_steps - self.step

        while steps_todo > 0:
            traci.simulationStep()
            self.step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self.queue_length_episode.append(queue_length)

    def _collect_waiting_times(self):
        """
        Collect and sum the waiting times of all vehicles in incoming roads.
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)

            if road_id in incoming_roads:
                self.waiting_times[car_id] = wait_time
            elif car_id in self.waiting_times:
                del self.waiting_times[car_id]

        return sum(self.waiting_times.values())

    def _choose_action(self, state):
        """
        Use the model to predict the best action based on the current state.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model.predict_one(state_tensor)).item()

    def _set_yellow_phase(self, old_action):
        """
        Set the yellow phase for the traffic light.
        """
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action):
        """
        Set the green phase for the traffic light based on the selected action.
        """
        if action == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        """
        Retrieve the total number of vehicles halted at the intersection.
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        return halt_N + halt_S + halt_E + halt_W

    def _get_state(self):
        """
        Retrieve the current state of the intersection in terms of cell occupancy.
        """
        state = np.zeros(self.num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = 750 - traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)

            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9
            else:
                lane_cell = -1

            if "_" in lane_id:
                lane_group = {
                    "W2TL": 0, "N2TL": 2, "E2TL": 4, "S2TL": 6
                }.get(lane_id.split("_")[0], -1)

                if lane_group != -1:
                    car_position = lane_group * 10 + lane_cell
                    state[car_position] = 1

        return state

    @property
    def queue_length_episode(self):
        return self.queue_length_episode

    @property
    def reward_episode(self):
        return self.reward_episode
