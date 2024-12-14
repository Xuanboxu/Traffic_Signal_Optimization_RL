import torch
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generate the route file for one simulation episode.
        """
        torch.manual_seed(seed)  # Ensure reproducibility

        # Generate car timings using a Weibull distribution
        timings = torch.sort(torch.weibull(2.0, (self._n_cars_generated,))).values

        # Rescale timings to fit the range [0, max_steps]
        min_old, max_old = math.floor(timings[0].item()), math.ceil(timings[-1].item())
        min_new, max_new = 0, self._max_steps
        car_gen_steps = ((max_new - min_new) / (max_old - min_old)) * (timings - max_old) + max_new
        car_gen_steps = torch.round(car_gen_steps)  # Round to nearest integer

        # Create route XML file
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                if torch.rand(1).item() < 0.75:  # 75% chance of going straight
                    route_id = torch.randint(1, 5, (1,)).item()
                    route_map = {1: "W_E", 2: "E_W", 3: "N_S", 4: "S_N"}
                    print(f'    <vehicle id="{route_map[route_id]}_{car_counter}" type="standard_car" route="{route_map[route_id]}" depart="{step.item()}" departLane="random" departSpeed="10" />', file=routes)
                else:  # 25% chance of turning
                    route_id = torch.randint(1, 9, (1,)).item()
                    route_map = {
                        1: "W_N", 2: "W_S", 3: "N_W", 4: "N_E",
                        5: "E_N", 6: "E_S", 7: "S_W", 8: "S_E"
                    }
                    print(f'    <vehicle id="{route_map[route_id]}_{car_counter}" type="standard_car" route="{route_map[route_id]}" depart="{step.item()}" departLane="random" departSpeed="10" />', file=routes)

            print("</routes>", file=routes)
