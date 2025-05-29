import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")


INTERSECTION_TYPES = ['classic', 'roundabout']
TRAFFIC_CONTROL = ['no_tls', 'actuated_tls', 'stop_signs']
TRAFFIC_LOADS = ['low', 'medium', 'high']

BASE_PATHS = {
    'classic': r'Classic 4-leg Intersection\osm.sumocfg',
    'roundabout': r'Roundabout 4-leg Intersection\osm.sumocfg'
}

METRICS = ['avg_travel_time', 'throughput', 'emissions_CO2', 'fuel_consumption']

class IntersectionSimulator:
    def __init__(self, config_file, output_dir="results"):
        self.config_file = config_file
        self.output_dir = output_dir
        self.results = defaultdict(list)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def run_simulation(self, traffic_load, traffic_control, seed=42, gui=False, max_steps=3600):
        """Run a single simulation with specified parameters"""
        sumo_cmd = ["sumo"]
        if gui:
            sumo_cmd = ["sumo-gui"]
            
        sumo_cmd.extend([
            "-c", self.config_file,
            "--seed", str(seed),
            "--no-step-log", "true",
            "--duration-log.statistics", "true"
        ])
        
        # Apply traffic load adjustment 
        if traffic_load == 'low':
            sumo_cmd.extend(["--scale", "1.0"])  # 100% of default traffic
        elif traffic_load == 'medium':
            sumo_cmd.extend(["--scale", "1.5"])  # 150% of default traffic
        elif traffic_load == 'high':
            sumo_cmd.extend(["--scale", "2.0"])  # 200% of default traffic
                    
        traci.start(sumo_cmd)
        
        step = 0
        total_travel_time = 0
        vehicles_arrived = 0
        total_co2 = 0
        total_fuel = 0
        vehicle_entry_times = {}
        arrived_ids = []
        max_step = max_steps
        
        try:
            while step < max_step and traci.simulation.getMinExpectedNumber() > 0:
                # Apply traffic control method at the beginning of each step
                if step == 0:
                    self.apply_traffic_control(traffic_control)
                
                # For stop sign control, manually enforce stopping at intersections
                if traffic_control == 'stop_signs':
                    for veh_id in traci.vehicle.getIDList():
                        # Check if vehicle is approaching an intersection
                        next_links = traci.vehicle.getNextLinks(veh_id)
                        
                        # If vehicle is approaching an intersection
                        if next_links and len(next_links) > 0:
                            # Get vehicle's distance to the end of current edge
                            lane_position = traci.vehicle.getLanePosition(veh_id)
                            lane_length = traci.lane.getLength(traci.vehicle.getLaneID(veh_id))
                            distance_to_junction = lane_length - lane_position
                            
                            # If vehicle is within stopping distance of junction
                            if 0 < distance_to_junction < 10:
                                # Slow down the vehicle significantly
                                current_speed = traci.vehicle.getSpeed(veh_id)
                                if current_speed > 1.0:
                                    traci.vehicle.slowDown(veh_id, 1.0, 3.0)
                                    
                                    # After slowing, wait a bit before accelerating
                                    if not hasattr(self, 'stop_wait_times'):
                                        self.stop_wait_times = {}
                                    
                                    if veh_id not in self.stop_wait_times:
                                        self.stop_wait_times[veh_id] = step + 10  # Wait 10 steps
                                        
                traci.simulationStep()
                
                # Track vehicle entry times
                for veh_id in traci.simulation.getDepartedIDList():
                    vehicle_entry_times[veh_id] = step

                # First get the arrived vehicles
                arrived_ids = traci.simulation.getArrivedIDList()
                vehicles_arrived += len(arrived_ids)

                for veh_id in arrived_ids:
                    if veh_id in vehicle_entry_times:
                        total_travel_time += (step - vehicle_entry_times[veh_id])
                
                # Get currently active vehicles
                current_vehicles = set(traci.vehicle.getIDList())
                
                # Collect CO2 and fuel consumption for active vehicles
                for veh_id in current_vehicles:
                    total_co2 += traci.vehicle.getCO2Emission(veh_id)
                    total_fuel += traci.vehicle.getFuelConsumption(veh_id)
                
                step += 1
                
            # Calculate mean travel time from simulation data
            if vehicles_arrived > 0:
                mean_travel_time = total_travel_time / vehicles_arrived
            else:
                mean_travel_time = 0
            
            # Store results
            scenario = f"{traffic_load}_{traffic_control}"
            self.results[scenario].append({
                'avg_travel_time': mean_travel_time,
                'throughput': vehicles_arrived / step,  # vehicles per simulation step
                'emissions_CO2': total_co2 / max(vehicles_arrived, 1),  # CO2 per vehicle
                'fuel_consumption': total_fuel / max(vehicles_arrived, 1)  # fuel per vehicle
            })
            
        finally:
            traci.close()
    
    def apply_traffic_control(self, traffic_control):
        """Apply traffic control method using TraCI commands"""
        # Get all traffic light IDs
        tls_ids = traci.trafficlight.getIDList()
        
        if traffic_control == 'no_tls':
            # Set all traffic lights to "off" mode (green for all)
            for tl_id in tls_ids:
                # Set all-green phase 
                phases = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases
                all_green_idx = 0  
                
                # Find a suitable all-green phase if available
                for i, phase in enumerate(phases):
                    if 'g' in phase.state and 'r' not in phase.state:
                        all_green_idx = i
                        break
                        
                traci.trafficlight.setPhase(tl_id, all_green_idx)
                traci.trafficlight.setProgram(tl_id, "off")
        
        elif traffic_control == 'actuated_tls':
            # Default behavior from OSMWebWizard actuated traffic lights
            pass
        
        elif traffic_control == 'stop_signs':
            # Turn off all traffic lights
            for tl_id in tls_ids:
                traci.trafficlight.setProgram(tl_id, "off")
                        
    def save_results(self, intersection_type):
        """Save simulation results to CSV"""
        results_df = pd.DataFrame()
        
        for scenario, metrics_list in self.results.items():
            # Average results across multiple runs
            avg_metrics = {metric: np.mean([run[metric] for run in metrics_list]) 
                           for metric in METRICS}
            
            # Add scenario information
            traffic_load, traffic_control = scenario.split('_', 1)
            scenario_df = pd.DataFrame({
                'intersection_type': [intersection_type],
                'traffic_load': [traffic_load],
                'traffic_control': [traffic_control],
                **{metric: [avg_metrics[metric]] for metric in METRICS}
            })
            
            results_df = pd.concat([results_df, scenario_df], ignore_index=True)
        
        output_file = os.path.join(self.output_dir, f"{intersection_type}_results.csv")
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        return results_df

def run_all_simulations(num_runs=3, gui=False):
    """Run simulations for all combinations of parameters"""
    all_results = pd.DataFrame()
    
    for intersection_type in INTERSECTION_TYPES:
        config_file = BASE_PATHS[intersection_type]
        simulator = IntersectionSimulator(config_file)
        
        for traffic_load in TRAFFIC_LOADS:
            for traffic_control in TRAFFIC_CONTROL:
                print(f"Running {intersection_type} with {traffic_load} traffic and {traffic_control}")
                
                for run in range(num_runs):
                    simulator.run_simulation(traffic_load, traffic_control, seed=42+run, gui=gui)
        
        intersection_results = simulator.save_results(intersection_type)
        all_results = pd.concat([all_results, intersection_results], ignore_index=True)
    
    return all_results

def analyze_results(results_df):
    """Analyze and visualize the simulation results"""
    figures_dir = "figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # For each metric, create comparison plots
    for metric in METRICS:
        plt.figure(figsize=(14, 8))
        
        # Group by intersection type and traffic load
        for i, intersection_type in enumerate(INTERSECTION_TYPES):
            for j, traffic_load in enumerate(TRAFFIC_LOADS):
                subset = results_df[(results_df['intersection_type'] == intersection_type) & 
                                   (results_df['traffic_load'] == traffic_load)]
                
                if not subset.empty:
                    plt.subplot(len(INTERSECTION_TYPES), len(TRAFFIC_LOADS), i*len(TRAFFIC_LOADS) + j + 1)
                    x = subset['traffic_control']
                    y = subset[metric]
                    plt.bar(x, y)
                    plt.title(f"{intersection_type.capitalize()}, {traffic_load} traffic")
                    plt.ylabel(metric.replace('_', ' '))
                    plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{metric}_comparison.png"))
        plt.close()
    
    # Create overall comparison summary
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(METRICS):
        plt.subplot(2, 2, i+1)
        
        # Pivot data for grouped bar chart
        pivot_df = results_df.pivot_table(
            index=['traffic_load', 'traffic_control'],
            columns='intersection_type',
            values=metric
        ).reset_index()
        
        # Plot groups of bars
        bar_width = 0.35
        x = np.arange(len(pivot_df))
        
        for j, intersection_type in enumerate(INTERSECTION_TYPES):
            plt.bar(x + j*bar_width, pivot_df[intersection_type], 
                   width=bar_width, label=intersection_type.capitalize())
        
        plt.xlabel('Traffic Load - Control Method')
        plt.ylabel(metric.replace('_', ' '))
        plt.title(f'{metric.replace("_", " ").title()} Comparison')
        plt.xticks(x + bar_width/2, 
                  [f"{row['traffic_load']}-{row['traffic_control']}" 
                   for _, row in pivot_df.iterrows()], 
                  rotation=90)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "overall_comparison.png"))
    plt.close()
    print(f"Analysis complete. Visualizations saved to {figures_dir}")

def find_optimal_configurations(results_df):
    """Find optimal intersection configuration for each traffic scenario"""
    # Define weights for a balanced scoring 
    weights = {
        'avg_travel_time': 0.4,     
        'throughput': 0.3,          
        'emissions_CO2': 0.15,      
        'fuel_consumption': 0.15    
    }
    
    scoring_df = results_df.copy()
    
    # Normalize metrics to 0-1 scale for fair comparison
    for metric in METRICS:
        if metric == 'throughput':
            # For throughput, higher is better
            scoring_df[f"{metric}_score"] = (scoring_df[metric] - scoring_df[metric].min()) / \
                                           (scoring_df[metric].max() - scoring_df[metric].min())
        else:
            # For other metrics, lower is better
            scoring_df[f"{metric}_score"] = 1 - (scoring_df[metric] - scoring_df[metric].min()) / \
                                           (scoring_df[metric].max() - scoring_df[metric].min())
    
    # Calculate weighted score
    scoring_df['total_score'] = sum(weights[metric] * scoring_df[f"{metric}_score"] for metric in METRICS)
    
    # Find optimal configuration for each traffic load
    print("\nOptimal Intersection Configurations:")
    print("="*50)
    
    for load in TRAFFIC_LOADS:
        load_results = scoring_df[scoring_df['traffic_load'] == load].sort_values(by='total_score', ascending=False)
        
        if not load_results.empty:
            best_config = load_results.iloc[0]
            print(f"\nFor {load} traffic:")
            print(f"  Best configuration: {best_config['intersection_type'].capitalize()} with {best_config['traffic_control']}")
            print(f"  Performance metrics:")
            print(f"    - Average travel time: {best_config['avg_travel_time']:.2f} seconds")
            print(f"    - Throughput: {best_config['throughput']:.4f} vehicles/step")
            print(f"    - CO2 emissions: {best_config['emissions_CO2']:.2f} mg")
            print(f"    - Fuel consumption: {best_config['fuel_consumption']:.2f} ml")
            print(f"  Score: {best_config['total_score']:.2f}")
    
    return scoring_df
                   
if __name__ == "__main__":
    print("Starting Optimal Intersection Analyzer...")
    results = run_all_simulations(num_runs=3, gui=False)
    
    print("\nVisualizing results...")
    analyze_results(results)
    
    print("\nFinding optimal configurations...")
    scored_results = find_optimal_configurations(results)
    
    print("\nAnalysis complete!")