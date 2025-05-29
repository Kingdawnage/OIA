# Traffic Intersection Optimization Analyzer

## 🧭 Overview
This project simulates and compares the performance of two intersection types—**classic 4-leg intersections** and **roundabouts**—under various traffic control strategies and traffic load scenarios using **[SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/)**.

The goal is to determine the most effective intersection configuration and control method for optimizing key traffic performance metrics.

---

## 📦 Requirements

### Software
- [SUMO (Simulation of Urban MObility)](https://sumo.dlr.de/)
  - Ensure the `SUMO_HOME` environment variable is set.

### Python
- Python 3.x

### Python Packages

Install required packages via pip:

```bash
pip install pandas matplotlib numpy traci
```

---

## 📁 Project Structure

```plaintext
.
├── Classic 4-leg Intersection/      # SUMO config for classic signalized intersection
├── Roundabout 4-leg Intersection/  # SUMO config for roundabout intersection
├── results/                         # Output CSV files with performance metrics
├── figures/                         # Generated plots and comparison charts
├── main.py                          # Main script for simulation and analysis
└── README.md                        # Project documentation
```

---

## ✨ Features

- Simulates two intersection types:
  - Classic 4-leg
  - Roundabout 4-leg

- Tests three traffic control methods:
  - `no_tls` (no traffic lights)
  - `actuated_tls` (adaptive signal control)
  - `stop_signs`

- Evaluates performance under three traffic load conditions:
  - Low (100% of base traffic)
  - Medium (150% of base traffic)
  - High (200% of base traffic)

- Measures key traffic metrics:
  - **Average travel time**
  - **Vehicle throughput**
  - **CO₂ emissions**
  - **Fuel consumption**

- Visualizes results with comparison charts
- Identifies optimal control configurations based on weighted evaluation

---

## ▶️ Usage

1. **Install SUMO** and ensure `SUMO_HOME` is correctly set:

```bash
export SUMO_HOME="/path/to/sumo"
```

2. **Run the main script**:

```bash
python main.py
```

This will:
- Run simulations for all combinations of:
  - Intersection type
  - Traffic control method
  - Traffic load
- Save results to the `results/` directory
- Generate charts in the `figures/` directory
- Print the best-performing configurations per traffic scenario

---

## 📊 Interpreting Results

- **Generated Charts**:
  - `avg_travel_time_comparison.png`
  - `throughput_comparison.png`
  - `co2_emissions_comparison.png`
  - `fuel_consumption_comparison.png`
  - `overall_comparison.png`

- **Optimization Scoring Weights**:
  - Average travel time: **40%**
  - Throughput: **30%**
  - CO₂ emissions: **15%**
  - Fuel consumption: **15%**

> 📌 Lower scores in travel time, emissions, and fuel consumption are better; higher throughput is better.

---

## 📬 Contact & Contributions

Feel free to open issues or contribute improvements via pull requests.
