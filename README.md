# Photovoltaic System AC Power Prediction

## Overview
This project involves the development of a dataset and an advanced neural network (NN) model to predict the AC power output (Pac) of a photovoltaic system.

---

## Objectives
1. **Dataset Generation**: Simulate the photovoltaic system using MATLAB and Simulink to create a dataset.
2. **Neural Network Model**: Train a neural network to predict the AC power output (Pac).

---

## Dataset Generation
The dataset was generated using MATLAB scripts and Simulink. Since the simulation required manual input combinations for multiple runs, a custom MATLAB script was written to automate the process. Due to time constraints, 1271 data points were collected.

### Key Details:
- Inputs: **Gir** (solar irradiance) and **Ta** (ambient temperature).
- Output: **Pac** (AC power output).
- Each simulation was run for at least 0.2 seconds to ensure the system reached a steady state.

---

## Methodology
1. **Simulation**:
   - Used the provided Simulink model to simulate the photovoltaic system's behavior.
   - Automated input variation and data collection using MATLAB scripting.

2. **Model Development**:
   - Built a neural network model for predicting Pac.
   - Optimized the model by tuning hyperparameters, implementing regularization, and evaluating performance.

---

## File Structure
