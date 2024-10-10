# Secure Over-the-Air Computation using Zero-Forced Artificial Noise

This repository contains the code used for my thesis project, **"Enhancing Security in Over-the-Air Computation Systems with Uniform Input Distributions"**. The project investigates the impact of input distributions (Gaussian vs. Uniform) on the security and performance of Over-the-Air Computation (OAC) systems employing zero-forced artificial noise. The findings show that uniformly distributed inputs can significantly improve the security of OAC systems without degrading the performance for legitimate users.

## Overview

Over-the-air computation (OAC) leverages the superposition property of wireless channels to compute functions directly over the air, reducing communication overhead in distributed systems such as IoT networks. However, these systems are vulnerable to eavesdropping due to the broadcast nature of wireless channels.

This research compares the effects of two different input distributions:
- **Gaussian distribution** (original system)
- **Uniform distribution** (modified system)

The repository includes the MATLAB and Python implementations for both distributions and the simulations used to evaluate the Mean Squared Error (MSE) at the legitimate receiver and the eavesdropper under different Signal-to-Noise Ratio (SNR) levels.

## Repository Contents

The repository is structured as follows:

- **`Zero_Forced_Artificial_Noise.py`**: Python implementation of the OAC system using **Gaussian input distribution**.
- **`Zero_Forced_Artificial_Noise.m`**: MATLAB implementation of the OAC system using **Gaussian input distribution**.
- **`Uniform.py`**: Python implementation of the OAC system using **Uniform input distribution**.
- **Simulation Data and Results**: The results from the simulations comparing MSE vs. SNR for both Gaussian and Uniform input distributions.
  - `MSE vs SNR 10000run (Gaussian dist).png`
  - `MSE vs SNR 10000run (Uniform dist).png`
  - `MSE_vs_SNR_data_10users_100000runs(gaussian).dat`
- **`README.md`**: This file.

## Running the Code

### Python
To run the simulations in Python:

1. Clone the repository:
    ```bash
    git clone https://github.com/mohamaddib147/Secure-Over-the-Air-Computation-using-Zero-Forced-Artificial-Noise.git
    ```

2. Install the required dependencies:
    ```bash
    pip install numpy matplotlib
    ```

3. Run the desired file:
   - For **Gaussian distribution** (original system):
     ```bash
     python Zero_Forced_Artificial_Noise.py
     ```
   - For **Uniform distribution** (modified system):
     ```bash
     python Uniform.py
     ```

### MATLAB
To run the MATLAB simulation:
- Open the `Zero_Forced_Artificial_Noise.m` file in MATLAB and run the script.
  
## Results

The results of the simulations compare the Mean Squared Error (MSE) at both the legitimate receiver and the eavesdropper under different SNR levels. The figures showing the performance for both Gaussian and Uniform distributions are provided below.

- **MSE vs SNR (Gaussian Distribution)**

  ![Gaussian Distribution](./MSE%20vs%20SNR%20100000run%20(Gaussion%20dist).png)

- **MSE vs SNR (Uniform Distribution)**

  ![Uniform Distribution](./MSE%20vs%20SNR%20100000run%20(Uniform%20dist).png)




