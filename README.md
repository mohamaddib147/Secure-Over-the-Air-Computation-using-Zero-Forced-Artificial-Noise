

# Secure Over-the-Air Computation using Zero-Forced Artificial Noise

This repository contains the code and simulation results for my thesis project, **"Enhancing Security in Over-the-Air Computation Systems with Uniform Input Distributions"**. The project investigates the impact of input distributions (Gaussian vs. Uniform) on the security and performance of Over-the-Air Computation (OAC) systems employing zero-forced artificial noise. The findings demonstrate that uniformly distributed inputs can significantly improve the security of OAC systems without degrading performance for legitimate users.

## Overview

Over-the-Air Computation (OAC) leverages the superposition property of wireless channels to compute functions directly over the air, reducing communication overhead in distributed systems such as IoT networks and wireless sensor networks. However, these systems are susceptible to eavesdropping due to the broadcast nature of wireless channels.

This research compares the effects of two different input distributions on the security and accuracy of OAC systems:

- **Gaussian Distribution** (Original System)
- **Uniform Distribution** (Modified System)

The repository includes implementations in Python and MATLAB for both distributions, along with simulations used to evaluate the Mean Squared Error (MSE) at the legitimate receiver and the eavesdropper under different Signal-to-Noise Ratio (SNR) levels.

## Repository Contents

The repository is structured as follows:

- **`Zero_Forced_Artificial_Noise.py`**: Python implementation of the OAC system using **Gaussian input distribution**.
- **`Zero_Forced_Artificial_Noise.m`**: MATLAB implementation of the OAC system using **Gaussian input distribution**.
- **`Uniform.py`**: Python implementation of the OAC system using **Uniform input distribution**.
- **`README.md`**: This file.
- **Simulation Data and Results**:
  - **Figures**:
    - `MSE_vs_SNR_Gaussian.png`: Plot of MSE vs. SNR for the Gaussian input distribution.
    - `MSE_vs_SNR_Uniform.png`: Plot of MSE vs. SNR for the Uniform input distribution.
  - **Data Files**:
    - `MSE_vs_SNR_data_Gaussian.dat`: Simulation data for the Gaussian distribution.
    - `MSE_vs_SNR_data_Uniform.dat`: Simulation data for the Uniform distribution.

## Getting Started

### Prerequisites

To run the simulations, you need the following installed on your system:

- **Python 3.x** with the following packages:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `cvxpy`
  - `pandas`

- **MATLAB** (for the MATLAB implementation)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/mohamaddib147/Secure-Over-the-Air-Computation-using-Zero-Forced-Artificial-Noise.git
   cd Secure-Over-the-Air-Computation-using-Zero-Forced-Artificial-Noise
   ```

2. **Install Python Dependencies**

   You can install the required Python packages using `pip`:

   ```bash
   pip install numpy scipy matplotlib cvxpy pandas
   ```

   > **Note:** It's recommended to use a virtual environment to avoid conflicts with other projects.

## Running the Code

### Python Implementation

The Python scripts allow you to simulate the OAC system with either Gaussian or Uniform input distributions.

#### Running the Simulation with Gaussian Distribution

```bash
python Zero_Forced_Artificial_Noise.py
```

#### Running the Simulation with Uniform Distribution

```bash
python Uniform.py
```

#### Parameters

You can adjust simulation parameters within the scripts:

- **Number of Users (`M`)**
- **Power Constraint (`P`)**
- **Noise Standard Deviations (`sigma_y`, `sigma_z`)**
- **Channel Parameters (`sigma_h`, `sigma_g`)**
- **SNR Range (`SNR_dB`)**
- **Number of Runs (`num_runs`)**
- **Data Dimension (`k`)**

### MATLAB Implementation

To run the MATLAB simulation:

1. Open the `Zero_Forced_Artificial_Noise.m` file in MATLAB.
2. Adjust any parameters as needed within the script.
3. Run the script.

## Results

The simulations compare the Mean Squared Error (MSE) at both the legitimate receiver and the eavesdropper under varying SNR levels. The results illustrate how the choice of input distribution affects both security and performance.

### MSE vs. SNR (Gaussian Distribution)

![Gaussian Distribution](./MSE%20vs%20SNR%20100000run%20(Gaussion%20dist).png)

### MSE vs. SNR (Uniform Distribution)

![Uniform Distribution](./MSE%20vs%20SNR%20100000run%20(Uniform%20dist).png)

**Observations:**

- With **Uniform input distribution**, the MSE at the eavesdropper increases significantly compared to the Gaussian distribution, enhancing security.
- The MSE at the legitimate receiver remains relatively unaffected by the change in input distribution, maintaining performance.

## Project Structure

Here's a detailed breakdown of the key files and their functionalities:

### Python Files

- **`Zero_Forced_Artificial_Noise.py`**

  - Implements the OAC system with Gaussian inputs.
  - Simulates the transmission, reception, and MSE calculation.
  - Includes functions for channel generation, precoding matrix optimization, signal computation, and MSE evaluation.

- **`Uniform.py`**

  - Similar to the above but uses Uniform input distribution.
  - Adjusted to reflect the differences in input processing due to the uniform distribution.

### MATLAB File

- **`Zero_Forced_Artificial_Noise.m`**

  - MATLAB version of the OAC system simulation with Gaussian inputs.
  - Useful for users more comfortable with MATLAB or for cross-verification of results.

### Simulation Data and Figures

- **Data Files (`.dat`)**

  - Contain raw simulation results (e.g., MSE values at different SNR levels).
  - Can be used for further analysis or plotting.

- **Figures (`.png`)**

  - Visual representations of the simulation results.
  - Useful for quickly understanding the impact of different input distributions.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please open an issue or submit a pull request.


## Contact

For any questions or inquiries, please contact:

- **Name:** Mohammad Dib
- **Email:** [mdeeb@kth.se](mailto:mdeeb@kth.se)

## Acknowledgments

- **Supervisor:** Dr. Xiaojing Yan, Thanks for your guidance and support throughout the project.
- **Institution:** KTH Royal Institute of Technology, School of Electrical Engineering and Computer Science


## References

- Relevant literature and papers on Over-the-Air Computation and wireless security.
- Massny, L., & Wachter-Zeh, A. (2023). *Secure Over-the-Air Computation Using Zero-Forced Artificial Noise*. In *2023 IEEE Information Theory Workshop (ITW)* (pp. 221–226). [https://doi.org/10.1109/ITW55543.2023.10161677](https://doi.org/10.1109/ITW55543.2023.10161677)

## Additional Notes

- **Reproducibility:** The random seed is not fixed in the simulations. To reproduce exact results, set a random seed using `np.random.seed(seed_value)` at the beginning of the scripts.
- **Performance:** The simulation time increases with the number of runs (`num_runs`). Adjust accordingly based on your computational resources.

---

Thank you for your interest in this project! Your feedback and suggestions are highly appreciated.
