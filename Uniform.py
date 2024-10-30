import numpy as np
from scipy.stats import rayleigh
import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd

# Function definitions

def generate_channel_coefficients(M, sigma_h, sigma_g):
    """
    Generate channel coefficients for users to the legitimate receiver (h)
    and to the eavesdropper (g) using Rayleigh fading.

    Returns:
    h (ndarray): Channel gains from users to the legitimate receiver.
    g (ndarray): Channel gains from users to the eavesdropper.
    """
    h = rayleigh.rvs(loc=0, scale=sigma_h, size=M)
    # Ensure that h[0] has a minimum value to avoid deep fades
    h[0] = np.sqrt(np.pi - np.sqrt(4 - np.pi)) / np.sqrt(2)
    g = rayleigh.rvs(loc=0, scale=sigma_g, size=M)
    # Ensure that the rest of h are at least as large as h[0]
    h[1:] = np.maximum(h[1:], h[0])
    return h, g

def optimize_precoding_matrix(M, h, g, c_squared, P, with_CSI):
    """
    Optimize the precoding matrix for artificial noise generation
    to maximize the MSE at the eavesdropper while satisfying power constraints.

    Parameters:
    M (int): Number of users.
    h (ndarray): Channel gains to the legitimate receiver.
    g (ndarray): Channel gains to the eavesdropper.
    c_squared (float): Scaling factor squared.
    P (float): Power constraint.
    with_CSI (bool): Flag indicating if eavesdropper's CSI is known.

    Returns:
    d (ndarray): Optimized artificial noise power allocation vector.
    """
    # Variable for squared artificial noise power allocation
    d_squared = cp.Variable(M - 1)

    if with_CSI:
        # Objective function when eavesdropper's CSI is known
        objective = cp.Maximize(cp.sum(cp.multiply(
            d_squared,
            ((h[:M-1]**2 / h[M-1]**2) * g[M-1]**2
             - 2 * (h[:M-1] / h[M-1]) * g[:M-1] * g[M-1]
             + g[:M-1]**2)
        )))
    else:
        # Objective function when eavesdropper's CSI is unknown
        objective = cp.Maximize(cp.sum(cp.multiply(
            d_squared,
            (2 * (h[:M-1]**2 / h[M-1]**2)
             - (np.pi / 2) * (h[:M-1] / h[M-1])
             + 1)
        )))

    # Power constraints
    constraints = [
        d_squared <= P - (c_squared / h[:M-1]**2),  # Individual power constraints
        cp.sum(cp.multiply(d_squared, (h[:M-1]**2 / h[M-1]**2))) <= P - (c_squared / h[M-1]**2)
    ]

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)

    if prob.status not in ["infeasible", "unbounded"]:
        d_squared_value = d_squared.value
        # Ensure non-negativity of power allocations
        d_squared_value[d_squared_value < 0] = 0
        d = np.sqrt(d_squared_value)
    else:
        # If problem is infeasible, set d to zeros
        d = np.zeros(M - 1)

    return d

def create_reduced_row_echelon_form(h, M):
    """
    Create the matrix A_prime used to generate the artificial noise precoding matrix.

    Parameters:
    h (ndarray): Channel gains to the legitimate receiver.
    M (int): Number of users.

    Returns:
    A_prime (ndarray): Matrix in reduced row-echelon form.
    """
    A_prime = np.zeros((M - 1, M))
    for m in range(M - 1):
        A_prime[m, m] = 1
    A_prime[:, M - 1] = -h[:M - 1] / h[M - 1]
    return A_prime

def create_precoding_matrix(A_prime, d):
    """
    Create the artificial noise precoding matrix A.

    Parameters:
    A_prime (ndarray): Reduced row-echelon form matrix.
    d (ndarray): Artificial noise power allocation vector.

    Returns:
    A (ndarray): Precoding matrix for artificial noise.
    """
    A_prime = A_prime[:, :M-1] 
    A = np.diag(d) @ A_prime
    return A

def generate_gamma(M, k):
    """
    Generate the pre-processed input data gamma for all users.

    Parameters:
    M (int): Number of users.
    k (int): Dimension of the data vector.

    Returns:
    gamma (ndarray): Pre-processed input data matrix of shape (M, k).
    """
    a = np.sqrt(3)  # Parameter for uniform distribution
    gamma = np.random.uniform(-a, a, (M, k))
    return gamma

def compute_transmit_signals(A_without_CSI, A_with_CSI, h, gamma, c, M, k):
    """
    Compute the transmit signals for users with and without eavesdropper's CSI.

    Parameters:
    A_without_CSI (ndarray): Precoding matrix without eavesdropper's CSI.
    A_with_CSI (ndarray): Precoding matrix with eavesdropper's CSI.
    h (ndarray): Channel gains to the legitimate receiver.
    gamma (ndarray): Pre-processed input data.
    c (float): Scaling factor.
    M (int): Number of users.
    k (int): Dimension of the data vector.

    Returns:
    x (ndarray): Transmit signals without artificial noise.
    x_without_CSI (ndarray): Transmit signals with artificial noise (without eavesdropper's CSI).
    x_with_CSI (ndarray): Transmit signals with artificial noise (with eavesdropper's CSI).
    """
    V_without_CSI = np.random.randn(M - 1, k)  # Random noise vector for artificial noise
    V_with_CSI = np.random.randn(M - 1, k)
    w_without_CSI = np.vstack([A_without_CSI @ V_without_CSI, np.zeros((1, k))])  # Artificial noise without CSI
    w_with_CSI = np.vstack([A_with_CSI @ V_with_CSI, np.zeros((1, k))])           # Artificial noise with CSI
    x_without_CSI = c * (gamma / h[:, np.newaxis]) + w_without_CSI  # Transmit signals with artificial noise
    x_with_CSI = c * (gamma / h[:, np.newaxis]) + w_with_CSI
    x = c * (gamma / h[:, np.newaxis])  # Transmit signals without artificial noise
    return x, x_without_CSI, x_with_CSI

def compute_received_signals(x, h, g, x_without_CSI, x_with_CSI, sigma_y, sigma_z):
    """
    Compute the received signals at the legitimate receiver and the eavesdropper.

    Parameters:
    x (ndarray): Transmit signals without artificial noise.
    h (ndarray): Channel gains to the legitimate receiver.
    g (ndarray): Channel gains to the eavesdropper.
    x_without_CSI (ndarray): Transmit signals with artificial noise (without eavesdropper's CSI).
    x_with_CSI (ndarray): Transmit signals with artificial noise (with eavesdropper's CSI).
    sigma_y (float): Noise standard deviation at the legitimate receiver.
    sigma_z (float): Noise standard deviation at the eavesdropper.

    Returns:
    y (ndarray): Received signal at the legitimate receiver.
    z_with_CSI (ndarray): Received signal at the eavesdropper with CSI.
    z_without_CSI (ndarray): Received signal at the eavesdropper without CSI.
    """
    n_y = sigma_y * np.random.randn()  # Noise at legitimate receiver
    n_z = sigma_z * np.random.randn()  # Noise at eavesdropper
    y = np.sum(h * x, axis=0) + n_y  # Received signal at legitimate receiver
    z_with_CSI = np.sum(g * x_with_CSI, axis=0) + n_z  # Eavesdropper's received signal with CSI
    z_without_CSI = np.sum(g * x_without_CSI, axis=0) + n_z  # Eavesdropper's received signal without CSI
    return y, z_with_CSI, z_without_CSI

def compute_objective_function(gamma):
    """
    Compute the sum of the pre-processed inputs across all users.

    Parameters:
    gamma (ndarray): Pre-processed input data.

    Returns:
    s (ndarray): Sum of the pre-processed inputs.
    """
    s = np.sum(gamma, axis=0)
    return s

def calculate_MSE_new(M, c, h, g, A_with_CSI, A_without_CSI, A_SVD, sigma_y, sigma_z, k):
    """
    Calculate the Mean Squared Error (MSE) for the legitimate receiver and the eavesdropper.

    Parameters:
    M (int): Number of users.
    c (float): Scaling factor.
    h (ndarray): Channel gains to the legitimate receiver.
    g (ndarray): Channel gains to the eavesdropper.
    A_with_CSI (ndarray): Precoding matrix with eavesdropper's CSI.
    A_without_CSI (ndarray): Precoding matrix without eavesdropper's CSI.
    A_SVD (ndarray): Precoding matrix using SVD.
    sigma_y (float): Noise standard deviation at the legitimate receiver.
    sigma_z (float): Noise standard deviation at the eavesdropper.
    k (int): Dimension of the data vector.

    Returns:
    D (float): MSE at the legitimate receiver.
    S_with_CSI (float): MSE at the eavesdropper with CSI.
    S_without_CSI (float): MSE at the eavesdropper without CSI.
    S_noiseless (float): MSE at the eavesdropper without artificial noise.
    S_SVD (float): MSE at the eavesdropper using SVD.
    """
    # Calculate MSE at the legitimate receiver
    denominator_D = c**2 * M / k + sigma_y**2
    D = M - (c**2 * M**2) / (k * denominator_D)

    def calculate_S(A):
        # Calculate MSE at the eavesdropper
        numerator = c**2 * (np.sum(g / h))**2
        denominator = c**2 * np.sum((g / h)**2) + k * (np.linalg.norm(A)**2 + sigma_z**2)
        return M - numerator / denominator

    S_with_CSI = calculate_S(A_with_CSI)        # Eavesdropper with CSI
    S_without_CSI = calculate_S(A_without_CSI)  # Eavesdropper without CSI
    S_noiseless = calculate_S(np.zeros_like(A_with_CSI))  # Eavesdropper without artificial noise
    S_SVD = calculate_S(A_SVD)                  # Eavesdropper using SVD
    return D, S_with_CSI, S_without_CSI, S_noiseless, S_SVD

def create_precoding_matrix_SVD(h, P):
    """
    Create a precoding matrix using the Singular Value Decomposition (SVD) method.

    Parameters:
    h (ndarray): Channel gains to the legitimate receiver.
    P (float): Power constraint.

    Returns:
    A_SVD (ndarray): Precoding matrix using SVD.
    """
    v = np.random.randn(len(h))  # Random vector
    v_orthogonal = v - (np.dot(h, v) / np.dot(h, h)) * h  # Orthogonal component to h
    scaling_factor = np.sqrt(P / np.dot(v_orthogonal, v_orthogonal))  # Scaling to meet power constraint
    A_SVD = v_orthogonal * scaling_factor
    return A_SVD

# Initialization of parameters
M = 10               # Number of users
P = 1                # Power constraint
sigma_y = 0.1        # Noise standard deviation at the legitimate receiver
sigma_z = 0.1        # Noise standard deviation at the eavesdropper
sigma_h = 1          # Scale parameter for h
sigma_g = 1          # Scale parameter for g
SNR_dB = np.arange(0, 15, 2)  # SNR values in dB
k = 1                # Dimension of the data vector
num_runs = 100       # Number of simulation runs per SNR value

# Simulation
print('Starting simulations...')
MSE_legitimate_avg = np.zeros(len(SNR_dB))                 # Average MSE at legitimate receiver
MSE_eavesdropper_with_CSI_avg = np.zeros(len(SNR_dB))      # Average MSE at eavesdropper with CSI
MSE_eavesdropper_without_CSI_avg = np.zeros(len(SNR_dB))   # Average MSE at eavesdropper without CSI
MSE_eavesdropper_noiseless_avg = np.zeros(len(SNR_dB))     # Average MSE at eavesdropper without artificial noise
MSE_eavesdropper_SVD_avg = np.zeros(len(SNR_dB))           # Average MSE at eavesdropper using SVD
all_mse_data = []                                          # List to store all MSE data

for idx, snr_db in enumerate(SNR_dB):
    print(f'Processing SNR = {snr_db} dB')
    MSE_legitimate = np.zeros(num_runs)
    MSE_eavesdropper_with_CSI = np.zeros(num_runs)
    MSE_eavesdropper_without_CSI = np.zeros(num_runs)
    MSE_eavesdropper_noiseless = np.zeros(num_runs)
    MSE_eavesdropper_SVD = np.zeros(num_runs)

    for run in range(num_runs):
        if run % 1000 == 0:
            print(f'  Run {run} of {num_runs}')

        # Generate channel coefficients
        h, g = generate_channel_coefficients(M, sigma_h, sigma_g)
        SNR_linear = 10**(snr_db / 10)
        c_squared = SNR_linear * sigma_y**2 / M
        c = np.sqrt(c_squared)

        # Optimize precoding matrices without and with eavesdropper's CSI
        d_without_CSI = optimize_precoding_matrix(M, h, g, c_squared, P, with_CSI=False)
        A_prime = create_reduced_row_echelon_form(h, M)
        A_without_CSI = create_precoding_matrix(A_prime, d_without_CSI)

        d_with_CSI = optimize_precoding_matrix(M, h, g, c_squared, P, with_CSI=True)
        A_with_CSI = create_precoding_matrix(A_prime, d_with_CSI)

        # Generate pre-processed input data
        gamma = generate_gamma(M, k)

        # Compute transmit signals
        x, x_without_CSI, x_with_CSI = compute_transmit_signals(
            A_without_CSI, A_with_CSI, h, gamma, c, M, k
        )

        # Compute the desired sum (objective function)
        s = compute_objective_function(gamma)

        # Compute received signals at legitimate receiver and eavesdropper
        y, z_with_CSI, z_without_CSI = compute_received_signals(
            x, h, g, x_without_CSI, x_with_CSI, sigma_y, sigma_z
        )

        # Create SVD precoding matrix
        A_SVD = create_precoding_matrix_SVD(h, P)

        # Calculate MSEs
        D, S_with_CSI, S_without_CSI, S_noiseless, S_SVD = calculate_MSE_new(
            M, c, h, g, A_with_CSI, A_without_CSI, A_SVD, sigma_y, sigma_z, k
        )

        # Store MSEs for this run
        MSE_legitimate[run] = D
        MSE_eavesdropper_with_CSI[run] = S_with_CSI
        MSE_eavesdropper_without_CSI[run] = S_without_CSI
        MSE_eavesdropper_noiseless[run] = S_noiseless
        MSE_eavesdropper_SVD[run] = S_SVD

    # Compute average MSEs over all runs for this SNR value
    MSE_legitimate_avg[idx] = np.mean(MSE_legitimate)
    MSE_eavesdropper_with_CSI_avg[idx] = np.mean(MSE_eavesdropper_with_CSI)
    MSE_eavesdropper_without_CSI_avg[idx] = np.mean(MSE_eavesdropper_without_CSI)
    MSE_eavesdropper_noiseless_avg[idx] = np.mean(MSE_eavesdropper_noiseless)
    MSE_eavesdropper_SVD_avg[idx] = np.mean(MSE_eavesdropper_SVD)

    # Store results in data list
    all_mse_data.append({
        'SNR(dB)': snr_db,
        'MSE_Legitimate': MSE_legitimate_avg[idx],
        'MSE_Eavesdropper_with_CSI': MSE_eavesdropper_with_CSI_avg[idx],
        'MSE_Eavesdropper_without_CSI': MSE_eavesdropper_without_CSI_avg[idx],
        'MSE_Eavesdropper_noiseless': MSE_eavesdropper_noiseless_avg[idx]
    })

# Convert the data to a DataFrame and save it as a CSV file
mse_df = pd.DataFrame(all_mse_data)
filename = f'MSE_vs_SNR_data_{M}users_{num_runs}runs.dat'
mse_df.to_csv(filename, index=False, sep=' ')
print(filename)
print(mse_df)

# Plotting results
plt.figure()
plt.plot(SNR_dB, MSE_legitimate_avg, 'b-o', label='Legitimate Receiver')
plt.plot(SNR_dB, MSE_eavesdropper_with_CSI_avg, 'r-s', label='Eavesdropper with CSI')
plt.plot(SNR_dB, MSE_eavesdropper_without_CSI_avg, 'g-^', label='Eavesdropper without CSI')
plt.plot(SNR_dB, MSE_eavesdropper_noiseless_avg, 'm-*', label='Eavesdropper Noise-less')
plt.plot(SNR_dB, MSE_eavesdropper_SVD_avg, 'k-x', label='Eavesdropper SVD')
plt.xlabel('SNR (dB)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE vs. SNR for Legitimate Receiver and Eavesdropper')
plt.legend()
plt.grid(True)
plt.show()

print('Simulation completed.')
