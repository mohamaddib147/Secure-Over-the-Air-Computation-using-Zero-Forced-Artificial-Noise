import numpy as np
from scipy.stats import rayleigh
import cvxpy as cp
import matplotlib.pyplot as plt

# Function definitions

def generate_channel_coefficients(M, sigma_h, sigma_g):
    h = rayleigh.rvs(loc=0, scale=sigma_h, size=M)
    h[0] = np.sqrt(np.pi - np.sqrt(4 - np.pi)) / np.sqrt(2)
    g = rayleigh.rvs(loc=0, scale=sigma_g, size=M)
    h[1:] = np.maximum(h[1:], h[0])
    return h, g



def optimize_precoding_matrix(M, h, g, c_squared, P, with_CSI):
    d_squared = cp.Variable(M - 1)
    if with_CSI:
        # Element-wise multiplication is correctly used here; ensure clarity and correctness
        objective = cp.Maximize(cp.sum(cp.multiply(d_squared, ((h[:M-1]**2 / h[M-1]**2) * g[M-1]**2 - 2 * (h[:M-1] / h[M-1]) * g[:M-1] * g[M-1] + g[:M-1]**2))))
    else:
        objective = cp.Maximize(cp.sum(cp.multiply(d_squared, (2 * (h[:M-1]**2 / h[M-1]**2) - (np.pi / 2) * (h[:M-1] / h[M-1]) + 1))))
    
    # Adjust constraints to ensure they are vectorized and correctly interpreted
    constraints = [
        d_squared <= P - (c_squared / h[:M-1]**2),
        cp.sum(cp.multiply(d_squared, (h[:M-1]**2 / h[M-1]**2))) <= P - (c_squared / h[M-1]**2)
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if prob.status not in ["infeasible", "unbounded"]:
        d_squared_value = d_squared.value
        # Ensure non-negativity
        d_squared_value[d_squared_value < 0] = 0
        d = np.sqrt(d_squared_value)
    else:
        d = np.zeros(M - 1)
    
    return d

def create_reduced_row_echelon_form(h, M):
    A_prime = np.zeros((M - 1, M))
    for m in range(M - 1):
        A_prime[m, m] = 1
    A_prime[:, M - 1] = -h[:M - 1] / h[M - 1]
    return A_prime

def create_precoding_matrix(A_prime, d):
    A_prime = A_prime[:, :M-1]  # Keep only the first M-1 columns
    return np.diag(d) @ A_prime


def generate_gamma(k, M):
    Sigma = np.eye(k)
    gamma = np.random.multivariate_normal(np.zeros(k), Sigma, M).T
    return gamma

def compute_transmit_signals(A_without_CSI, A_with_CSI, h, gamma, c, M, k):
    V_without_CSI = np.random.randn(M - 1, k)  # Change to M-1 rows
    V_with_CSI = np.random.randn(M - 1, k)  # Change to M-1 rows
    w_without_CSI = np.vstack([A_without_CSI @ V_without_CSI, np.zeros((1, k))])
    w_with_CSI = np.vstack([A_with_CSI @ V_with_CSI, np.zeros((1, k))])
    x_without_CSI = c * (h**(-1))[:, np.newaxis] * gamma + w_without_CSI
    x_with_CSI = c * (h**(-1))[:, np.newaxis] * gamma + w_with_CSI
    x = c * (h**(-1))[:, np.newaxis] * gamma
    return x, x_without_CSI, x_with_CSI

def compute_received_signals(x, h, g, x_without_CSI, x_with_CSI, sigma_y, sigma_z):
    n_y = sigma_y * np.random.randn()
    n_z = sigma_z * np.random.randn()
    y = np.sum(h * x, axis=0) + n_y
    z_with_CSI = np.sum(g * x_with_CSI, axis=0) + n_z
    z_without_CSI = np.sum(g * x_without_CSI, axis=0) + n_z
    return y, z_with_CSI, z_without_CSI

def compute_objective_function(gamma):
    return np.sum(gamma, axis=0)

def calculate_MSE_new(M, c, h, g, A_with_CSI, A_without_CSI, A_SVD, sigma_y, sigma_z, k):
    D = M - (c**2 * M**2) / (c**2 * M + k * sigma_y)
    def calculate_S(A):
        numerator = c**2 * (np.sum(g / h))**2
        denominator = c**2 * np.sum((g**2) / (h**2)) + k * (np.linalg.norm(A)**2 + sigma_z)
        return M - numerator / denominator
    S_with_CSI = calculate_S(A_with_CSI)
    S_without_CSI = calculate_S(A_without_CSI)
    S_noiseless = calculate_S(0)
    S_SVD = calculate_S(A_SVD)
    return D, S_with_CSI, S_without_CSI, S_noiseless, S_SVD

def create_precoding_matrix_SVD(h, P):
    v = np.random.randn(len(h))
    v_orthogonal = v - (np.dot(h, v) / np.dot(h, h)) * h
    scaling_factor = np.sqrt(P / np.dot(v_orthogonal, v_orthogonal))
    A_SVD = v_orthogonal * scaling_factor
    return A_SVD

# Initialization of parameters
M = 10
P = 1
sigma_y = 0.1
sigma_z = 0
sigma_h = 1
sigma_g = 1
SNR_dB = np.arange(0, 15, 2)
k = 1
num_runs = 100

# Simulation
print('Starting simulations...')
MSE_legitimate_avg = np.zeros(len(SNR_dB))
MSE_eavesdropper_with_CSI_avg = np.zeros(len(SNR_dB))
MSE_eavesdropper_without_CSI_avg = np.zeros(len(SNR_dB))
MSE_eavesdropper_noiseless_avg = np.zeros(len(SNR_dB))
MSE_eavesdropper_SVD_avg = np.zeros(len(SNR_dB))

for idx, snr_db in enumerate(SNR_dB):
    print(f'Processing SNR = {snr_db} dB')
    MSE_legitimate = np.zeros(num_runs)
    MSE_eavesdropper_with_CSI = np.zeros(num_runs)
    MSE_eavesdropper_without_CSI = np.zeros(num_runs)
    MSE_eavesdropper_noiseless = np.zeros(num_runs)
    MSE_eavesdropper_SVD = np.zeros(num_runs)

    for run in range(num_runs):
        if run % 100 == 0:
            print(f'  Run {run} of {num_runs}')

        h, g = generate_channel_coefficients(M, sigma_h, sigma_g)
        SNR_linear = 10**(snr_db / 10)
        c_squared = SNR_linear * sigma_y / M
        c = np.sqrt(c_squared)

        d_without_CSI = optimize_precoding_matrix(M, h, g, c_squared, P, with_CSI=False)
        A_prime = create_reduced_row_echelon_form(h, M)
        A_without_CSI = create_precoding_matrix(A_prime, d_without_CSI)

        d_with_CSI = optimize_precoding_matrix(M, h, g, c_squared, P, with_CSI=True)
        A_with_CSI = create_precoding_matrix(A_prime, d_with_CSI)

        gamma = generate_gamma(k, M)
        x, x_without_CSI, x_with_CSI = compute_transmit_signals(A_without_CSI, A_with_CSI, h, gamma, c, M, k)

        s = compute_objective_function(gamma)
        y, z_with_CSI, z_without_CSI = compute_received_signals(x, h, g, x_without_CSI, x_with_CSI, sigma_y, sigma_z)

        A_SVD = create_precoding_matrix_SVD(h, P)
        D, S_with_CSI, S_without_CSI, S_noiseless, S_SVD = calculate_MSE_new(M, c, h, g, A_with_CSI, A_without_CSI, A_SVD, sigma_y, sigma_z, k)

        MSE_legitimate[run] = D
        MSE_eavesdropper_with_CSI[run] = S_with_CSI
        MSE_eavesdropper_without_CSI[run] = S_without_CSI
        MSE_eavesdropper_noiseless[run] = S_noiseless
        MSE_eavesdropper_SVD[run] = S_SVD

    MSE_legitimate_avg[idx] = np.mean(MSE_legitimate)
    MSE_eavesdropper_with_CSI_avg[idx] = np.mean(MSE_eavesdropper_with_CSI)
    MSE_eavesdropper_without_CSI_avg[idx] = np.mean(MSE_eavesdropper_without_CSI)
    MSE_eavesdropper_noiseless_avg[idx] = np.mean(MSE_eavesdropper_noiseless)
    MSE_eavesdropper_SVD_avg[idx] = np.mean(MSE_eavesdropper_SVD)

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
