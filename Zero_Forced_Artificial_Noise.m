%% Initialization of parameters
M = 10; % Number of users
P = 1; % Power constraint for each user
sigma_y = 0.1; % Noise standard deviation at the legitimate receiver
sigma_z = 0; % Eavesdropper's noise variance (worst case scenario)
sigma_h = 1; % Scale parameter for Rayleigh distribution of h
sigma_g = 1; % Scale parameter for Rayleigh distribution of g
SNR_dB = 0:2:14; % Signal-to-Noise Ratio values in dB
k = 1; % Number of coordinates for input data
num_runs = 10000; % Number of Monte Carlo runs

%% Simulation
disp('Starting simulations...');
% Pre-allocate arrays to store averaged MSE results
MSE_legitimate_avg = zeros(1, length(SNR_dB));
MSE_eavesdropper_with_CSI_avg = zeros(1, length(SNR_dB));
MSE_eavesdropper_without_CSI_avg = zeros(1, length(SNR_dB));
MSE_eavesdropper_noiseless_avg = zeros(1, length(SNR_dB));
MSE_eavesdropper_SVD_avg = zeros(1, length(SNR_dB));

for idx = 1:length(SNR_dB)
    disp(['Processing SNR = ' num2str(SNR_dB(idx)) ' dB']);
    % Temporary variables to accumulate MSE for each run
    MSE_legitimate = zeros(num_runs, 1);
    MSE_eavesdropper_with_CSI = zeros(num_runs, 1);
    MSE_eavesdropper_without_CSI = zeros(num_runs, 1);
    MSE_eavesdropper_noiseless = zeros(num_runs, 1);
    MSE_eavesdropper_SVD = zeros(num_runs, 1);

    parfor  run = 1:num_runs
        % Display progress
        if mod(run, 100) == 0
            disp(['  Run ' num2str(run) ' of ' num2str(num_runs)]);
        end
        
        % Channel coefficient generation
        [h, g] = generate_channel_coefficients(M, sigma_h, sigma_g);

        % Signal-to-Noise Ratio calculations
        SNR_linear = 10^(SNR_dB(idx)/10);
        c_squared = SNR_linear * sigma_y / M;
        c = sqrt(c_squared); % Ensure no imaginary parts if accidentally calculated

        % Precoding matrix optimization and generation
        d_without_CSI = optimize_precoding_matrix(M, h, g, c_squared, P, false);
        A_prime = create_reduced_row_echelon_form(h, M);
        A_without_CSI = create_precoding_matrix(A_prime, d_without_CSI);

        d_with_CSI = optimize_precoding_matrix(M, h, g, c_squared, P, true);
        A_with_CSI = create_precoding_matrix(A_prime, d_with_CSI);

        % Signal transmission
        gamma = generate_gamma(k, M);
        [x, x_without_CSI, x_with_CSI] = compute_transmit_signals(A_without_CSI, A_with_CSI, h, gamma, c, M, k);

        % Noise vectors generation and received signals computation
        gradcov = eye(k); % Covariance matrix for the input signals
        u = randn(M, k) * chol(gradcov);
        u = u / sqrt(mean(vecnorm(u).^2)); % Normalize signals to unit power
        s = compute_objective_function(gamma); % Compute the objective function
        [y, z_with_CSI, z_without_CSI] = compute_received_signals(x, h, g, x_without_CSI, x_with_CSI, sigma_y, sigma_z);

        % Calculation of MSE for various scenarios
        A_SVD = create_precoding_matrix_SVD(h, P);
        [D, S_with_CSI, S_without_CSI, S_noiseless, S_SVD] = calculate_MSE_new(M, c, h, g, A_with_CSI, A_without_CSI, A_SVD, sigma_y, sigma_z, k);
        
        % Store MSE results
        MSE_legitimate(run) = D;
        MSE_eavesdropper_with_CSI(run) = S_with_CSI;
        MSE_eavesdropper_without_CSI(run) = S_without_CSI;
        MSE_eavesdropper_noiseless(run) = S_noiseless;
        MSE_eavesdropper_SVD(run) = S_SVD;
    end

    % Average MSE over runs for each SNR
    MSE_legitimate_avg(idx) = mean(MSE_legitimate);
    MSE_eavesdropper_with_CSI_avg(idx) = mean(MSE_eavesdropper_with_CSI);
    MSE_eavesdropper_without_CSI_avg(idx) = mean(MSE_eavesdropper_without_CSI);
    MSE_eavesdropper_noiseless_avg(idx) = mean(MSE_eavesdropper_noiseless);
    MSE_eavesdropper_SVD_avg(idx) = mean(MSE_eavesdropper_SVD);
end

% Plotting results
figure;
hold on;
plot(SNR_dB, MSE_legitimate_avg, 'b-o', 'DisplayName', 'Legitimate Receiver');
plot(SNR_dB, MSE_eavesdropper_with_CSI_avg, 'r-s', 'DisplayName', 'Eavesdropper with CSI');
plot(SNR_dB, MSE_eavesdropper_without_CSI_avg, 'g-^', 'DisplayName', 'Eavesdropper without CSI');
plot(SNR_dB, MSE_eavesdropper_noiseless_avg, 'm-*', 'DisplayName', 'Eavesdropper Noise-less');
plot(SNR_dB, MSE_eavesdropper_SVD_avg, 'k-x', 'DisplayName', 'Eavesdropper SVD');
xlabel('SNR (dB)');
ylabel('Mean Squared Error (MSE)');
title('MSE vs. SNR for Legitimate Receiver and Eavesdropper');
legend show;
grid on;
hold off;

disp('Simulation completed.');

% All function definitions follow here...

function [h, g] = generate_channel_coefficients(M, sigma_h, sigma_g)
    h = raylrnd(sigma_h, M, 1);  % Rayleigh distribution for h
    h(1) = sqrt(pi - sqrt(4-pi))/sqrt(2);  % Specific value for h1
    g = raylrnd(sigma_g, M, 1);  % Rayleigh distribution for g
    h(2:end) = max(h(2:end), h(1));  % Ensuring h2,...,hM ≥ h1
end



function d = optimize_precoding_matrix(M, h, g, c_squared, P, with_CSI)
    if with_CSI
        % Optimization with CSI
        cvx_begin quiet
            variable d_squared(M-1)
            maximize(sum(d_squared .* ((h(1:M-1).^2 / h(M)^2) * g(M)^2 ...
                      - 2 * (h(1:M-1) / h(M)) .* g(1:M-1) * g(M) + g(1:M-1).^2)))
            subject to
                d_squared <= P - (c_squared ./ h(1:M-1).^2)
                sum(d_squared .* (h(1:M-1).^2 / h(M)^2)) <= P - (c_squared / h(M)^2)
        cvx_end
    else
        % Optimization without CSI
        cvx_begin quiet
            variable d_squared(M-1)
            maximize(sum(d_squared .* (2 * (h(1:M-1).^2 / h(M)^2) ...
                      - (pi/2) * (h(1:M-1) / h(M)) + 1)))
            subject to
                d_squared <= P - (c_squared ./ h(1:M-1).^2)
                sum(d_squared .* (h(1:M-1).^2 / h(M)^2)) <= P - (c_squared / h(M)^2)
        cvx_end
    end
    d = sqrt(d_squared); % Extracting the non-squared d values
    % disp(cvx_status);
end




function [x,x_without_CSI, x_with_CSI] = compute_transmit_signals(A_without_CSI, A_with_CSI, h, gamma, c, M, k)
    % Make sure the dimensions match: A should be (M-1) x M, V should be (M-1) x k
     % Generate noise vectors for each user with k columns
    V_without_CSI = randn(10, k);  % Make sure V is 10 x k to match A_without_CSI which is 9 x 10
    V_with_CSI = randn(10, k);
    noisless = randn(M, k);
    
    % Artificial noise vectors, append zero vector for the last user
    
    w_without_CSI = [A_without_CSI * V_without_CSI; zeros(1, k)];  % Appending a 1 x k zero row to make it M x k
    w_with_CSI = [A_with_CSI * V_with_CSI; zeros(1, k)];
    
    %gamma_scaled = c * diag(h.^(-1)) * gamma;    
    % Transmitted signals for each user
    % Use element-wise multiplication for h and u
    x_without_CSI = c * (h.^(-1)) .* gamma + w_without_CSI;
    x_with_CSI = c * (h.^(-1)) .* gamma + w_with_CSI;
    x = c * (h.^(-1)) .* gamma;
    
end





function [y, z_with_CSI, z_without_CSI] = compute_received_signals(x,h, g, x_without_CSI, x_with_CSI, sigma_y, sigma_z)
    % Noise addition
    n_y = sigma_y * randn;  % Scalar since it's added to a sum of signals
    n_z = sigma_z * randn;  % Scalar since it's added to a sum of signals
    
    % Compute received signals at the legitimate receiver and eavesdropper
    y = sum(h .* x) + n_y; % At legitimate receiver
    z_with_CSI = sum(g .* x_with_CSI) + n_z; % At eavesdropper with CSI
    z_without_CSI = sum(g .* x_without_CSI) + n_z; % At eavesdropper without CSI
end




function [MSE_legitimate, MSE_eavesdropper_with_CSI, MSE_eavesdropper_without_CSI] = calculate_MSE(s, y, z_with_CSI, z_without_CSI)

    % Ensure s, y, z_with_CSI, and z_without_CSI are column vectors and have the same length
    s = s(:); y = y(:); z_with_CSI = z_with_CSI(:); z_without_CSI = z_without_CSI(:);

    % Compute MSEs
    MSE_legitimate = mean((s - y).^2);  % Should be (y - s) if 'y' includes 's' and noise
    MSE_eavesdropper_with_CSI = mean((s - z_with_CSI).^2);  % Should be (z - s) for eavesdropper
    MSE_eavesdropper_without_CSI = mean((s - z_without_CSI).^2);  % Same for eavesdropper without CSI
end







function s = compute_objective_function(gamma)
    s = sum(gamma, 2); % assuming linear post-processing as stated in the paper
end

function A_prime = create_reduced_row_echelon_form(h, M)
    A_prime = zeros(M-1, M);
    for m = 1:M-1
        A_prime(m, m) = 1;
    end
    A_prime(:, M) = -h(1:M-1) / h(M);
end

function A = create_precoding_matrix(A_prime, d)
    A = diag(d) * A_prime;
end

function gamma = generate_gamma(k, M)
    % Assuming Σ is an identity matrix as per the paper's i.i.d assumption
    Sigma = eye(k);  
    gamma = mvnrnd(zeros(k, 1), Sigma, M)';  % M x k matrix
end

function [D, S_with_CSI, S_without_CSI,S_noiseless,S_SVD]= calculate_MSE_new(M, c, h, g, A_with_CSI, A_without_CSI,A_SVD, sigma_y, sigma_z, k)

    % MSE calculation for legitimate receiver
    D = M - (c^2 * M^2) / (c^2 * M + k * sigma_y);

    % Helper function to calculate S
    function S = calculate_S(A)
        numerator = (c^2 * (sum(g./h))^2);
        denominator = c^2 * sum((g.^2)./(h.^2)) + k * (norm(A)^2 + sigma_z);
        S = M - numerator / denominator;
    end

    % MSE calculation for eavesdropper with CSI
    S_with_CSI = calculate_S(A_with_CSI);

    % MSE calculation for eavesdropper without CSI
    S_without_CSI = calculate_S(A_without_CSI);

    S_noiseless=calculate_S(0);
    
    S_SVD = calculate_S(A_SVD);
end
function A_SVD = create_precoding_matrix_SVD(h, P)
    % Create a random vector v
    v = randn(size(h));
    
    % Project v onto h and subtract to get the component orthogonal to h
    v_orthogonal = v - (h' * v) / (h' * h) * h;
    
    % Scale the orthogonal component to satisfy the power constraint
    scaling_factor = sqrt(P / (v_orthogonal' * v_orthogonal));
    A_SVD = v_orthogonal * scaling_factor;
end



