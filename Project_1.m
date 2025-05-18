cd '/Users/anthonymerlin/Desktop/Math Modeling 1'
load('adjacency_matrices.mat')
% Print adjacency matrix A1
disp('Adjacency Matrix A1:');
disp(A1);

% Print adjacency matrix A2
disp('Adjacency Matrix A2:');
disp(A2);

% Print adjacency matrix A3
disp('Adjacency Matrix A3:');
disp(A3);
##################(Now let type out our graph Laplacian and get the eigenvalues and eigenvectors)#######################
L1 = [3 -1 -1 -1; -1 3 -1 -1; -1 -1 3 -1; -1 -1 -1 3]

L2 = [2 -1 -1 0; -1 2 -1 0; -1 -1 3 -1; 0 0 -1 1]

L3 = [1 -1 0 0; -1 1 0 0; 0 0 1 -1; 0 0 -1 1]

[V1, D1] = eig(L1);

[V2, D2] = eig(L2);

[V3, D3] = eig(L3);

#now based off of = this asside from what the eigevalue 0 does geometrically in a trasformation sense, we can see that
#the eigenvalue 0 is being shared in L1 and L3 as well as the number of eigenvectors associated with 0 eigenvalue tells us
#the structurenetw of the network we are working with for example in L1 the 0 eigenvalue had 1 eigenvector associating with a
#complete network, yet in L2 we can see there are two eigenvectors for the 0 eigenvalue showing 2 complete networks.
#######################################################(Problem 3.3)###############################################################

#so here we now want to generate a random network with N = 4 (N is just nodes) and p = to 0.1(and p is just a uniform probablity(that is the same of all the edges))

N1 = 4;
p1 = 0.1;
N2 = 4;
p2 = 0.9;

% Generate the first random network with N = 4 and p = 0.1
network1 = rand(N1, N1) < p1;
network1 = network1 - diag(diag(network1));

%description
#based on my drawing from the whiteboard I can say that network 1 is not really connected, with only node 1 and 4 having a relationship.

% Generate the second random network with N = 4 and p = 0.9
network2 = rand(N2, N2) < p2;
network2 = network2 - diag(diag(network2));

%decription
%based on my drawing from the whiteboard I can say that the network 2 is densly connected as we see a total opposite from network 1 with all nodes seeming to have relationships.


#############################################(Problem 4.3(1))###########################################################
% Here we have laoded our new initial_conditions vector

load('initial_conditions.mat');
disp(initial_conditions);

#############################################(2)##########################################################################
%now we are going to simulate the discrete time model for 20 steps for each of the inaction networks A1 and A2.

num_steps = 20;


% Set the values of ε for each network
epsilon_A1_consensus = 0.2;
epsilon_A1_no_consensus = 0.6;
epsilon_A2_consensus = 0.2;
epsilon_A2_no_consensus = 0.6;

% Simulate the discrete time model for A1 with consensus
x_A1_consensus = zeros(4, num_steps + 1);
x_A1_consensus(:, 1) = initial_conditions;
for k = 1:num_steps
    x_A1_consensus(:, k + 1) = (eye(4) - epsilon_A1_consensus * L1) * x_A1_consensus(:, k);
end

% Simulate the discrete time model for A1 without consensus
x_A1_no_consensus = zeros(4, num_steps + 1);
x_A1_no_consensus(:, 1) = initial_conditions;
for k = 1:num_steps
    x_A1_no_consensus(:, k + 1) = (eye(4) - epsilon_A1_no_consensus * L1) * x_A1_no_consensus(:, k);
end

% Simulate the discrete time model for A2 with consensus
x_A2_consensus = zeros(4, num_steps + 1);
x_A2_consensus(:, 1) = initial_conditions;
for k = 1:num_steps
    x_A2_consensus(:, k + 1) = (eye(4) - epsilon_A2_consensus * L2) * x_A2_consensus(:, k);
end

% Simulate the discrete time model for A2 without consensus
x_A2_no_consensus = zeros(4, num_steps + 1);
x_A2_no_consensus(:, 1) = initial_conditions;
for k = 1:num_steps
    x_A2_no_consensus(:, k + 1) = (eye(4) - epsilon_A2_no_consensus * L2) * x_A2_no_consensus(:, k);
end


% Plot the results
figure;
subplot(2, 2, 1);
plot(0:num_steps, x_A1_consensus');
title('A1 with consensus');
xlabel('Time steps');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

subplot(2, 2, 2);
plot(0:num_steps, x_A1_no_consensus');
title('A1 without consensus');
xlabel('Time steps');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

subplot(2, 2, 3);
plot(0:num_steps, x_A2_consensus');
title('A2 with consensus');
xlabel('Time steps');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

subplot(2, 2, 4);
plot(0:num_steps, x_A2_no_consensus');
title('A2 without consensus');
xlabel('Time steps');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

#############################################(3)##########################################################################

%Now we are going to take a look at the effects of epilson with A3

% Set the values of ε for A3
epsilon_A3_consensus = 0.5;
epsilon_A3_no_consensus = 1.2;

% Simulate the discrete time model for A3 with consensus
x_A3_consensus = zeros(4, num_steps + 1);
x_A3_consensus(:, 1) = initial_conditions;
for k = 1:num_steps
    x_A3_consensus(:, k + 1) = (eye(4) - epsilon_A3_consensus * L3) * x_A3_consensus(:, k);
end

% Simulate the discrete time model for A3 without consensus
x_A3_no_consensus = zeros(4, num_steps + 1);
x_A3_no_consensus(:, 1) = initial_conditions;
for k = 1:num_steps
    x_A3_no_consensus(:, k + 1) = (eye(4) - epsilon_A3_no_consensus * L3) * x_A3_no_consensus(:, k);
end

% Plot the results for A3 with consensus
figure;
subplot(1, 2, 1);
plot(0:num_steps, x_A3_consensus');
title('A3 with consensus');
xlabel('Time steps');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

% Plot the results for A3 without consensus
subplot(1, 2, 2);
plot(0:num_steps, x_A3_no_consensus');
title('A3 without consensus');
xlabel('Time steps');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

#############################################(4)##########################################################################
#for this part we want to impletement the Runge Kutta method from module 11 to then simulate the
# continuous time model for 20 units of time with a timp step of 0.5. simulate the system for A1, A2, and A3


% Set simulation parameters
t0 = 0;
tf = 20;
h = 0.5;
n = (tf - t0) / h;

% Define the continuous time model function
function dx = continuous_model(t, x, L)
    dx = -L * x;
end

% Define the Runge-Kutta function
function [t, y] = RK(f, t0, y0, n, h)
    t = zeros(1, n + 1);
    y = zeros(length(y0), n + 1);

    t(1) = t0;
    y(:, 1) = y0;

    for k = 1:n
        t(k + 1) = t(k) + h;

        k1 = f(t(k), y(:, k));
        k2 = f(t(k) + h/2, y(:, k) + h/2 * k1);
        k3 = f(t(k) + h/2, y(:, k) + h/2 * k2);
        k4 = f(t(k) + h, y(:, k) + h * k3);

        y(:, k + 1) = y(:, k) + h/6 * (k1 + 2*k2 + 2*k3 + k4);
    end
end

% Simulate the continuous time model for A1, A2, and A3 using Runge-Kutta method
[t_A1, x_A1_continuous] = RK(@(t, x) continuous_model(t, x, L1), t0, initial_conditions, n, h);
[t_A2, x_A2_continuous] = RK(@(t, x) continuous_model(t, x, L2), t0, initial_conditions, n, h);
[t_A3, x_A3_continuous] = RK(@(t, x) continuous_model(t, x, L3), t0, initial_conditions, n, h);

% Plot the results for the continuous time model
figure;
subplot(3, 1, 1);
plot(t_A1, x_A1_continuous');
title('A1 - Continuous Time Model');
xlabel('Time');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

subplot(3, 1, 2);
plot(t_A2, x_A2_continuous');
title('A2 - Continuous Time Model');
xlabel('Time');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

subplot(3, 1, 3);
plot(t_A3, x_A3_continuous');
title('A3 - Continuous Time Model');
xlabel('Time');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

% Compare the results between continuous and discrete time models
figure;
subplot(3, 2, 1);
plot(0:num_steps, x_A1_consensus');
title('A1 - Discrete Time Model (Consensus)');
xlabel('Time steps');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

subplot(3, 2, 2);
plot(t_A1, x_A1_continuous');
title('A1 - Continuous Time Model');
xlabel('Time');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

subplot(3, 2, 3);
plot(0:num_steps, x_A2_consensus');
title('A2 - Discrete Time Model (Consensus)');
xlabel('Time steps');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

subplot(3, 2, 4);
plot(t_A2, x_A2_continuous');
title('A2 - Continuous Time Model');
xlabel('Time');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

subplot(3, 2, 5);
plot(0:num_steps, x_A3_consensus');
title('A3 - Discrete Time Model (Consensus)');
xlabel('Time steps');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

subplot(3, 2, 6);
plot(t_A3, x_A3_continuous');
title('A3 - Continuous Time Model');
xlabel('Time');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');


#############################################(Problem 4.4(1/2))###########################################################
#we want to preform a discrete time algoritm where the interaction network is an indepedent realization of a random
#network that switches at each step.
#also
#for this question we are being asked to simulate the algorithm with a switching network using parameter values
#N = 4 and ε = 0.4 and p = {0.1 0.9} for 20 time steps
% Set simulation parameters
num_steps = 20;
N = 4;
epsilon = 0.4;

% Initialize matrices to store results
x_switching_p1 = zeros(N, num_steps + 1);
x_switching_p2 = zeros(N, num_steps + 1);

x_switching_p1(:, 1) = initial_conditions;
x_switching_p2(:, 1) = initial_conditions;

% Compute the graph Laplacian for network1 (p = 0.1)
L_switching_p1 = diag(sum(network1, 2)) - network1;

% Compute the graph Laplacian for network2 (p = 0.9)
L_switching_p2 = diag(sum(network2, 2)) - network2;

% Simulate the discrete time model with switching networks for p = 0.1
for k = 1:num_steps
    % Update the state values using network1
    x_switching_p1(:, k + 1) = (eye(N) - epsilon * L_switching_p1) * x_switching_p1(:, k);
end

% Simulate the discrete time model with switching networks for p = 0.9
for k = 1:num_steps
    % Update the state values using network2
    x_switching_p2(:, k + 1) = (eye(N) - epsilon * L_switching_p2) * x_switching_p2(:, k);
end

% Plot the results for the switching networks
figure;
subplot(2, 1, 1);
plot(0:num_steps, x_switching_p1');
title('Switching Network with p = 0.1');
xlabel('Time steps');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');

subplot(2, 1, 2);
plot(0:num_steps, x_switching_p2');
title('Switching Network with p = 0.9');
xlabel('Time steps');
ylabel('State values');
legend('x1', 'x2', 'x3', 'x4');
#############################################(Problem 4.4(3))###########################################################

% Set simulation parameters
num_steps = 20;
N = 4;
epsilon = 0.4;

% Initialize matrices to store results
x_switching_p1 = zeros(N, num_steps + 1);
x_switching_p2 = zeros(N, num_steps + 1);
x_static_A2 = zeros(N, num_steps + 1);

x_switching_p1(:, 1) = initial_conditions;
x_switching_p2(:, 1) = initial_conditions;
x_static_A2(:, 1) = initial_conditions;

% Compute the graph Laplacian for network1 (p = 0.1)
L_switching_p1 = diag(sum(network1, 2)) - network1;

% Compute the graph Laplacian for network2 (p = 0.9)
L_switching_p2 = diag(sum(network2, 2)) - network2;

% Compute the graph Laplacian for the static network A2
L_static_A2 = diag(sum(A2, 2)) - A2;

% Initialize matrices to store disagreement values
disagreement_p1 = zeros(1, num_steps + 1);
disagreement_p2 = zeros(1, num_steps + 1);
disagreement_A2 = zeros(1, num_steps + 1);

% Simulate the discrete time model with switching networks for p = 0.1
for k = 1:num_steps
    % Update the state values using network1
    x_switching_p1(:, k + 1) = (eye(N) - epsilon * L_switching_p1) * x_switching_p1(:, k);

    % Compute the consensus state for switching network (p = 0.1)
    x_bar = (1/N) * sum(x_switching_p1(:, 1));
    consensus_state = x_bar * ones(N, 1);

    % Compute the disagreement vector
    disagreement_p1(k + 1) = norm(x_switching_p1(:, k + 1) - consensus_state);
end

% Simulate the discrete time model with switching networks for p = 0.9
for k = 1:num_steps
    % Update the state values using network2
    x_switching_p2(:, k + 1) = (eye(N) - epsilon * L_switching_p2) * x_switching_p2(:, k);

    % Compute the consensus state for switching network (p = 0.9)
    x_bar = (1/N) * sum(x_switching_p2(:, 1));
    consensus_state = x_bar * ones(N, 1);

    % Compute the disagreement vector
    disagreement_p2(k + 1) = norm(x_switching_p2(:, k + 1) - consensus_state);
end

% Simulate the discrete time model with the static network A2
for k = 1:num_steps
    % Update the state values using the static network A2
    x_static_A2(:, k + 1) = (eye(N) - epsilon * L_static_A2) * x_static_A2(:, k);

    % Compute the consensus state for static network A2
    x_bar = (1/N) * sum(x_static_A2(:, 1));
    consensus_state = x_bar * ones(N, 1);

    % Compute the disagreement vector
    disagreement_A2(k + 1) = norm(x_static_A2(:, k + 1) - consensus_state);
end

% Plot the disagreement for each simulation
figure;
plot(0:num_steps, disagreement_p1, 'r', 'LineWidth', 0.5);
hold on;
plot(0:num_steps, disagreement_p2, 'b', 'LineWidth', 0.5);
plot(0:num_steps, disagreement_A2, 'g', 'LineWidth', 0.5);
hold off;
xlabel('Time steps');
ylabel('Disagreement');
title('Disagreement for Switching Networks and Static Network');
legend('Switching Network (p = 0.1)', 'Switching Network (p = 0.9)', 'Static Network (A2)');



