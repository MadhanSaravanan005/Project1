function expval_all = circuit_two_cnots( phi )
% CIRCUIT_TWO_CNOTS Simulates a 4-qubit quantum circuit with parameter rotations
% and two CNOT entangling gates, outputting Pauli-Z expectation values.
%
% Input:
%   phi - 4-element numeric array representing normalized pixel values [phi1, phi2, phi3, phi4]
% Output:
%   expval_all - 1x4 array of Pauli-Z expectation values in range [-1, 1]

phi = phi(:);
if numel(phi) ~= 4
    error('Input phi must have exactly 4 elements.');
end

% Index masks corresponding to qubit states where qi = 0 (1-based indexing)
index1 = 1:8;                      % Qubit 1 = 0
index2 = [1:4, 9:12];              % Qubit 2 = 0
index3 = [1, 2, 5, 6, 9, 10, 13, 14]; % Qubit 3 = 0
index4 = 1:2:15;                   % Qubit 4 = 0

% Parameterized rotation angles
phi_rad = pi * phi;

% Single-qubit RY rotation matrices
RY1 = [cos(phi_rad(1)/2), -sin(phi_rad(1)/2); sin(phi_rad(1)/2), cos(phi_rad(1)/2)];
RY2 = [cos(phi_rad(2)/2), -sin(phi_rad(2)/2); sin(phi_rad(2)/2), cos(phi_rad(2)/2)];
RY3 = [cos(phi_rad(3)/2), -sin(phi_rad(3)/2); sin(phi_rad(3)/2), cos(phi_rad(3)/2)];
RY4 = [cos(phi_rad(4)/2), -sin(phi_rad(4)/2); sin(phi_rad(4)/2), cos(phi_rad(4)/2)];

% Initial ground state |0000>
x = zeros(16, 1);
x(1) = 1;

% Separable state preparation operator: T1 = RY1 (x) RY2 (x) RY3 (x) RY4
T1 = kron(RY1, kron(RY2, kron(RY3, RY4)));

% Entangling operator T2 implementing two CNOT gates: CNOT(q1 -> q2) and CNOT(q0 -> q3)
T2 = [
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0
];

% Statevector after circuit execution
y = T2 * (T1 * x);

% Probability of each qubit measuring 0
p0 = [
    sum(abs(y(index1)).^2), ...
    sum(abs(y(index2)).^2), ...
    sum(abs(y(index3)).^2), ...
    sum(abs(y(index4)).^2)
];

% Pauli-Z expectation values <Z_i> = P(0) - P(1) = 2*P(0) - 1
expval_all = 2 * p0 - 1;
end