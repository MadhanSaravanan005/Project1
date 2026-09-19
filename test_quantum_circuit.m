% TEST_QUANTUM_CIRCUIT Automated test script for circuit_two_cnots in MATLAB

fprintf('Running tests for circuit_two_cnots...\n');

% Test 1: Ground state input [0, 0, 0, 0]
phi_zero = [0; 0; 0; 0];
expval_zero = circuit_two_cnots(phi_zero);
assert(isequal(size(expval_zero), [1, 4]), 'Output shape must be 1x4');
assert(all(abs(expval_zero - 1.0) < 1e-6), 'For ground state input, expectation values should all be 1.0');
fprintf('  [PASS] Test 1: Zero rotation gives <Z> = [1, 1, 1, 1]\n');

% Test 2: Full pi rotation [1, 1, 1, 1]
phi_one = [1; 1; 1; 1];
expval_one = circuit_two_cnots(phi_one);
assert(isequal(size(expval_one), [1, 4]), 'Output shape must be 1x4');
assert(all(abs(expval_one - (-1.0)) < 1e-6), 'For pi rotation input, expectation values should all be -1.0');
fprintf('  [PASS] Test 2: Pi rotation gives <Z> = [-1, -1, -1, -1]\n');

% Test 3: Output range [-1, 1] for arbitrary random inputs
rng(42);
for t = 1:20
    phi_rand = rand(4, 1);
    expval_rand = circuit_two_cnots(phi_rand);
    assert(all(expval_rand >= -1.0 - 1e-6 & expval_rand <= 1.0 + 1e-6), ...
        'Expectation values must be within [-1, 1]');
end
fprintf('  [PASS] Test 3: All expectation values stay bounded in [-1, 1]\n');

% Test 4: Dimension validation
try
    circuit_two_cnots([0.1; 0.2]);
    error('Should have failed for 2-element input');
catch
    fprintf('  [PASS] Test 4: Invalid vector dimension throws error as expected\n');
end

fprintf('All MATLAB quantum circuit tests PASSED successfully.\n');
