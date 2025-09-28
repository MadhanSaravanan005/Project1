clear;
clc;

% Load metadata CSV
metadata = readtable('HAM10000_metadata.csv');

% Extract image IDs and corresponding labels
image_ids = metadata.image_id;
labels = metadata.dx;

% Convert categorical labels to numeric labels if necessary
[unique_labels, ~, label_indices] = unique(labels);
num_classes = length(unique_labels);
numeric_labels = label_indices;

% Initialize empty arrays for storing image data and labels
num_images = length(image_ids);
img_size = [64, 64];  % Define the target image size
ZTrain = zeros(img_size(1), img_size(2), 1, num_images, 'single');  % For grayscale images
label_array = zeros(num_images, 1, 'uint8');

% Create a mapping from image IDs to labels
id_to_label = containers.Map(image_ids, numeric_labels);

% Parallel processing setup (if you have Parallel Computing Toolbox)
parfor i = 1:num_images
    image_id = image_ids{i};
    image_file = fullfile('HAM10000_images_part_1', strcat(image_id, '.jpg')); % First folder
    if ~exist(image_file, 'file')
        image_file = fullfile('HAM10000_images_part_2', strcat(image_id, '.jpg')); % Try second folder
    end
    if exist(image_file, 'file')
        % Read image and preprocess (e.g., resize)
        img = imread(image_file);
        img = imresize(img, img_size); % Resize to target size
        
        % Apply Quantum Pre-processing Filter (QPF)
        img_qpf = applyQPF(img, img_size);
        ZTrain(:,:,:,i) = img_qpf;
        
        % Assign label
        label_array(i) = id_to_label(image_id);
    else
        warning('Image file %s does not exist.', image_file);
    end
end

% Save preprocessed images and labels
save('preprocessed_HAM10000_QPF.mat', 'ZTrain', 'label_array');

% Function to apply Quantum Pre-processing Filter (QPF)
function processed_img = applyQPF(img, img_size)
    % Convert the image to grayscale if needed
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % Resize image to ensure consistency
    img_resized = imresize(img, img_size); % Adjust size as needed
    
    % Normalize image pixel values
    img_normalized = double(img_resized) / 255;
    
    % Prepare an output matrix for the QPF results
    [rows, cols] = size(img_normalized);
    processed_img = zeros(rows, cols, 'single'); % Create grayscale image
    
    % Apply QPF using the `circuit_two_cnots` function
    % Vectorize block processing for efficiency
    block_size = 2;
    num_blocks_x = floor(rows / block_size);
    num_blocks_y = floor(cols / block_size);
    
    % Reshape img_normalized into blocks and process
    for bx = 1:num_blocks_x
        for by = 1:num_blocks_y
            % Extract a 2x2 block from the image
            block = img_normalized((bx-1)*block_size+1:bx*block_size, ...
                                   (by-1)*block_size+1:by*block_size);
            
            % Apply `circuit_two_cnots` to the block
            processed_block = circuit_two_cnots(block(:));
            
            % Assign processed_block to the grayscale image
            processed_img((bx-1)*block_size+1:bx*block_size, ...
                          (by-1)*block_size+1:by*block_size) = processed_block;
        end
    end
end

% Function to simulate a quantum circuit with two CNOT gates
function output = circuit_two_cnots(input)
    % Ensure input is a column vector
    if numel(input) ~= 4
        error('Input must be a vector with 4 elements.');
    end
    
    % Example placeholder quantum circuit simulation
    % Replace this with actual quantum computing logic if available
    output = sum(input) / numel(input);
end
