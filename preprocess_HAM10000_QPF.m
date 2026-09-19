clear;
clc;

% Load metadata CSV
metadata_file = 'HAM10000_metadata.csv';
if ~isfile(metadata_file)
    error('Metadata file %s not found in repository root.', metadata_file);
end
metadata = readtable(metadata_file);

% Extract image IDs and corresponding diagnosis labels
image_ids = metadata.image_id;
labels = metadata.dx;

% Convert categorical labels to numeric labels (1 to num_classes)
[unique_labels, ~, label_indices] = unique(labels);
num_classes = length(unique_labels);
numeric_labels = uint8(label_indices);

% Verify availability of raw image folders
folder1 = 'HAM10000_images_part_1';
folder2 = 'HAM10000_images_part_2';
if ~isfolder(folder1) && ~isfolder(folder2)
    fprintf('=== HAM10000 Raw Images Not Found ===\n');
    fprintf('The raw image directories ("%s" and "%s") were not found.\n', folder1, folder2);
    fprintf('To preprocess raw images from scratch, download the HAM10000 dataset parts\n');
    fprintf('from Harvard Dataverse / Kaggle and extract them into the repository root.\n');
    fprintf('Note: The preprocessed dataset is already available in "preprocessed_HAM10000_QPF.mat".\n');
    return;
end

% Initialize empty arrays for storing image data and labels
num_images = length(image_ids);
img_size = [64, 64];  % Target image size
ZTrain = zeros(img_size(1), img_size(2), 1, num_images, 'single');
label_array = zeros(num_images, 1, 'uint8');

% Create mapping from image ID to label
id_to_label = containers.Map(image_ids, numeric_labels);

fprintf('Starting Quantum Pre-processing Filter (QPF) on %d images...\n', num_images);
processed_mask = false(num_images, 1);

% Parallel processing (uses Parallel Computing Toolbox if available, otherwise serial)
parfor i = 1:num_images
    image_id = image_ids{i};
    image_file = fullfile(folder1, strcat(image_id, '.jpg'));
    if ~isfile(image_file)
        image_file = fullfile(folder2, strcat(image_id, '.jpg'));
    end

    if isfile(image_file)
        % Read image and resize to target dimension
        img = imread(image_file);
        img = imresize(img, img_size);

        % Apply Quantum Pre-processing Filter (QPF) using 4-qubit circuit
        img_qpf = applyQPF(img, img_size);
        ZTrain(:,:,:,i) = img_qpf;
        label_array(i) = id_to_label(image_id);
        processed_mask(i) = true;
    else
        warning('Image file %s does not exist.', image_file);
    end
end

processed_count = sum(processed_mask);
if processed_count > 0
    % Filter down to successfully processed images if any were missing
    if processed_count < num_images
        ZTrain = ZTrain(:,:,:,processed_mask);
        label_array = label_array(processed_mask);
    end
    save('preprocessed_HAM10000_QPF.mat', 'ZTrain', 'label_array', '-v7.3');
    fprintf('Successfully processed %d images and saved to "preprocessed_HAM10000_QPF.mat".\n', processed_count);
else
    warning('No images were found or processed. Existing "preprocessed_HAM10000_QPF.mat" preserved.');
end

% Function to apply Quantum Pre-processing Filter (QPF)
function processed_img = applyQPF(img, img_size)
    % Convert to grayscale if 3-channel RGB
    if size(img, 3) == 3
        img = rgb2gray(img);
    end

    % Ensure consistent size
    if ~isequal(size(img, [1, 2]), img_size)
        img = imresize(img, img_size);
    end

    % Normalize pixel values to [0, 1] range for quantum rotation angles
    img_normalized = double(img) / 255.0;

    [rows, cols] = size(img_normalized);
    processed_img = zeros(rows, cols, 'single');

    block_size = 2;
    num_blocks_x = floor(rows / block_size);
    num_blocks_y = floor(cols / block_size);

    for bx = 1:num_blocks_x
        for by = 1:num_blocks_y
            row_idx = (bx-1)*block_size + 1 : bx*block_size;
            col_idx = (by-1)*block_size + 1 : by*block_size;

            % Extract 2x2 block
            block = img_normalized(row_idx, col_idx);

            % Simulate 4-qubit circuit with 2 CNOT gates via circuit_two_cnots.m
            expval = circuit_two_cnots(block(:));

            % Map expectation values in [-1, 1] to image intensity [0, 1]
            % <Z> = 2*P(0) - 1 => P(0) = (<Z> + 1) / 2
            norm_block = (reshape(expval, [block_size, block_size]) + 1.0) / 2.0;

            processed_img(row_idx, col_idx) = single(norm_block);
        end
    end
end
