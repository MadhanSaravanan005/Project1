% train_HAM10000_QPF.m
% Trains a custom Convolutional Neural Network on quantum-preprocessed HAM10000 images.

clearvars -except optimizer_choice;
clc;

% Select optimizer ('rmsprop' or 'sgdm', default: 'rmsprop')
if ~exist('optimizer_choice', 'var')
    optimizer_choice = 'rmsprop';
end

dataset_file = 'preprocessed_HAM10000_QPF.mat';
if ~isfile(dataset_file)
    error('Preprocessed data file "%s" not found. Please run preprocess_HAM10000_QPF.m or ensure the .mat file is in the workspace.', dataset_file);
end

fprintf('Loading preprocessed quantum-filtered data from %s...\n', dataset_file);
load(dataset_file, 'ZTrain', 'label_array');

% Safe normalization check: only divide by 255 if data is not already normalized in [0, 1]
if max(ZTrain(:)) > 1.0
    fprintf('Normalizing pixel values to [0, 1]...\n');
    ZTrain = double(ZTrain) / 255.0;
else
    ZTrain = double(ZTrain);
end

% Dataset parameters
num_images = size(ZTrain, 4);
unique_classes = unique(label_array);
num_classes = length(unique_classes);
fprintf('Total images: %d across %d diagnostic classes.\n', num_images, num_classes);

% Split data (80% training, 20% validation) with fixed seed for reproducibility
rng(42);
indices = randperm(num_images);
split_point = floor(0.8 * num_images);

train_indices = indices(1:split_point);
val_indices = indices(split_point+1:end);

XTrain = ZTrain(:, :, :, train_indices);
YTrain = categorical(label_array(train_indices));
XVal = ZTrain(:, :, :, val_indices);
YVal = categorical(label_array(val_indices));

% Custom CNN Architecture (Input -> 3 Conv/ReLU/Pool Blocks -> FullyConnected -> Softmax)
layers = [
    imageInputLayer([64 64 1], 'Normalization', 'none')
    convolution2dLayer(3, 16, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer
    fullyConnectedLayer(num_classes)
    softmaxLayer
    classificationLayer
];

% Configure training options
if strcmpi(optimizer_choice, 'sgdm')
    solver = 'sgdm';
    save_file = 'trained_network_HAM10000_SGDM.mat';
else
    solver = 'rmsprop';
    save_file = 'trained_network_HAM10000_RMSProp.mat';
end

fprintf('Training CNN using solver: %s (MaxEpochs: 15, MiniBatchSize: 64, LearnRate: 1e-4)...\n', solver);

options = trainingOptions(solver, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-4, ...
    'MiniBatchSize', 64, ...
    'ValidationData', {XVal, YVal}, ...
    'Verbose', false, ...
    'Plots', 'none', ...
    'Shuffle', 'every-epoch');

% Train network
[net, info] = trainNetwork(XTrain, YTrain, layers, options);

% Evaluate validation accuracy
YPred = classify(net, XVal);
accuracy = sum(YPred == YVal) / numel(YVal);
fprintf('Validation Accuracy (%s): %.2f%%\n', solver, accuracy * 100);

% Save model artifact
save(save_file, 'net', 'info', 'accuracy');
fprintf('Saved trained model and training metrics to "%s".\n', save_file);
