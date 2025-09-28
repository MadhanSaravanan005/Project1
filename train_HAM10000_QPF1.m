% train_HAM10000_RMSProp.m

clear;
clc;

% Load preprocessed data
load('preprocessed_HAM10000_QPF.mat', 'ZTrain', 'label_array');

% Normalize images
ZTrain = double(ZTrain) / 255;

% Define number of images and classes
num_images = size(ZTrain, 4);
num_classes = length(unique(label_array));

% Manually split data for training and validation
rng(42);  % For reproducibility
indices = randperm(num_images);
split_point = floor(0.8 * num_images);  % 80% training, 20% validation

train_indices = indices(1:split_point);
val_indices = indices(split_point+1:end);

XTrain = ZTrain(:,:,:,train_indices);
YTrain = categorical(label_array(train_indices));
XVal = ZTrain(:,:,:,val_indices);
YVal = categorical(label_array(val_indices));

% Define a custom CNN architecture
layers = [
    imageInputLayer([64 64 1], 'Normalization', 'none') % No normalization here, since normalization is applied earlier
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

% Training options
options = trainingOptions('rmsprop', ...  % RMSProp
    'MaxEpochs', 15, ... % Adjust based on model performance
    'InitialLearnRate', 1e-4, ...
    'MiniBatchSize', 64, ...
    'ValidationData', {XVal, YVal}, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch'); % Shuffle data every epoch for better generalization

% Train the network
[net, info] = trainNetwork(XTrain, YTrain, layers, options);

% Evaluate accuracy
YPred = classify(net, XVal);
accuracy = sum(YPred == YVal) / numel(YVal);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

% Save trained network and training information
save('trained_network_HAM10000_RMSProp.mat', 'net', 'info', 'accuracy');
