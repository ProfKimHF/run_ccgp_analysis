function [neural_X_train_new, neural_X_test_new] = Geometric_random_model(neural_X_train,label_train,neural_X_test,label_test)

data = [neural_X_train; neural_X_test];
label = [label_train; label_test];

[all_comb,~,data_label] = unique(label,'rows');
numConditions = size(all_comb,1); % Number of conditions

numTrialsPerCondition = sum(data_label == 1); % Number of trials per condition
dimensionality = size(data,2); % Dimensionality of the data

% Calculate the Euclidean distance (squared) for each point from the center of each condition
distances = zeros(numConditions, numTrialsPerCondition);

for condition = 1:numConditions

    temp_data = data(data_label == condition,:);

    % Calculate the center of each condition
    centers(condition,:) = mean(temp_data,1);

    % Calculate the squared Euclidean distance for each point
    distances(condition, :) = sum((temp_data - centers(condition, :)).^2, 2);
end

% Calculate the signal variance 
origVar = trace(cov(centers));

% isotropic Gaussian shift: 
sigma_shift = 1;
shifts = sigma_shift * randn(size(centers));
newCentroids = centers + shifts;

% Calculate the signal variance after shift
newVar = trace(cov(newCentroids));

% rescale
scaleFactor = sqrt(origVar / newVar);
rescaledCentroids = newCentroids * scaleFactor;

% apply new center to all points
new_data = data;
for condition = 1:numConditions
    delta = rescaledCentroids(condition,:) - centers(condition,:);
    new_data(data_label == condition,:) = new_data(data_label == condition,:) + delta;
end

neural_X_train_new = new_data(1:size(neural_X_train,1),:);
neural_X_test_new = new_data(size(neural_X_train,1)+1:end,:);

% Plot the original and changed points for each condition
% figure;
% for condition = 1:numConditions
%     scatter3(data(data_label == condition, 1), data(data_label == condition, 2), data(data_label == condition, 3), 'Marker', '.', 'DisplayName', 'Original Points');
%     hold on
% end
% 
% figure;
% for condition = 1:numConditions
%     scatter3(new_data(data_label == condition, 1), new_data(data_label == condition, 2), new_data(data_label == condition, 3), 'Marker', '.', 'DisplayName', 'Original Points');
%     hold on
% end

end