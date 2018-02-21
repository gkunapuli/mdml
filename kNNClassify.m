% KNNCLASSIFY  Performs (possibly multi-class) classification on test data
% given labeled training data, and the number of nearest neighbors to
% consider using the specified Mahalanobis distance.
%
% YTEST = kNNClassify(L, Xtrain, ytrain, Xtest, K)
%    L                  Mahalanobis projection matrix
%    Xtrain, ytrain     Labeled training data
%    ytest              Data to be classified
%    K                  Number of nearest neighbors to consider
%
%  version 3.7
%  Gautam Kunapuli (gkunapuli@gmail.com)
%  January 17, 2012
%
% This program comes with ABSOLUTELY NO WARRANTY; See the GNU General Public
% License for more details. This is free software, and you are welcome to 
% modify or redistribute it.

function ytest = kNNClassify(L, Xtrain, ytrain, Xtest, K)

% Compute the Mahalanobis distance between the train and test points
D = mahalanobisDistance(L, Xtrain, Xtest);

% Sort the distances
[weights, neighbors] = sort(D);

% Keep only the indices of the K nearest neighbors (the first one is the
% data point itself and should be skipped)
neighbors = neighbors(2:K+1, :);
weights = 1 ./ weights(2:K+1, :);

% Determine the class of each column using weights
neighborLabels = ytrain(neighbors);

nTest = size(Xtest, 1);
nClasses = length(unique(ytrain));

if K == 1
    ytest = neighborLabels;
else
    classWeights = zeros(nClasses, nTest);

    for i = 1:nTest
        uniqueLabels = unique(neighborLabels(:, i));
        for j = uniqueLabels'
            classWeights(j, i) = sum(weights(neighborLabels(:, i) == j, i));
        end
    end

    [~, ytest] = max(classWeights);
    ytest = ytest';
end