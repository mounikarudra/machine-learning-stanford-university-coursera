function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = [ 0.01,0.03,0.1,0.3,1,3,10,30];
sigma =  [0.01,0.03,0.1,0.3,1,3,10,30];
cost = [];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
##for i=C
##  for j=sigma
##  model= svmTrain(X, y, i, @(x1, x2) gaussianKernel(x1, x2, j));
##  predictions = svmPredict(model, Xval);
##  cost  = [cost;mean(double(predictions ~= yval)) i j];  
##  endfor
##endfor
###min_cost = cost(find(cost == min(cost(:,1))),:);
##min_cost = cost(find(cost(:,1) == min(cost(:,1))),:);
##C = min_cost(1,1);
##sigma = min_cost(1,2);

errorRow=0;
for i=C
  for j=sigma
  errorRow = errorRow + 1;  
  model= svmTrain(X, y, i, @(x1, x2) gaussianKernel(x1, x2, j));
  predictions = svmPredict(model, Xval);
  cost(errorRow,:)  = [i,j,mean(double(predictions ~= yval))];  
  endfor
endfor
#min_cost = cost(find(cost == min(cost(:,1))),:);
#min_cost = cost(find(cost(:,1) == min(cost(:,1))),:);
min_cost=sortrows(cost,3);
C = min_cost(1,1);
sigma = min_cost(1,2);





% =========================================================================

end
