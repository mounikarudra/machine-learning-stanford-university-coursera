function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%input_layer_size  = 400;  % 20x20 Input Images of Digits
%hidden_layer_size = 25;   % 25 hidden units
%num_labels = 10; 
%theta1 = 25 x 401
%theta2 = 10 x 26
X = [ones(size(X, 1), 1) X];
a1 = sigmoid(X * Theta1');% 5000 x 25
a1 = [ones(size(a1, 1), 1) a1];% 5000 x 26
a2 = sigmoid(Theta2 * a1');% 10 x 5000
[max, index] = max(a2); % 5000x10
p = index';






% =========================================================================


end
