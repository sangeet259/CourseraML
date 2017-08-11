function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%


		% Let's calculate the cost  function
		% cost eill be on the error in prediction of ratings by an user 
		%  ie user's prdeicted rating - actual rating ka square 
		% thersore the loop will be on the number of users

% -----------------Trying out a vectorized implementation of J----------------------------
%  first calculate as it is 
%  then just do a hadarmard product and sum over the 2 dimensions

temp_matrix=X*Theta';

% now calculate cost 
cost_matrix=(temp_matrix-Y).^2;
cost_matrix=cost_matrix.*R;
J=(1/2)*sum(sum(cost_matrix))+(lambda/2)*( (sum(sum(X.^2))) + ((sum(sum(Theta.^2)))) );





% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

X_grad=(-1).*((Y-temp_matrix).*R)*Theta + lambda*(X);
Theta_grad=(-1).*((Y-temp_matrix).*R)'*X + lambda*(Theta);














% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
