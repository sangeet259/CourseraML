function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1,K
%
% Note: You can use a for-loop over the examples to compute this.
%


% idx is a m X 1 vector where idx(i) is min(x(i) - centroids(j))
n = size(X,2);

for i=1:size(X,1),

	% focus on i'th training example
	% calculate the distance of x(i) from all centroids

	sq=(X(i,:)-centroids).^2;
	dis=zeros(size(centroids,1),1);
	for j=1:size(X,2),
		dis=dis+sq(:,j);
	end;
	shortest_dist=min(dis);
	idx_pos=find(dis==shortest_dist);

			
	if size(i)==size(idx_pos)
		% fprintf('the sizes seem same \n');
		idx(i)=idx_pos;
	else
		% fprintf('size  mismatch :::::: \n');
		% fprintf('The size of i :: \t');
		% disp(size(i));
		% fprintf('\n');
		% fprintf('The size of thr find result :: \t');
		% disp(size(idx_pos));
		% fprintf('\n');
		idx(i)=idx_pos(1,1);

	end
	% distances=zeros(K,1);


	% for k=1:K,
	% 	% now let's focus on the i'th training example and k'th centroid
	% 	% calculate the distance (i,k)
	% 	% the number of columns in i and k are same
	% 	curr_dis=sum((X(i)-centroids(k)).^2);
	% 	distances(i,1)=curr_dis;
	% end;
	% lowest_distance_idx=find(distances==min(distances));
	% idx(i)=lowest_distance_idx(end,:);
end;

% =============================================================

end

