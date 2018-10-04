function [Cpreds] = my_knn_classify(Xtrn, Ctrn, Xtst, Ks)
% Construct the needed matrix
n = size(Xtst, 1);
l = size(Ks,1);
Cpreds = zeros(n, l);
% Construct the distance matrix
training_rows = size(Xtrn,1);
training = sum(Xtrn.*Xtrn,2);
test = sum(Xtst.*Xtst,2);
DI = repmat(test, 1, training_rows) - 2 * Xtst * Xtrn' + repmat(training,1, n)';
% Sort it and take the indeces
[~,index] = sort(DI,2, 'ascend');
%Go through each k
for j=1:l
   k = Ks(j);
   indeces = index(:,1:k); % keep the first ’Knn’ indexes columns
   predictionClass = mode(Ctrn(indeces),2); % the mode on those ’Knn’ indexes columns
   Cpreds(:,j) = predictionClass;
end
% Go through each test example
%for i=1:n
%    distances = euclidean_distance(Xtrn, Xtst(i,:));
%    [~,idx] = sort(distances, 2, 'ascend');
%    % Go through each k from Ks
%    for j=1:l
%        k = Ks(j);
%        indeces = idx(1:k); % keep the first ’Knn’ indexes
%        predictionClass = mode(Ctrn(indeces)); % the mode on those ’Knn’ indexes
%        Cpreds(i,j) = predictionClass;
%    end
%end
end