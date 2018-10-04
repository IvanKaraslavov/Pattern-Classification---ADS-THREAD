function [Cpreds] = my_bnb_classify(Xtrn, Ctrn, Xtst, treshold)
% Construct the needed matrices
n = size(Xtst, 1);
Cpreds = zeros(n, 1);
classNumber = zeros(26,1);
classAcc = zeros(n,26);
% Binerization of the matrices using the treshold
Xtrn = Xtrn >= treshold;
Xtst = Xtst >= treshold;
%Find the number of vectors for each class
for i=1:size(Ctrn,1)
    classNumber(Ctrn(i)) = classNumber(Ctrn(i)) + 1;
end
% Go through each class
for i=1:26
    %Take the vectors from each class
    classMatrix = Xtrn(Ctrn(:,1) == i, :);
    % Compute the probability matrix
    probabilityMatrix = sum(classMatrix,1)' ./ classNumber(i);
    %Replace 0 with a very small number, so we do not have -Inf when using
    %the log operation
    probabilityMatrix(probabilityMatrix == 0) = 1.0E-10;
    % Go though each test vector
    for j=1:n
        % Compute the probability that the vector belongs to that class
        % We are using the log operation, because the probabilities can
        % become really small when working with big data
        classAcc(j,i) = log(prod(probabilityMatrix(Xtst(j,:) == 1,:)) * prod((1-probabilityMatrix(Xtst(j,:) == 0,:))));
    end
end
[~,index] = sort(classAcc,2, 'descend');
Cpreds(:) = index(:,1);
end