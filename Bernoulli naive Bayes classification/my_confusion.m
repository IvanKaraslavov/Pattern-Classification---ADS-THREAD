function [CM, acc] = my_confusion(Ctrues, Cpreds)
k = 26; % we are given that k is a constant
CM = zeros(k,k); % creating the desired matrix
n = size(Ctrues,1);
for i=1:n
    CM(Ctrues(i), Cpreds(i)) = CM(Ctrues(i), Cpreds(i)) + 1; %increasing the number of samples classified
end
totalSum = sum(CM(:));
diagonalSum = sum(diag(CM)); % the number of the correct classification is the diagonal of the matrix
acc = diagonalSum / totalSum;
end