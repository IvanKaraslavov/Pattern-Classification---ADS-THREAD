function mean = myMean(A)
mean = sum(A,1) ./ size(A,1);
end