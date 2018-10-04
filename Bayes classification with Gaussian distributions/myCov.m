function covarianceMatrix = myCov(A)
covarianceMatrix = (A' * A) / (size(A,1));
end