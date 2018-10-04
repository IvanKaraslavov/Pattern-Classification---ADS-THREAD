function [PC_X, EVecs] = myPca( X )
    % computes principal components of a data set
    % returns evalues in descending order and corresponding evectors

    % center data
    x_mean = myMean(X);
    X = bsxfun(@minus, X, x_mean);
    
    % calculate covariance matrix
    covar_m = myCov(X);
    
    % Principal Component (Eigenvectors), 
    % Corresponding Eigenvalues
    [EVecs, EVals] = eig(covar_m);
    
    %sort eigenvalues by largest eigenvalues
    EVals = diag(EVals);
    [~, ridx] = sort(EVals, 1, 'descend'); 
    
    %EVals = EVals(ridx); 
    EVecs = EVecs(:, ridx); 
    
    % extra constraints: 
    % first element of each evec is non-negative
    for i = 1 : size(EVecs, 2)
        if (EVecs(1, i) < 0)
            EVecs(:, i) = EVecs(:, i) * -1; 
        end
    end
    PC_X = X * EVecs(:,1:70);
end