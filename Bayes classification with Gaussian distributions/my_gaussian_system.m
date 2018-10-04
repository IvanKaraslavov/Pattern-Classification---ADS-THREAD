%Load the data
filename = 'data.mat';
delimiterIn = '\t';
headerlinesIn = 1;

file_data = importdata(filename, delimiterIn, headerlinesIn);
Xtrn = double(file_data.train.images) ./ 255;
Ctrn = file_data.train.labels;
Xtst = double(file_data.test.images) ./ 255;
Ctst = file_data.test.labels;
%Find the predicted classes
tic
[predictions, Ms, Covs] = my_gaussian_classify(Xtrn, Ctrn, Xtst, 0.01);
toc
n = size(Ctst,1);

% Compute the condusion matrix
[CM, acc] = my_confusion(Ctst, predictions(:));
% Save the files
save('cm.mat', 'CM');
meanLastClass = Ms(:,26);
save('m26.mat', 'meanLastClass');
covarianceLastClass = Covs(:,:,26);
save('cov26.mat', 'covarianceLastClass');
% Print the results
fprintf('The number of test samples: %i\nThe number of wrongly classified test samples: %i\nAccuracy: %.4f\n', n,sum(CM(:)) - sum(diag(CM)),acc);