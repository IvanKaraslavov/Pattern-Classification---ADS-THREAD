%Load the data
filename = 'data.mat';
delimiterIn = '\t';
headerlinesIn = 1;

file_data = importdata(filename, delimiterIn, headerlinesIn);
Xtrn = double(file_data.train.images);
Ctrn = file_data.train.labels;
Xtst = double(file_data.test.images);
Ctst = file_data.test.labels;
%Find the predicted classes
tic
predictions = my_bnb_classify(Xtrn, Ctrn, Xtst, 1);
toc
n = size(Ctst,1);

% Compute the condusion matrix
[CM, acc] = my_confusion(Ctst, predictions(:));
% Save the files
save('cm.mat', 'CM');
% Print the results
fprintf('The number of test samples: %i\nThe number of wrongly classified test samples: %i\nAccuracy: %.4f\n', n,sum(CM(:)) - sum(diag(CM)),acc);