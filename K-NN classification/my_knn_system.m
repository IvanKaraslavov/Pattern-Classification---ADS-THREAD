%Load the data
filename = 'data.mat';
delimiterIn = '\t';
headerlinesIn = 1;

file_data = importdata(filename, delimiterIn, headerlinesIn);
Xtrn = single(file_data.train.images); 
Ctrn = file_data.train.labels;
Xtst = single(file_data.test.images);
Ctst = file_data.test.labels;

% We are given Ks
Ks = [1,3,5,10,20];
tic
predictions = my_knn_classify(Xtrn, Ctrn, Xtst, Ks');
toc

n = size(Ctst,1);

%Go through each k
for i=1:size(Ks',1)
    [CM, acc] = my_confusion(Ctst, predictions(:,i));
    % Save the files
    file_name = ['cm',num2str(Ks(i)), '.mat'];
    save(file_name, 'CM');
    % Print the needed information
    fprintf('The number of nearest neighbours: %i\nThe number of test samples: %i\nThe number of wrongly classified test samples: %i\nAccuracy: %.4f\n\n', Ks(i), n,sum(CM(:)) - sum(diag(CM)),acc);
end