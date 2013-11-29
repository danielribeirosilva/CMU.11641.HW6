dataFolderPath = 'Users/daniel/Documents/MATLAB/+SearchEnginesHW5/+SVM/'; 
trainFilePath = 'Users/daniel/Documents/MATLAB/+SearchEnginesHW5/+SVM/train.txt';
testFilePath = 'Users/daniel/Documents/MATLAB/+SearchEnginesHW5/+SVM/test.txt';
%binaryFolder = 'Users/daniel/Documents/MATLAB/+SearchEnginesHW5/SVM/';
binaryFolder = '\+SearchEnginesHW5/+SVM/';
binaryFolder2 = '+SearchEnginesHW5/+SVM/';

tic;
currentC = 0.0001;
labels = 1:17;


%TRAIN -> GENERATE MODEL
fprintf('\n%i Generating Models ...\n', size(labels,2));
for currentClass = labels
    
    command = strcat(binaryFolder,'svm_learn');
    commandOpts = strcat('-c',{' '},num2str(currentC));
    commandInput = strcat(binaryFolder,'train_',num2str(currentClass),'.txt');
    commandOutput = strcat(binaryFolder,'model_',num2str(currentC),'_',num2str(currentClass));

    terminalCommand = strcat(command, {' '}, commandOpts, {' '}, commandInput, {' '}, commandOutput);
    [status,cmdout] = system(terminalCommand{1});

end



%TEST -> GENERATE PROJECTIONS
fprintf('\nClassifying test data ...\n');
for currentClass = 1:17

    command = strcat(binaryFolder,'svm_classify');
    commandTestInput = strcat(binaryFolder,'test_',num2str(currentClass),'.txt');
    commandModelInput = strcat(binaryFolder,'model_',num2str(currentC),'_',num2str(currentClass));
    commandOutput = strcat(binaryFolder,'classification_',num2str(currentC),'_',num2str(currentClass));

    terminalCommand = strcat(command, {' '}, commandTestInput, {' '}, commandModelInput, {' '}, commandOutput);
    [status,cmdout] = system(terminalCommand{1});

end



%COMPILE RESULTS
fprintf('\nCompiling results ...\n');
allProjections = [];
for currentClass = 1:17

    classificationFile = strcat(binaryFolder2,'classification_',num2str(currentC),'_',num2str(currentClass));
    fid = fopen(classificationFile,'r');
    projection = fscanf(fid,'%f');
    allProjections = [allProjections projection];
    
end


%DO MULTI-CLASS CLASSIFICATION BY USING MAX
%select model that gives highest projection for given Xi
fprintf('Computing Predictions ...\n');

[Xtest,Ytest] = SearchEnginesHW5.readLabeledSparseMatrix (testFilePath);
[maxP, classPrediction] = max(allProjections');
classPrediction = classPrediction';

precisionVector = zeros(size(labels));
recallVector = zeros(size(labels));
F1Vector = zeros(size(labels));

for i = 1:length(labels)
    
    label = labels(i);
    fprintf('current label: %i\n', label);
    
    %true negatives
    a = sum((classPrediction~=label).*(Ytest~=label));
    %false positives
    b = sum((classPrediction==label).*(Ytest~=label));
    %false negatives
    c = sum((classPrediction~=label).*(Ytest==label));
    %true positives
    d = sum((classPrediction==label).*(Ytest==label));

    precision = d / (d + b);
    recall = d / (c + d);
    accuracy = (a + d) / (a + b + c + d);
    F1 = 2*precision*recall / (precision + recall);
    
    precisionVector(i,1) = precision;
    recallVector(i,1) = recall;
    F1Vector(i,1) = F1;

    %fprintf('P:%.3f, R:%.3f, A:%.3f \n', precision, recall, accuracy);
    fprintf('P:%.3f, R:%.3f, F1:%.3f \n', precision, recall, F1);
end

%total running time
elapsedTime = toc;
fprintf('\nelapsed time: %f seconds\n',elapsedTime);

%save results
SVMresult.precision = precisionVector;
SVMresult.avgPrecision = mean(precisionVector);
SVMresult.recall = recallVector;
SVMresult.avgRecall = mean(recallVector);
SVMresult.F1 = F1Vector;
SVMresult.avgF1 = mean(F1Vector);
SVMresult.predictions = classPrediction;
SVMresult.C = currentC;
SVMresult.elapsedTime = elapsedTime;

save SVMresult.mat SVMresult;

%output .txt file for eval.cpp
evalFileName = strcat('eval_SVM_',num2str(currentC),'.txt');
fileID = fopen(evalFileName,'w');
fprintf(fileID,'%i %i\n',[classPrediction'; Ytest']);
fclose(fileID);

