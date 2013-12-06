dataFilePath = strcat(pwd,'/../DATA.TXT');
[fileLocationTrain,fileLocationTest,C] = readDataFile (dataFilePath);


tic;

%Import database
 %fprintf('loading training data...\n');
 [Xtrain,Ytrain,QueryIdTrain] = readLabeledSparseMatrix (fileLocationTrain);
 %fprintf('loading testing data...\n');
 [Xtest,Ytest,QueryIdTest] = readLabeledSparseMatrix (fileLocationTest);
% fprintf('data loaded\n\n');

%{
load('train.mat');
Xtrain = data.X;
Ytrain = data.Y;
QueryIdTrain = data.QueryId;
load('test.mat');
Xtest = data.X;
Ytest = data.Y;
clearvars 'data';
%}

%Add custom parameters

%Feature 1
fOne = 39;
fTwo = 40;
avgOne = mean(Xtrain(:,fOne));
avgTwo = mean(Xtrain(:,fTwo));
Xtrain(:,size(Xtrain,2)+1) = sqrt( (Xtrain(:,fOne) - avgOne).*(Xtrain(:,fTwo) - avgTwo));
Xtest(:,size(Xtest,2)+1) = sqrt( (Xtest(:,fOne) - avgOne).*(Xtest(:,fTwo) - avgTwo));

%Feature 2
Xtrain(:,size(Xtrain,2)+1) = Xtrain(:,14).*Xtrain(:,1);
Xtest(:,size(Xtest,2)+1) = Xtest(:,14).*Xtest(:,1);

%Feature 3
Xtrain(:,size(Xtrain,2)+1) = Xtrain(:,34).*Xtrain(:,14);
Xtest(:,size(Xtest,2)+1) = Xtest(:,34).*Xtest(:,14);










%Pairwise training
[Xv,Qid] = buildPairwiseTrainingSet (Xtrain,Ytrain,QueryIdTrain);
Xtrain = Xv;

%Normalize rows
Xtrain = normalizeMatrix(Xtrain);
Xtest = normalizeMatrix(Xtest);
%numRowsTrain = size(Xtrain,1);
%numRowsTest = size(Xtest,1);
%Xtrain = spdiags(1./sum(Xtrain,2),0,numRowsTrain,numRowsTrain)*Xtrain;
%Xtest = spdiags(1./sum(Xtest,2),0,numRowsTest,numRowsTest)*Xtest;

%Output v vectors (for training) and x vectors (for testing)
%to txt file in SVM_light format
%fprintf('\nOutputing data do file ...\n');
trainingFileName = 'temp_train.txt';
outputMatrixToTxtFile (Xtrain, trainingFileName);
testingFileName = 'temp_test.txt';
outputMatrixToTxtFile (Xtest, testingFileName);


%TRAIN -> GENERATE MODEL
%fprintf('\nGenerating Model ...\n');
command = './svm_learn';
commandOpts = ['-b 0 -# 20000 -c ',num2str(C)];
modelFileName = ['SVMmodel_C' num2str(C)];

terminalCommand = [command, ' ', commandOpts, ' ', trainingFileName, ' ', modelFileName];
[statusA,cmdoutA] = system(terminalCommand);


%TEST -> GENERATE RATING
%fprintf('\nClassifying test data ...\n');
command = './svm_classify';

%predictionsFileName = ['SVMpredictions_C' num2str(C) '.txt'];
predictionsFileName = './hw6_predictions.txt';

terminalCommand = [command ' ' testingFileName ' ' modelFileName ' ' predictionsFileName];
[statusB,cmdoutB] = system(terminalCommand);


%RUN EVALUATION SCRIPT
%{
evalOutputFileName = ['evalSVM_C' num2str(C) '.txt'];
terminalCommand = ['perl Eval-Score.pl ' fileLocationTest ' ' predictionsFileName ' ' evalOutputFileName ' 0'];
[statusC,cmdoutC] = system(terminalCommand);

toc
%}

