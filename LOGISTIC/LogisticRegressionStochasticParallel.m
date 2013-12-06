dataFilePath = strcat(pwd,'/../DATA.TXT');
[fileLocationTrain,fileLocationTest,C] = readDataFile (dataFilePath);


tic;


%Import database
% fprintf('loading training data...\n');
 [Xtrain,Ytrain,QueryIdTrain] = readLabeledSparseMatrix (fileLocationTrain);
% fprintf('loading testing data...\n');
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




Xtrain = normalizeMatrix(Xtrain);

%Pairwise training
[Xv,Qid] = buildPairwiseTrainingSet (Xtrain,Ytrain,QueryIdTrain);
totalRowsV = size(Xv,1);
Xtrain = [Xv;-Xv];
Ytrain = [ones(totalRowsV,1);zeros(totalRowsV,1)];

%Normalize rows
%Xtrain = normalizeMatrix(Xtrain);
Xtest = normalizeMatrix(Xtest);

%numRowsTrain = size(Xtrain,1);
%numRowsTest = size(Xtest,1);
%Xtrain = spdiags(1./sum(Xtrain,2),0,numRowsTrain,numRowsTrain)*Xtrain;
%Xtest = spdiags(1./sum(Xtest,2),0,numRowsTest,numRowsTest)*Xtest;


%Model parameters
alpha = 0.004 / max(log(C/0.0001),1);
convPrecision = 0.005;
maxT = 100;
adaptativeLearningRate = 0.95;
FV_dimension = size(Xtrain,2);

%TRAINING
w = zeros(1, FV_dimension);
lastValue = 0;
convValue = convPrecision + 1;
currentAlpha = alpha;
T = 0;
while T < maxT && convValue > convPrecision

    T = T + 1;

    %shuffle data for stochastic descent algorithm
    randIndex = randperm(size(Xtrain,1));
    shuffledXtrain = Xtrain(randIndex,:);
    shuffledYtrain = Ytrain(randIndex);

    %stochastic gradient descent
    for i=1:size(shuffledXtrain,1)

        currentX = shuffledXtrain(i,:);
        currentY = shuffledYtrain(i);

        p = 1 / (1 + exp(-dot(currentX,w)) );
        w = w + currentAlpha*( (currentY - p)*currentX - C*w ); 

    end 

    %compute loss funcion and decide if is close enough to max/min
    %point. (Check how much loss function changed)
    P = 1 + exp(-Xtrain*w');
    P = bsxfun(@rdivide,1,P);

    LossFunction = bsxfun(@times,Ytrain,log(P)) + bsxfun(@times,(1-Ytrain),log(1-P));
    LossFunction = sum(LossFunction);

    currentValue = LossFunction - 0.5*C*sum(dot(w,w));

    convValue = abs(lastValue - currentValue);
    lastValue = currentValue;

    fprintf('T=%i (alpha=%.9f): %f %f \n', T, currentAlpha, convValue, currentValue);

    %Adaptative Learning
    %learning rate decreases with time: the closer we are to
    %the global max/min, the lower we want the learning rate to be
    if T > 0
       currentAlpha = currentAlpha * adaptativeLearningRate;
    end

end

%OUTPUTING MODEL
predictions = 1 + exp(-Xtest*w');
predictions = bsxfun(@rdivide,1,predictions);

%predictionFileName = ['predLR_C', num2str(C) ,'_alpha', num2str(alpha) ,'_T', num2str(maxT),'.txt'];
predictionFileName = './hw6_predictions.txt';

fileID = fopen(predictionFileName,'w');

fprintf(fileID,'%f %i\n',[predictions'; Ytest']);

fclose(fileID);

%RUN EVALUATION SCRIPT
%{
evalOutputFileName = ['evalLR_C', num2str(C) ,'_alpha', num2str(alpha) ,'_T', num2str(maxT),'.txt'];

terminalCommand = ['perl Eval-Score.pl ' fileLocationTest ' ' predictionFileName ' ' evalOutputFileName ' 0'];

[status,cmdout] = system(terminalCommand);

toc
%}

