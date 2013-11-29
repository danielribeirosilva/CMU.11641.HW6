dataFilePath = strcat(pwd,'/DATA.TXT');
[fileLocationTrain,fileLocationTest,C] = readDataFile (dataFilePath);

%IF WISHES TO MANUALLY ENTRING DATA
%C = 0.0001;
%fileNameTrain = 'citeseer.train.ltc.svm';
%fileNameTest = 'citeseer.test.ltc.svm';
%dir = 'Users/daniel/Documents/MATLAB/+SearchEnginesHW5/+Data/';
%fileLocationTrain = strcat(dir,fileNameTrain);
%fileLocationTest = strcat(dir,fileNameTest);

% checking to see if pool is already open
if matlabpool('size') == 0 
    matlabpool open
end

tic;

%get max FV size
maxFVSizeTrain = getFVdimension (fileLocationTrain);
maxFVSizeTest = getFVdimension (fileLocationTest);
maxFVSize = max([maxFVSizeTrain maxFVSizeTest]);

%Import database
fprintf('loading training data...\n');
[Xtrain,Ytrain] = readLabeledSparseMatrix (fileLocationTrain, maxFVSize);
fprintf('loading testing data...\n');
[Xtest,Ytest] = readLabeledSparseMatrix (fileLocationTest, maxFVSize);
fprintf('data loaded\n\n');

%Add column of 1's to data
Xtrain = [(0.01*ones(size(Xtrain,1),1)) Xtrain];
Xtest = [(0.01*ones(size(Xtest,1),1)) Xtest];


%Model parameters
alpha = 0.1*0.001/C;
convPrecision = min([0.1*0.001/C 0.1]);
maxT = 15;
adaptativeLearningRate = 0.9;
FV_dimension = size(Xtrain,2);
labels = unique(Ytrain);
predictions = zeros(size(Ytest,1),size(labels,1));


%Train one logistic regression model for each class (one vs. rest)
parfor i_label = 1:length(labels)
    
    label = labels(i_label);
    fprintf('\ncurrent label: %i\n', label);

    currentYtrain = zeros(size(Ytrain));
    currentYtrain(Ytrain==label) = 1;
    
    currentAlpha = alpha;

    %TRAINING
    w = zeros(1, FV_dimension);
    lastValue = 0;
    convValue = convPrecision + 1;

    T = 0;
    while T < maxT && convValue > convPrecision
        
        T = T + 1;
        
        %shuffle data for stochastic descent algorithm
        randIndex = randperm(size(Xtrain,1));
        shuffledXtrain = Xtrain(randIndex,:);
        shuffledYtrain = currentYtrain(randIndex);

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

        LossFunction = bsxfun(@times,currentYtrain,log(P)) + bsxfun(@times,(1-currentYtrain),log(1-P));
        LossFunction = sum(LossFunction);

        currentValue = LossFunction - 0.5*C*sum(dot(w(2:end),w(2:end)));

        convValue = abs(lastValue - currentValue);
        lastValue = currentValue;

        fprintf('T=%i (alpha=%f): %f %f \n', T, currentAlpha, convValue, currentValue);
        
        %Adaptative Learning
        %learning rate decreases with time: the closer we are to
        %the global max/min, the lower we want the learning rate to be
        if T > 0
           currentAlpha = currentAlpha * adaptativeLearningRate;
        end

    end


    %TESTING
    currentPred = 1 + exp(-Xtest*w');
    currentPred = bsxfun(@rdivide,1,currentPred);
    predictions(:,i_label) = currentPred;

end

%CLASS PREDICTION
%select model that gives highest probability for given Xi
[maxP, classPrediction] = max(predictions');
classPrediction = classPrediction';

precisionVector = zeros(size(labels));
recallVector = zeros(size(labels));
F1Vector = zeros(size(labels));

%my own eval function for matlab
fprintf('Computing Predictions ...\n');
for i = 1:length(labels)
    
    label = labels(i);
    %fprintf('current label: %i\n', label);
    
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

    fprintf('P:%.3f, R:%.3f, F1:%.3f \n', precision, recall, F1);
end

%total running time
elapsedTime = toc;
disp(elapsedTime);

%save results
LogReg.precision = precisionVector;
LogReg.avgPrecision = mean(precisionVector);
LogReg.recall = recallVector;
LogReg.avgRecall = mean(recallVector);
LogReg.F1 = F1Vector;
LogReg.avgF1 = mean(F1Vector);
LogReg.predictions = classPrediction;
LogReg.C = C;
LogReg.adaptativeLearningRate = adaptativeLearningRate;
LogReg.alpha = alpha;
LogReg.maxT = maxT;
LogReg.elapsedTime = elapsedTime;

%output all results as a .mat file
save LogReg.mat LogReg;

%output .txt file for eval.cpp
fileID = fopen('eval.txt','w');

%set test labels to zero
%Ytest = zeros(size(Ytest));

fprintf(fileID,'%i %i\n',[classPrediction'; Ytest']);

fclose(fileID);








