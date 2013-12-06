function outputMatrixToTxtFile (vTrain, vTrainFileName)

  disp(vTrainFileName);
  fileID = fopen(vTrainFileName,'w');  
  
  FVsize = size(vTrain,2);
  
  rowFormat = '+1 ';
  for i = 1:FVsize
      rowFormat = [rowFormat ' ' num2str(i) ':%.15f'];
  end
  rowFormat = [rowFormat '\n'];

  %fprintf(fileID,rowFormat,vTrain');

end