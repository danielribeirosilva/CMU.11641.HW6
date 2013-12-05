function outputVtrain (vTrain, vTrainFileName, FVsize)

  fileID = fopen(vTrainFileName,'w');  
  
  rowFormat = '+1 ';
  for i = 1:FVsize
      rowFormat = [rowFormat ' ' num2str(i) ':%f'];
  end
  rowFormat = [rowFormat '\n'];

  fprintf(fileID,rowFormat,vTrain');

end