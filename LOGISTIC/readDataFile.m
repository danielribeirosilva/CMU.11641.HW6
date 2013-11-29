function [trainPath,testPath,C] = readDataFile (fileLocation)

    fid = fopen(fileLocation,'r');
    tline = fgets(fid);
    
    while ischar(tline)
        split = textscan(tline,'%s','delimiter','=');
        variableName = split{1}(1,1);
        variableValue = split{1}(2,1);
        
        if strcmp(variableName,'train')
            trainPath = variableValue{1};
        elseif strcmp(variableName,'test')
            testPath = variableValue{1};
        elseif strcmp(variableName,'c')
            C = str2double(variableValue);
        end

        tline = fgets(fid);
    end
    
    fclose(fid);
    
end