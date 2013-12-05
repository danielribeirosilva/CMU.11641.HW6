function [X,Y,QueryId] = readLabeledSparseMatrix (fileLocation)

    
    fid = fopen(fileLocation,'rt','n','UTF-8');
    tline = fgets(fid);
    
    X = [];
    Y = [];
    QueryId = [];
    
    currentI = 1;
    while ischar(tline)
        split = textscan(tline,'%s','delimiter',' :');
        split = split{1};
        
        %get label
        Y(1,currentI) = str2double(split(1));
        
        %get query id
        QueryId(1,currentI) = str2double(split(3));
        
        for i = 4:2:(size(split,1)-1)
            if split{i}(1) == '#'
                break;
            end
            featureLabel = str2double(split{i});
            featureValue = str2double(split{i+1});
            X(currentI, featureLabel) = featureValue;
        end
        
        currentI = currentI + 1;
        tline = fgets(fid);
    end
    
    Y = Y';
    
    data.X = X;
    data.Y = Y;
    data.QueryId = QueryId;
    
    save data.mat data;
    
    fclose(fid);
    
end