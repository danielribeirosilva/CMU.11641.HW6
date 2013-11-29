function [maxFVsize] = getFVdimension (fileLocation)

    
    fid = fopen(fileLocation,'r');
    tline = fgets(fid);
    
    maxFVsize = 0;
    
    currentI = 1;
    while ischar(tline)
        split = textscan(tline,'','delimiter',' :');
        
        lastPosition = size(split,2)-1;
        Jcell = split(1,lastPosition);
        if Jcell{1}>maxFVsize
            maxFVsize = Jcell{1};
        end
        
        currentI = currentI + 1;
        tline = fgets(fid);
    end

    fclose(fid);
    
end