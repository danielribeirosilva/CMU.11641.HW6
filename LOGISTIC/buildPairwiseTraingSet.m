function [Xv,Qid] = buildPairwiseTraingSet (Xcoarse,Ycoarse,QueryId)
    
    queryIds = unique(QueryId);
    totalQueries = size(queryIds,2);
    Xv=[];
    Qid = [];
    
    for i=1:totalQueries
       currentX = Xcoarse(QueryId==queryIds(i),:);
       currentY = Ycoarse(QueryId==queryIds(i));
       positiveX = currentX(currentY>0,:);
       negativeX = currentX(currentY<=0,:);
       totalNegativeLabels = size(negativeX,1);
       totalPositiveLabels = size(positiveX,1);
       
       positiveX = repmat(positiveX,totalNegativeLabels,1);
       negativeX = sortrows(repmat(negativeX,totalPositiveLabels,1));
       currentV = positiveX - negativeX;
       currentQid = queryIds(i)*ones(size(currentV,1),1);
       Xv = [Xv;currentV];
       Qid = [Qid;currentQid];
    end
    
end