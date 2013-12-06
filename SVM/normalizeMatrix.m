function B = normalizeMatrix(A)

    normVector = sqrt(sum(A.^2,2));
    normVector = 1./normVector;
    
    B = bsxfun(@times,normVector,A);
    
    B(isnan(B)) = 0;
end