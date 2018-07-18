function [ xm,xv ] = MeanVariance( X )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    [u,v]=size(X);
    xm =sum(X)'/u;
    xv =zeros(v,1);
    for i=1:v
        for j=1:u
            xv(i)=xv(i)+(X(j,i)-xm(i))*(X(j,i)-xm(i))/u;
        end
    end
end

