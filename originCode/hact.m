function [H] = hact( h )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
	[m,~]=size(h);
    H=zeros(m,1);
    for i=1:m
        H(i)= 1 / (1 + exp(-h(i)));
    end
	
end

