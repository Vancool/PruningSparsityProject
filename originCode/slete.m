function [ V ] = slete( dstd,dmean )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    IX = 3;
    IY = 4009;
    IZ = 234;
    PI=3.1415926;
	V = dmean + dstd*cos(2 * PI*rand1(IX, IY, IZ))*sqrt(-2.0*log(rand1(IX, IY, IZ)));
end