function [Z ] = olffunc(I,batch, N, Nh, X,Y, xa, wih, th, w, dwih,dth )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
	d1 = 0; 
    d2 = 0;
    for i=(I-1)*batch+1+1:I*batch 
        xa(1:N) = X(i,:)';
        xa(N+1) = 1;

        n = wih* xa(1:N);
        t1= dwih * xa(1:N);
        n = n + th;
        t1 = t1 + dth;
        
        OO = hact(n);
        fder = OO.* (1 - OO);

        yy = w(:,1:N) * xa(1:N)+w(:,N);
        yy = yy + w(:,N+2:N+Nh+1) * OO;
        t2 =w(:,N+2:N+Nh+1) * (fder.* t1);
        d1 = d1 + (Y(i,:)' - yy)'*t2;
        d2 = d2 + t2'* t2;
    end
	Z = d1 / d2;
	
end

