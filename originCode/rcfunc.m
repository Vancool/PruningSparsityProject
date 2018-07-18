function [ R,C ] = rcfunc( I,batch ,N, M, Nu, X,Y, xa, wih, th)
%UNTITLED Summary of this function goes here
%  Passing thru the data to calculate the updated R and C matrices
	R =zeros(Nu,Nu);
	C =zeros(M,Nu);
	
    for i=(I-1)*batch+1+1:I*batch 
        xa(1:N) = X(i,:)';
        h = wih*X(i,:)'+th;
        xa(N+1) = 1;
        xa(N+2:Nu) = hact(h);

        R = R+xa * xa';
        C = C +Y(i,:)'* xa';
    end
    R = R /batch;
    C = C /batch;
	

end

