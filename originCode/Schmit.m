function [ MSE,w ] = Schmit( R, C, Nu, M, Et )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
	gmin = 0.000001;
    MSE = 0;
    a=zeros(Nu,Nu);
	b=zeros(Nu,1);
    c=zeros(Nu,1); 
	E = Et;
	gmin = gmin*gmin;
	g = R(1,1);
    NLin=0;
    
    if (g<gmin) 
        a(1,1) = 0;
        NLin = NLin + 1;
    else 
        g = sqrt(g);
        a(1,1) = 1 / g;
    end

	c(1) = a(1,1) * R(1,2);
	b(1) = -c(1) * a(1,1);
	b(2) = 1;
	g = R(2,2) - c(1) * c(1);
    if (g<gmin)
        a(2,1) = 0;
        a(2,2) = 0;
        NLin = NLin + 1;	
    else 
        g =sqrt(g);
        a(2,1) = b(1) / g;
        a(2,2) = b(2) / g;
    end
	
    for i = 3:Nu
        for j = 1:i - 1
            c(j) = 0;
            c(j) = c(j) + a(j,1:j) * R(1:j,i);
        end

        b(i) = 1;
        for j = 1: i - 1
            b(j) = 0;
            b(j) = b(j) - c(j:i-1)' * a(j:i-1,j);
        end

        sumck = 0;
        sumck =sumck+ c(1:i-1)' * c(1:i-1);
        g = R(i,i)- sumck;

        if (g<gmin)
            for k = 1:i
                a(i,k) = 0;
            end
            NLin = NLin + 1;
        else
            g = 1 / sqrt(g);
            for k =1:i
                a(i,k) = b(k) * g;
            end
        end
    end
	% find orthonormal output weights
    Wt = zeros(M,Nu);
    for i =1:M
        for m = 1:Nu    
            Wt(i,m) =Wt(i,m)+ a(m,1:m) * C(i,1:m)';
        end
    end
	% find training errors E[i] and E
    for i = 1:M
        for k = 1:Nu
            E(i) = E(i) - Wt(i,k) * Wt(i,k);
        end
    end
	
	MSE = MSE + sum(E);
	w=zeros(M,Nu);
	% find output weights for orginal system
    for i = 1:M
        for k = 1:Nu
            w(i,k) =  Wt(i,k:Nu)*a(k:Nu,k);
        end
    end
end

