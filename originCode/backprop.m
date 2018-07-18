function [ dth,dwih ] = backprop(I,batch,N, M, Nh, Nu,X,Y, xa, w, wih, th, who )
%UNTITLED2 Summary of this function goes here
%   ¸üÐÂdwih,dth
	dph = zeros(Nh,1);
	dth = zeros(Nh,1);
    dwih = zeros(Nh,N);
    
    for i=(I-1)*batch+1+1:I*batch 
        xa(1:N) = X(i,:)';
        h = wih*X(i,:)'+th;
        xa(N+1) = 1;
        xa(N+2:Nu) = hact(h);
        % Calculating dpo[k]
        y =   w*xa;
        dpo = 2.0*(Y(i,:)' - y);
         % Calculating dph[j]
        dph =dph+ who' * dpo;
        dph = dph.*hact(h).*(1 - hact(h));
        dth = dth +dph * 1;% Delta for Hidden Units's Thresholds
        dwih = dwih +dph* xa(1:N)';% Delta for Hidden Weights
    end
    
	% Normalizing the deltas
    dth=dth/ batch;
    dwih = dwih/batch;
end

