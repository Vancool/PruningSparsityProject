function [ wih ] = wihnorm(wih, N, Nh)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    mm=zeros(Nh,1);
    vv=zeros(Nh,1);
    for i =1:Nh
        for k =1:N
            mm(i) = mm(i)+wih(i,k)/N;
        end
    end

    for i = 1:Nh
        for k =1:N
            vv(i) = vv(i) +(wih(i,k) - mm(i))*(wih(i,k) - mm(i));
        end
    end
    for i = 1:Nh
        vv(i) = sqrt(vv(i) / N);
    end
    for i = 1:Nh
        for k = 1:N
            wih(i,k) = (wih(i,k) - mm(i)) / vv(i);
        end
    end
end

