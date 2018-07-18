clear;
fid=fopen('Data.txt','r');
data=textscan(fid,'%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f');
data=cell2mat(data);
[Nv,~]=size(data);
X=data(:,1:16);
Y=data(:,17:19);
N=16;
M=3;
Nh=20;
Nu=N+1+Nh;
batch=Nv;
hmean=0.5;
hvar=1;
Nit=1000;
olderr = 1.0e10;
Z1=0;
Z2=0;
RR=0;

xa = zeros(Nu,1);
Oc = zeros(Nu,1);
h = zeros(Nu,1);
Et = zeros(M,1);
th2 = zeros(Nh,1);
hm = zeros(Nh,1);
hv = zeros(Nh,1);
dth = zeros(Nh,1);
% Nh x N
wih2 = zeros(Nh,N);
dwih = zeros(Nh,N);
% M x N
wio = zeros(M,N);
% M x Nh
who = zeros(M,Nh);
who2 = zeros(M,Nh);
dwho = zeros(M,Nh);
% M x Nu
% w = zeros(M,Nu);
w2 = zeros(M,Nu);
% Nu x Nu
R = zeros(Nu,Nu);
% M x Nu
C = zeros(M,Nu);
woo = zeros(M,Nu);
% 归一化输入
[xm,xv]=MeanVariance(X);
for i=1:Nv
    for j=1:N
        if(xv(j)>0)
            X(i,j)=(X(i,j)-xm(j))/sqrt(xv(j)); 
        else
            X(i,j)=(X(i,j)-xm(j));
        end
    end
end
% 归一化输出
[ym,yv]=MeanVariance(Y);
for i=1:Nv
    for j=1:M
        if(yv(j)>0)
            Y(i,j)=(Y(i,j)-ym(j))/sqrt(yv(j)); 
        else
            Y(i,j)=(Y(i,j)-ym(j));
        end
    end
end
% 用高斯随机分布初始化wih
wih=normrnd(0.5,1,Nh,N);
th = normrnd(0.5,1,Nh,1);
% for j = 1:Nh
%     for i =1:N
%         wih(j,i) =wih(j,i) * sqrt(xv(i));
%     end
% end

wih=wihnorm(wih, N, Nh);

for i=1:Nv
    hm = hm+wih*X(i,:)'+th;
end
hm = hm/ Nv;
for i=1:Nv
    hv=hv+(wih*X(i,:)'+th-hm).*(wih*X(i,:)'+th-hm);
    Et = Et + Y(i,:)'.*Y(i,:)';
end
hv=sqrt(hv/Nv);
Et=Et/Nu;

for j =1:Nh
    wih(j,:) = wih(j,:) * (hvar / hv(j));
    th(j) = th(j) * (hvar / hv(j));  %this is correct?
    th(j) = th(j) - (hm(j) * hvar / hv(j)) + hmean;
end
for i=1:Nv
    xa(1:N) = X(i,:)';
    xa(N+1) = 1;
    h = wih*X(i,:)'+th;
    xa(N+2:Nu) = hact(h);%h=wih*X(i,:)'+th
    R = R + xa * xa';
    C = C + Y(i,:)' * xa';
end
R = R/Nv;
C  =C/Nv;

disp('Iteration       Z1      Old Error       New Error       Z2      RR'); 
[E,w] = Schmit(R, C, Nu, M, Et);
% Iteration Nit times
for it =1:Nit
    who = w(1:M,N+2:Nu);
    if (it + 1>9)
        disp(['    ' num2str(it + 1) '           ' num2str(Z1) '      ' num2str(olderr) '      ' num2str(E) '      ' num2str(Z2) '       ' num2str(RR)]); 
    else
        if (olderr>10000)
            disp(['    ' num2str(it + 1) '           ' num2str(Z1) '      ' num2str(olderr) '      ' num2str(E) '          ' num2str(Z2) '        ' num2str(RR)]); 
        else 
            disp(['    ' num2str(it + 1) '           ' num2str(Z1) '      ' num2str(olderr) '      ' num2str(E) '          ' num2str(Z2) '        ' num2str(RR)]); 
        end
    end
    olderr = E;
    [dth,dwih]=backprop(1,batch,N, M, Nh, Nu,X,Y, xa, w, wih, th, who);%求输入层到隐含层的权重梯度，更新dwih,dth
    Z1 = olffunc(1,batch,  N, Nh, X,Y, xa, wih, th, w, dwih,dth);
    % Updating Hidden Weights and Thesholds
    th = th+Z1*dth;
    wih=wih+ Z1*dwih;
    %Z2=olffunc(infile,N,M,Nh,Nu,x,xm,xa,t,wih,dwih,th,w,dth); 
    [R,C]=rcfunc(1,batch,N, M, Nu, X,Y, xa, wih, th);
    
    [E,w] = Schmit(R, C, Nu, M, Et);
%     w=C/R;
    %RR=Z2/Z1;	
end












