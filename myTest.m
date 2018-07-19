
R=zeros(785,785);
R=gpuArray(R);
tic


for i=1:4000
    x=gpuArray(randn(785,1));
    
    R=R+x*x';
end
t=toc
