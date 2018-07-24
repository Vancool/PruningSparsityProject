
%HWO -count the learning direction using jacobi matrix
function hMatrix=HWO(R0,G,tol)
	[Q,R]=qr(R0);
	R=R';
	G=G';
	[m,n]=size(G);
	y=[];
	for i=1:n
		y=[y solveLowerTriangle(G(:,i),R,tol)];
	end
	y=y';
	hMatrix=y*Q';
end
