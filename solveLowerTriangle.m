%solve the result of lower triangle matrix
function result=solveLowerTriangle(y,C,tol)
	m=length(y);
	result=zeros(m,1);
	for i=1:m
		k=0;
		if(i>=2)
			for j=1:i-1
				k=k+C(i,j)*result(j);
			end	
		end
		k=y(i)-k;
		if(abs(C(i,i))<tol)
			result(i)=0;
		else
			result(i)=k/C(i,i);
        end
	end

end