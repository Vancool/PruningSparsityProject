function yMatrix=OneHot(y,k)
	dataSize=length(y);
	yMatrix=zeros(k,dataSize);
	y=y+1;% change from 0-9 to 1-10
	for i=1:dataSize
		label=y(i);
		yMatrix(label,i)=1;
	end
end