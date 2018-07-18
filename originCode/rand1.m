function [ V ] = rand1(  ix, iy, iz  )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
	ixx = (ix) / 177;
	ix = 171 * mod(ix,177) - 2 * ixx;

    if (ix < 0)
        ix =ix + 30269;
    end
    iyy = (iy) / 176;
    iy = 176 * mod(iy,176) - 2 * iyy;

    if (iy < 0)
        iy = iy +30307;
    end
    izz = (iz) / 178;
    iz = 170 * mod(iz,178) - 2 * izz;

    if (iz < 0)
        iz = iz +30323;
    end
	temp = ix / 30629.0 + iy / 30307.0 + iz / 30323.0;
	itemp = floor(temp);
	V= temp - itemp;
end

