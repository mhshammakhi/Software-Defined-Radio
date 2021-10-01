function Out = CubicInterpolation(samp0,samp1,samp2,samp3,mu)
v0 = samp1;
v1 =  (-1/3)*samp0 + (-0.5)*samp1 + samp2 + (-1/6)*samp3;
v2 = (0.5)*samp0 + (-1.0)*samp1 + (0.5)*samp2;
v3 = (-1/6)*samp0 + (0.5)*samp1 + (-0.5)*samp2 + (1/6)*samp3;
Out = v0+mu*(v1+mu*(v2+mu*v3));
end