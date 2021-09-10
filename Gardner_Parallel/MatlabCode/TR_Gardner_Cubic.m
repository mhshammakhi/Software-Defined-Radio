function TR_Out = TR_Gardner_Cubic(TR_In , sps2 , TR_Alpha , TR_Betta)
mu=0;tedAcc=0;muAcc=0;
a=2^(-TR_Alpha);b=2^(-TR_Betta);
val1=0;val2=0;
%%
TR_Out=zeros(numel(TR_In),1);
flagTed=true;
%%
i=4;
j=1;
TR_In = [0 ; 0 ; 0 ; TR_In];
while(i<numel(TR_In))
    samp0 = TR_In(i-3);
    samp1 = TR_In(i-2);
    samp2 = TR_In(i-1);
    samp3 = TR_In(i);
    v0 = samp1;
    v1 =  (-1/3)*samp0 + (-0.5)*samp1 + samp2 + (-1/6)*samp3;
    v2 = (0.5)*samp0 + (-1.0)*samp1 + (0.5)*samp2;
    v3 = (-1/6)*samp0 + (0.5)*samp1 + (-0.5)*samp2 + (1/6)*samp3;
    val0 = v0+mu*(v1+mu*(v2+mu*v3));
    if(flagTed == true)
        ted = real(conj(val1)*(val2-val0));
        flagTed = false;
        TR_Out(j)=val1;
        j=j+1;
        TR_Out(j)=val0;
        j=j+1;
    else
        ted = 0;
        flagTed = true;
    end
    jitter = a*ted + b*tedAcc;
    tedAcc = tedAcc+ted;
    mu = mu+(sps2 + jitter);
    muAcc =muAcc + sps2 + jitter;
    val2 = val1;
    val1 = val0;
    i=i+floor(mu);
    mu=mu-floor(mu);
end
TR_Out=TR_Out(1:(j-1));
