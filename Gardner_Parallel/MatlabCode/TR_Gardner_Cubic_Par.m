function TR_Out2 = TR_Gardner_Cubic_Par(TR_In , sps2 , TR_Alpha , TR_Betta)
a=2^(-TR_Alpha);b=2^(-TR_Betta);
%%
mu_frac=0;
mu_pre=sps2;
tedAcc=0;
muAcc=0;
N_input=numel(TR_In);
%%
N=64;
n_thread=N;

mu_new=zeros(1,2*N);
mu_new_fraction=zeros(1,2*N);
samp3=zeros(1,N);
samp2=zeros(1,N);
samp1=zeros(1,N);
samp0=zeros(1,N);
ted=zeros(1,N);
val=zeros(1,2*N+1);
EOF=0;
K_EOF=nan;
%%
%%
i=0;
%%
i = i + 4;
j=1;
%% Part II
while(i<numel(TR_In))
    
    val(1)=val(end);
    for k=1:2*N
        mu_new(k)=i+mu_frac+k*mu_pre;
        mu_new_fraction(k)= mu_new(k)-floor( mu_new(k));
        index_k=floor(mu_new(k));
        if(index_k+1>N_input)
            EOF=true;
            K_EOF=k;
            break;
        end
        samp3(k)=TR_In(index_k+1);
        samp2(k) = TR_In(index_k);
        samp1(k) = TR_In(index_k-1);
        samp0(k) = TR_In(index_k-2);
        val(k+1)=  CubicInterpolation(samp0(k),samp1(k),samp2(k),samp3(k),mu_new_fraction(k));
    end
    if(EOF)
        n_thread=floor((K_EOF-1)/2);
    end
    for k=1:n_thread
        ted(k)=real(conj(val(2*k))*(val(2*k-1)-val(2*k+1)));
        TR_Out2(j,1)=val(2*k);
        j=j+1;
        TR_Out2(j,1)=val(2*k+1);
        j=j+1;
    end
    ted_final=mean(ted(1:n_thread));
    
    jitter = N*a*ted_final + b*tedAcc;
    
    tedAcc = tedAcc+ted_final;
    mu_pre = sps2 + jitter;
    mu_frac=mu_new_fraction(2*N);
    muAcc =muAcc + mu_pre;
    if(EOF)
        break;
    end
    i=floor(mu_new(2*N));
    
end



