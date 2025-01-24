
function [PLL_Out, phiBegin, phiEnd] = PLL_V3_EFB(PLL_In , pow , PLL_Alpha , PLL_Betta,nIteration)

PLL_Out         = zeros(size(PLL_In)) ;


PLL_PED_Vec     = 0 ;
PLL_PED_ACC_Vec = 0 ;
PLL_Phi_Vec     = 0 ;

Phi = 0 ;
iter =1;
for k = 1:nIteration
while iter <=length(PLL_Out)
    PLL_Out(iter)     = PLL_In(iter) * Phi ;
    PLL_Out_pow       = PLL_Out(iter)^pow ;
    PLL_Out_pow_HD    = (sign(real(PLL_Out_pow)) + 1j*sign(imag(PLL_Out_pow))) / sqrt(2) ;
    PLL_Costas_Eq     = -imag(PLL_Out_pow * conj(PLL_Out_pow_HD)) ;
    PLL_PED_Vec = PLL_Costas_Eq / pow ;
    
    PLL_Jitter            = 2^-PLL_Alpha * PLL_PED_Vec + 2^-PLL_Betta * PLL_PED_ACC_Vec ;
    
    
    PLL_PED_ACC_Vec = PLL_PED_ACC_Vec + PLL_PED_Vec ;
    PLL_Phi_Vec     = PLL_Phi_Vec     + PLL_Jitter ;
    
    if     PLL_Phi_Vec >= 1
        PLL_Phi_Vec = PLL_Phi_Vec - 1 ;
    elseif PLL_Phi_Vec < 0
        PLL_Phi_Vec = PLL_Phi_Vec + 1 ;
    end
    
    Phi = exp(1j*2*pi*PLL_Phi_Vec) ;
    iter = iter+1;
end
phiEnd = PLL_Phi_Vec;

%% set initial conditions
PLL_Phi_Vec     =  PLL_Phi_Vec - 2 * PLL_Jitter;
PLL_PED_ACC_Vec = -PLL_PED_ACC_Vec;
iter = iter-2;

while iter >=1
    PLL_Out(iter)     = PLL_In(iter) * Phi ;
    PLL_Out_pow       = PLL_Out(iter)^pow ;
    PLL_Out_pow_HD    = (sign(real(PLL_Out_pow)) + 1j*sign(imag(PLL_Out_pow))) / sqrt(2) ;
    PLL_Costas_Eq     = -imag(PLL_Out_pow * conj(PLL_Out_pow_HD)) ;
    PLL_PED_Vec = PLL_Costas_Eq / pow ;
    
    PLL_Jitter = 2^-PLL_Alpha * PLL_PED_Vec + 2^-PLL_Betta * PLL_PED_ACC_Vec ;
    
    
    PLL_PED_ACC_Vec = PLL_PED_ACC_Vec + PLL_PED_Vec ;
    PLL_Phi_Vec     = + PLL_Phi_Vec     + PLL_Jitter ;
    
    if     PLL_Phi_Vec >= 1
        PLL_Phi_Vec = PLL_Phi_Vec - 1 ;
    elseif PLL_Phi_Vec < 0
        PLL_Phi_Vec = PLL_Phi_Vec + 1 ;
    end
    
    Phi = exp(1j*2*pi* PLL_Phi_Vec) ;
    iter = iter-1;
end

phiEnd = PLL_Phi_Vec;

%% set initial conditions
PLL_Phi_Vec     =  PLL_Phi_Vec - 2 * PLL_Jitter;
PLL_PED_ACC_Vec = -PLL_PED_ACC_Vec;
iter =2;
end
phiBegin = PLL_Phi_Vec(1);

