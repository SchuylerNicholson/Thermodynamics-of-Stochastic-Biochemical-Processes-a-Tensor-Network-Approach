# A collection of functions that calculate various thermodynamic observables such as:
# heat flux, exact entropy rate, Renyi-2 entropy, Renyi-2 entropy rate, exact entropy production rate etc.

# Note: Calls from SA_Functions.jl

function One_Site_Contraction(psi::MPS,Oper::ITensor,Osite::Int64,n::Int64)
    # Contract an MPS with the one site operator at site Osite, 
    sites = siteinds(psi::MPS)
    L = length(sites)
    orthogonalize!(psi::MPS,1)
    
    O1 = ITensor(ones(n),sites[1])
    if Osite == 1
        Obs = prime(O1)*Oper*psi[1]
    else
        Obs = O1*psi[1]
    end
    for l = 2:L 
        Ol = ITensor(ones(n),sites[l])
        Obs *= psi[l]
        if l == Osite 
            Obs *= Oper
            Obs *= prime(Ol)
        else
            Obs *= Ol 
        end
    end
    return scalar(Obs)
end
# ----------------------------------------=-----------------------------------------------
function Inner_Operator(psiL::MPS,psiR::MPS,Oper::ITensor,n)
    # Form the contraction <ψl|̂o|ψr> where ̂o is a multi site operator
    # Note, psiL and psiR must have the same site indices
    sites = siteinds(psiR)
    L = length(sites)
    FndObsSite = 0  # We need to track if we have contacted over an observable index yet
    if hasind(Oper,sites[1]) == true
        Obs = prime(psiL[1])*Oper*psiR[1]
        FndObsSite = 1
    else
        Obs = prime(psiL[1],tags="Link")*psiR[1]
    end
    for l = 2:L
        if hasind(Oper,sites[l]) == true
            if FndObsSite == 0 
                Obs *= prime(psiL[l]) # This could probably be made more efficient by breaking up the contraction into separate pieces
                Obs *= psiR[l]
                Obs *= Obs
                
                FndObsSite += 1
            else
                Obs *= prime(psiL[l])
                Obs *= psiR[l]
            end
        else
            Obs *= psiR[l]
            Obs *= prime(psiL[l],tags="Link")
        end
    end
    return scalar(Obs)
end
# ----------------------------------------=-----------------------------------------------
#=
function Heat_Flux(E,psidot::MPS,sites::Vector{Index{Int64}},n::Int64)
    # Calculates the heat flux for each site in E[:,1]
    ELen = size(E,1)
    One = ones(n)
    OMPS = OneMPS(n,sites)
    Qdot = zeros(ELen)
    for l = 1:ELen
        # Contract tenssor down to site l
       Oper = E[l,2]*op("NS",sites[ Int64(E[l,1]) ])
       Qdot[l] = One_Site_Contraction(psidot::MPS,Oper::ITensor,l::Int64,n::Int64)
    end
    return Qdot
end
=#
#------------------------------------------------------------------------------------------
function Heat_Flux(E,H::MPO,psi::MPS,sites::Vector{Index{Int64}},n::Int64)
#function Heat_Flux_VII(E,H::MPO,psi::MPS,sites::Vector{Index{Int64}},n::Int64)    

    # Calculate the heat flux taking advantage of the environment tensors, we will
    # sweep from left to right
    ELen = size(E,1)
    Qdot = zeros(ELen)
    # Build the environment tensors

    Env = BuildEnvironment_Tensors(H::MPO,psi::MPS,sites::Vector{Index{Int64}},n::Int64)
    # Calculate heat flux at 1
    EnvL = psi[1]*H[1]*ITensor(ones(n),sites[1])'
    for l = 1:ELen
        EOp = E[l,2]*op("NS",sites[ Int64(E[l,1]) ])*ITensor(ones(n),sites[l])
        if l == 1
            Qdot[1] = scalar(EOp*H[1]*psi[1]*Env[2])
            Env[1] = EnvL
        elseif l < L
            Qdl = EnvL*psi[l]
             Qdl *= H[l]
            Qdl *= EOp
            # Now combine EnvL and Env[l+1]
             Qdl *= Env[l+1]
             Qdot[l] = scalar(Qdl)

            # Now build EnvL
            EnvL *= psi[l]
            EnvL *= H[l]*ITensor(ones(n),sites[l])'
        else
            QdL = EnvL*psi[L]
            QdL *= H[L]*EOp
            Qdot[L] = scalar(QdL)
        end
    end
    return Qdot
end
#------------------------------------------------------------------------------------------
function Expected_Molecules(psi::MPS,sites::Vector{Index{Int64}},n::Int64)
# Calculates the heat flux for each site in E[:,1]
    L = length(sites)
    One = ones(n)
    OMPS = OneMPS(n,sites)
    ExMol = zeros(L)
    for l = 1:L
        # Contract tenssor down to site l
        Oper = op("NS",sites[l])
        ExMol[l] = One_Site_Contraction(psi::MPS,Oper::ITensor,l::Int64,n::Int64)
    end
    return ExMol
end
#------------------------------------------------------------------------------------------
function ChemicalWork(psi::MPS,HOperators,sites::Vector{Index{Int64}},UMat::Matrix{Int64},mu::Vector{Float64})
    # Calculate the chemical work for the set of operators HOperators
    HLen = length(HOperators)
    WdotChem = zeros(HLen)
    OMPS = OneMPS(n,sites)
    for r = 1:HLen 
        UVec = UMat[:,r]
        WdotChem[r] = -sum(UMat[:,r].*mu*inner(OMPS'::MPS,HOperators[r]::MPO,psi::MPS,cutoff=1e-15))
    end
    return sum(WdotChem)
end
#------------------------------------------------------------------------------------------
function Exact_Entropy(psi::MPS,psidot::MPS,sites::Vector{Index{Int64}},n::Int64,L::Int64)
    # cycle through all states and compute the entropy rate 
    States = Int64.(List_ProjectorCombinations(n::Int64,L::Int64))
    Sdot = 0.0
    S = 0.0
    N = size(States,1)
    # Now we need to generate a function that builds each state
    for k = 1:N
        StateMPS = MPS(L)
        for l = 1:L
            StateMPS[l] = onehot(sites[l]=>States[k,l]+1)
        end
        pdotState = inner(StateMPS::MPS,psidot::MPS)
        pState = inner(StateMPS::MPS,psi::MPS)
        # Now calculate the contribution to Sdot, S 

        if pState > 1e-15
            Sdot += -pdotState*log(pState)
            S += -pState*log(pState)
        end
    end
    return [S,Sdot]
end

#------------------------------------------------------------------------------------------
function Gen_State_MPS(State,sites,L)
    StateMPS = MPS(L)
    for l = 1:L
        StateMPS[l] = onehot(sites[l]=>State[l]+1)
    end
    return StateMPS 
end
#------------------------------------------------------------------------------------------
function Exact_EPR_VII(psi::MPS,sites::Vector{Index{Int64}},L::Int64,n::Int64,c::Vector{Float64},nChemo)
    # Build the exact EPR by looping over all states
    # Note due to exponentially large state spaces this function can be EXTREMELY slow.
    States = Int64.(List_ProjectorCombinations(n::Int64,L::Int64))
    N = size(States,1)
    HLen = length(sites);
    Sdoti = 0.0
    # cycle through each pair of forward and reverse reactions
    for k = 1:5
        r = Int64(k);
        fwdStr = "H"*"$r"*"f"
        revStr = "H"*"$r"*"r"
        Hf = SA_Mechanisms(sites,L,n,fwdStr,c,nChemo)
        Hr = SA_Mechanisms(sites,L,n,revStr,c,nChemo)
        # For each pair of reactions sum over all pairs of states
        # Now given Hf, Hr, we need to extract Hf(n',n) and Hr(n,n')
        for i = 1:N 
            n_MPS = Gen_State_MPS(States[i,:],sites,L)
            pn = inner(n_MPS::MPS,psi::MPS)       # pr(n')

            # We only need to count transitions that come from non-zero probabilities
            if pn > 1e-18
                st = States[i,:]
                for j = i+1:N
                    np_MPS = Gen_State_MPS(States[j,:],sites,L)
                    pnp = inner(np_MPS::MPS,psi::MPS)
                    if pnp > 0.0               
                        #Hr = Hr+ + Hr-
                        #Hr(n',n) = Hr+(n',n) + Hr-(n',n)
                        Hrm =  inner(np_MPS'::MPS,Hf::MPO,n_MPS::MPS) + inner(np_MPS'::MPS,Hr::MPO,n_MPS::MPS)
                        #HrmTr(n,n') = Hr+(n,n') + Hr-(n,n')
                        HrmTr = inner(n_MPS'::MPS,Hf::MPO,np_MPS::MPS) + inner(n_MPS'::MPS,Hr::MPO,np_MPS::MPS)
                        if Hrm > 0 && HrmTr > 0
                            Sdoti += Hrm*pn*(log(Hrm*pn) - log(HrmTr*pnp))
                            Sdoti += HrmTr*pnp*(log(HrmTr*pnp) - log(Hrm*pn))
                        end
                    end
                end
            end
        end
    end

return Sdoti
end

#-----------------------------------------------------------------------------------------------
function Exact_EPR(psi::MPS,sites::Vector{Index{Int64}},L::Int64,M::Int64,c::Vector{Float64},nChemo)
    # An older function to calculate the exact EPR.
    # groups HOperators by counting by 2*k-1:2*k
    # Each grouping will contribute the EPR
    cb = combiner(sites)
    Pr = cb*prod(psi)
    Pr = sparse(array(Pr,inds(Pr)))
    Sdoti = 0;
    for k = 1:4
        r = Int64(k); 
        fwdStr = "H"*"$r"*"f"
        revStr = "H"*"$r"*"r" 
        Hf = SA_Mechanisms(sites,L,M,fwdStr,c,nChemo)
        Hr = SA_Mechanisms(sites,L,M,revStr,c,nChemo)

        Hfm = cb'*prod(Hf)*cb 
        Hrm = cb'*prod(Hr)*cb 

        Hr = sparse(array(Hfm,inds(Hfm,plev=1)[1],inds(Hfm,plev=0)[1])) + sparse(array(Hrm,inds(Hrm,plev=1)[1],inds(Hrm,plev=0)[1]))
       # Hr = Hfm + Hrm;
        Aff = sparse(Hr*diagm(Pr)./(Hr*diagm(Pr))')
        Aff[broadcast(isnan,Aff)] .= 0
        Aff[broadcast(isinf,Aff)] .= 0
        Affnz = findall(>(1e-18),Aff)
        Aff[Affnz] .= broadcast(log,Aff[Affnz])
        Sdoti += sum( Hr*diagm(Pr).*Aff)
    end
    return Sdoti 
end
# -----------------------------------------------------------------------
#=
function Exact_Entropy(psi::MPS,psidot::MPS,sites::Vector{Index{Int64}})
    # Read in |ψ> and |̇ψ> and calculate ̇s and s from matrices
    
    cb = combiner(sites);
    psiV = prod(psi)*cb;
    psiV = array(psiV,inds(psiV))
    pVlog = deepcopy(psiV);
    pVlog[psiV .<= 1e-15] .=1
    psidotV = prod(psidot)*cb
    psidotV = array(psidotV,inds(psidotV))

    return [-psiV'*broadcast(log,pVlog),-psidotV'*broadcast(log,pVlog)]
end
=#
# -----------------------------------------------------------------------
function Combinatorial_Entropy(psidot::MPS,n,sites::Vector{Index{Int64}})
    # Given ̇ψ calculate <1|log n!|̇ψ>
    Len = length(sites)
    Sdot_Fac = 0.0
    OMPS = OneMPS(n,sites)

    for r = 1:Len 
        Oper = op("lognFac",sites[r])
        Sdot_Fac += -One_Site_Contraction(psidot::MPS,Oper::ITensor,r::Int64,n::Int64)
    end
    return Sdot_Fac

end
# ------------------------------------------------------------------------
#=
function Renyi_Two_Entropy(psi::MPS,psidot::MPS,n::Int64)
    # Older version uses psidot, which can be costly to compute
    # Calculate both the 2-renyi entropy and the 2-renyi entropy rate
    psiSq = inner(psi,psi)
    H2 = -log(psiSq)
    H2_dot = -2*inner(psi,psidot)/psiSq
    return [H2,H2_dot]
end
=#
# -----------------------------------------------------------------------
function Renyi_Two_Entropy(psi::MPS,H::MPO,n::Int64)
    psiSq = inner(psi,psi)
    H2 = -log(psiSq)
    H2_dot = -2*inner(psi'::MPS,H::MPO,psi::MPS)/psiSq
    return [H2,H2_dot]
end
# ------------------------------------------------------------------------
function Combinatorial_Entropy(H::MPO,psi::MPS,n::Int64,sites::Vector{Index{Int64}})
    # Given ̇ψ calculate <1|log n!|̇ψ>
    Len = length(sites)
    Sdot_Fac = 0.0
    Env = BuildEnvironment_Tensors(H::MPO,psi::MPS,sites::Vector{Index{Int64}},n::Int64)

    # Calculate heat flux at 1
    EnvL = psi[1]*H[1]*ITensor(ones(n),sites[1])'
    for l = 1:Len
        Oper = op("lognFac",sites[l])*ITensor(ones(n),sites[l])
        if l == 1
            Sdot_Fac += -scalar(Oper*H[1]*psi[1]*Env[2])
            Env[1] = EnvL
        elseif l < L
            # EnvL * psi[l]
            Ol = EnvL*psi[l]
            # O[l] * H[l]
            Ol *= H[l]
            #O[l] * ̂O *|1>
            Ol *= Oper
            # Now combine EnvL and EnvR[l+1]
            Ol *= Env[l+1]
            Sdot_Fac += -scalar(Ol)

            # Now build EnvL
            EnvL *= psi[l]
            EnvL *= H[l]*ITensor(ones(n),sites[l])'
        else
            OL = EnvL*psi[L]
            OL *= H[L]*Oper
            Sdot_Fac += -scalar(OL)
        end
    end
    return Sdot_Fac
end
# ------------------------------------------------------------------------


    








