# Collection of functions to be used with Evolve_SAModel_MPS.jl
#
function Initial_MPS(kw,L,sites,MaxDim, maxM=Int64(floor(nAV)))
    # ExtraComp is an optional set of variables that are needed for kw = MaxM
    psi = MPS(L)
    nLat = broadcast(dim,sites)
    
# Different initial MPS to start from
    if cmp(kw,"Uniform") == 0
        minDim = maximum(nLat)
        U = ones(n)./n
        psi[1] = ITensor(U,sites[1])
        for l = 2:L
            Ten = psi[l-1]*ITensor(U,sites[l])
            u,s,v = svd(Ten,(sites[l-1],commoninds(psi[l-1],Ten)),cutoff=0,mindim=n,maxdim=MaxDim)
            psi[l-1] = u 
            psi[l] = s*v
        end
    end
    if cmp(kw,"Allp") == 0
        p = reshape(collect(range(1e-8,1,n^4)),n,n,n,n)
        p = ITensor(p,sites)
        u,s,v = svd(p,sites[1],cutoff=0,mindim=n,maxdim=MaxDim)
        psi[1] = u;
        psi[2] = s*v
        for l = 2:L-1
            u,s,v = svd(psi[l],(sites[l],commoninds(psi[l-1],psi[l])),cutoff=0,mindim=n,maxdim=MaxDim)
            psi[l] = u;
            psi[l+1] = s*v
        end
    end
    if cmp(kw,"MaxM") == 0
        # ̂n = [1,2,3,4] = [M,M*,A_2*,A_2]
        nM = zeros(nLat[1])
        nM[maxM+1] = 1;
        psi[1] = ITensor(nM,sites[1])
        for l = 2:L
            NotM = zeros(nLat[l])
            NotM[1] = 1
              Ten = psi[l-1]*ITensor(NotM,sites[l])
             u,s,v = svd(Ten,(sites[l-1],commoninds(psi[l-1],Ten)),cutoff=0,mindim=n,maxdim=MaxDim)
            psi[l-1] = u 
            psi[l] = s*v
        end
# ----------------------------------------
    end
    return psi
end
# ----------------------------------------------------
function OneMPS(n,sites)
    # Build an MPS of all ones
    L = length(sites)
    OMps = MPS(L)
    OMps[1]= ITensor(ones(n),sites[1])
    for l = 2:L 
        OMps[l] = ITensor(ones(n),sites[l])
        u,s,v = svd(OMps[l-1]*OMps[l],commoninds(OMps[l-1]),0)
        OMps[l-1] = u;
        OMps[l] = s*v
    end
    return OMps 
end
# ----------------------------------------------------

function Contract_One_Vec(psi::MPS,n::Int64)
    # Read in an MPS and calculate <1|ψ>
    sites = siteinds(psi)
    L = length(sites)
    One = ones(n)
    Prob = ITensor(One,sites[1])*psi[1]
    for l = 2:L 
        Prob *= ITensor(One,sites[l])*psi[l]
    end
    return scalar(Prob)
end
# ----------------------------------------=-----------------------------------------------
function Apply_OneSite_Operator(OP,site,psi)
    # Apply operator OP to site in psi
    # First apply OP to site
    phi = deepcopy(psi)
    phi[site] *= OP
    noprime(phi[site])
    Obs = Contract_One_Vec(phi::MPS,n::Int64)
    return Obs
end
# ---------------------------------------------------------------------------------------
function Expected_Number_Molecules(phi,sites)
    L = length(sites)
    ExMol = zeros(L)
    for l = 1:L
        ExOp = op("NS",sites[l])
        ExMol[l] = Apply_OneSite_Operator(ExOp,l,phi)
    end
    return ExpMol 
end
# ---------------------------------------------------------------------------------------
function Gen_MPS2MPO_Tensor(site,l)
    # Given a site index from a tensor generate the rank-3 tensor such that we embed the MPS site into an MPO site
        n = dim(site)
        DTen = ITensor(dag(site),dag(site'),site'')
        for k = 1:n
            DTen[site=>k,site'=>k,site''=>k] = 1
        end
    
    return DTen
end
#------------------------------------------------------------------------------
function MPS2MPO(psi::MPS)
    # Given an MPS convert it to an MPO
    L = length(psi)
    sites = siteinds(psi);
    MPO_Out = MPO(L)
    # Build DTen for 1:L
    for l = 1:L
        DTen = Gen_MPS2MPO_Tensor(sites[l],1)
        MPO_Out[l] = prime(psi[l]*DTen,tags="Site",-1)
    end
    return MPO_Out
end
# -----------------------------------------------
function Sample_Tensor(T::ITensor)
    # Given a tensor T which represents a conditional distribution, sample one value from the tensor
    r = rand(Float64)
    T ./= sum(T)
    sind = ind(T,1)     # Get the first (and only) index of Ten
    N = sind.space
    c = 1               # increment counting the steps through Ten
    CDF_T = 0.0
# Quit when you find the first r s.t. r < CDF(T)
    while CDF_T <= r
        CDF_T += scalar( onehot(sind=>c)*T)

        if CDF_T < r
            c += 1
            if c == N
                CDF_T += 1
            end

        end
    end
    return c
end
#---------------------------------------------------------------
function Reduced_Marginal(psi::MPS,LMax::Int64,sites)
    # Given a joint distribiution psi, reduce it to a marginal distribution from sites 1:LMax
    L = length(sites)
    psiMarg = MPS(LMax)
    if LMax < L 
        # If LMax < L then we reduce the marginal distribution
        for k=1:LMax
            psiMarg[k] = psi[k]
        end
        TempTens = psi[L]*ITensor(ones(sites[L].space),sites[L])
        for l = L-1:-1:LMax+1
            TempTens *= psi[l]
            TempTens *= ITensor(ones(sites[l].space),sites[l])
        end

        psiMarg[LMax] *= TempTens
    else
        # If LMax == L, then we return the input distribution
        psiMarg = psi
    end
    return psiMarg 
end

#--------------------------------------------------------------------------------
# SAMPLING FUNCTIONS ----------------------------
function SampleStateVec(psi::MPS)
    # Step through psi and generate one conditional sample, MicroStateVec
    sites = siteinds(psi)
    L = length(sites)
    MicroStateVec = MPS(L)

    # function to calculate conditional distribution
    function Calc_ConditionalDist(psi::MPS,MicroStateVec::MPS)
        CondDist = psi[1]*MicroStateVec[1]
        MargLen = length(psi)
        for l = 2:MargLen-1
            CondDist *= psi[l]*MicroStateVec[l]
        end
        CondDist *= psi[MargLen]
    end
    #------------------------------------------------------------------
    # Start with the first element
    PrOne = prod(Reduced_Marginal(psi::MPS,1::Int64,sites))
    # Now sample this marginal
    MicroStateVec[1] = onehot(sites[1]=>Sample_Tensor(PrOne::ITensor))

    #Now we need to loop from 2 to L 
    for l = 2:L 
        # Generate marginal from 1:l 
        Prl = Reduced_Marginal(psi::MPS,l::Int64,sites)
        # Contract Prl with MicroStateVec[1:l-1] to generate Pr(l:l-1:1)
        CondDist = Calc_ConditionalDist(Prl::MPS,MicroStateVec::MPS)
        # Now we need to sample CondDist to get the new element to the microstate
        MicroStateVec[l] = onehot(sites[l]=>Sample_Tensor(CondDist::ITensor))
    end
    return MicroStateVec
end
#---------------------------------------------------------------------------------
function List_ProjectorCombinations(M::Int64,L::Int64)
    # Write out all combinations of N and Q operators up to length L
    # leaving out the 0,0,0.... vector which stands for all N's
    CombVec = zeros(M^L,L)
    St = 1:1:M
    TempVec = zeros(L)
    #TempVec = ones(L)
    CombVec[1,:] =  TempVec
    col = 1

    for i = 2:(M)^L
        if TempVec[1] < M-1
            TempVec[1] += 1
        elseif TempVec[col] == (M-1) && col < L
            PossIndexes = findall(<(TempVec[1]),TempVec)
            NewCol = PossIndexes[1]
            TempVec[1:NewCol-1] = zeros(NewCol-1)
            TempVec[NewCol] += 1
            col = 1
        end
        CombVec[i,:] = TempVec;
    end
    # I think we need to flip StVec left -> right
#    reverse(StVec,dims=2)
    CombVec
end
#--------------------------------------------------------------------------------------
function BuildEnvironment_Tensors(H::MPO,psi::MPS,sites::Vector{Index{Int64}},n::Int64)
    # Construct the environment tensors starting from Right and moving Left up to 2,
    L = length(sites)
    EnviroTens = Vector{ITensor}(undef,L)
    EnviroTens[L] = ITensor(ones(n),sites[L])'*H[L]*psi[L]
    for l = L-1:-1:2
        Env_l = psi[l]*EnviroTens[l+1]
        Env_l *= H[l]
        Env_l *= ITensor(ones(n),sites[l])'
        EnviroTens[l] = Env_l
    end
    return EnviroTens
end
#--------------------------------------------------------------------------------------
