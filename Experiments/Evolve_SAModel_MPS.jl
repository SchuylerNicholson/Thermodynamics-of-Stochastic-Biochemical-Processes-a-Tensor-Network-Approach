using ITensorMPS
using ITensors
using HDF5
using JLD2

# A version of the SA model used in uncertainty, but built around ITensors
# This function will evolve the system at a fixed volume from the initial conditon of [n^M = ⌊nAV⌋, 0, 0, 0]

include("Operators.jl")
include("SA_Mechanisms.jl")
include("SA_Functions.jl")
include("Thermo_Observables.jl")

VolVal = 3
VolScale = collect(range(1e-23,5e-22,50))
Vol = VolScale[VolVal]
nAV = 6.02214076e23*Vol
noM = Int64(floor(nAV))

M = noM+5;
n = M+1
global st = 1
MaxBD = 50      # MaxBD=30 works well up to Vol=1e-22
ExactValues = 0
SaveData = 0
to = 2
pokw = "MaxM"  # MaxM, Uniform
L = 4
dt = 0.005
T = 4     # Total simulation time
tLen = Int64(floor(T/dt));
@show tLen
dtObsStep = Int64(floor(tLen/160));    # This is to keep all data the same length independent of time step
@show dtObsStep

# The name of the file where observables are saved
FileOut = "SA_Thermo_Vol"*"$VolVal"*"_"*"$pokw"*"_Mol"*"$noM"*"_Time"*"$tLen"*"_MaxBD"*"$MaxBD"*"_.h5"
TSave = collect(0:dt*dtObsStep :dt*tLen)

# Keep track of how many save steps you have had 
SLen = Int64(floor(tLen/dtObsStep)+1)
cF = 2.0        # Concentration of Fuel
cW = 1.0        # Concentration of waste

nF = cF*nAV
nW = cW*nAV
 # ̂n = [1,2,3,4] = [M,M*,A_2*,A_2]
# Rxns:--- FWRD -------------- REV -----------
#       F+M -> M*,          M* -> F + M 
#       M + W -> M*,        M* -> M + W
#       2M* ->A*_2,         A*_2 -> 2M*
#       A*_2 -> A_2 + 2F,   A_2 + 2F -> A_2^*
#       A*_2 -> A_2 + 2W,   A_2 + 2W -> A*_2     
#----------------------------------------------

k = [5, 3.63e-2, 1e-3, 6.06e-2, 1, 2.27, 1, 1.74e-2, 5, .124, .1, 4.81e-4]
Keq = log( (k[2]*k[3])/(k[1]*k[4]))
k[7] = k[8]*exp(2*Keq - log(k[10]/k[9]))
c = [k[1]/nAV, k[2], k[3]/nAV, k[4] ,k[5]/nAV ,k[6],k[7] ,k[8]/nAV^2 ,k[9] ,k[10]/nAV^2,k[11],k[12]/nAV];

# Energies ------- E = [EM,EM*,EA*_2,EA_2,EF,EW]
EM=1    # Removed spaces so sed wont change EM -> M
EW=1
logc1 = log(c[2]/c[1]); logc2 = log(c[4]/c[3]); logc3 = log(c[6]/c[5]); logc4 = log(c[8]/c[7]); logc5 = log(c[10]/c[9]);

EF = logc2 - logc1 + EW
EA2 = logc3+logc4 + 2*logc1 + 2*EM
EA2s = 2*EF + 2*EM + 2*logc1+logc3
EMs = logc1 + EM + EF;

# -----------------------
mu = [EF + log(nF), EW + log(nW)]
E = hcat(range(1,6,6), [EM,EMs,EA2s,EA2,EF,EW])  # 1st column denotes the site energies, 2nd column are corresponding values.
# Stochiometric matrix for chemostated species
UChemo = [-1 1  0 0 0 0  2 -2 0 0;    # F (fuel)
           0 0 -1 1 0 0  0  0 2 -2]    # W (waste)
nChemo = [nF,nW]

# Define the observables ----------------------------------------
psiProb = zeros(SLen)        # Total probability of psi
Qdot = zeros(L,SLen)         # Store rate of heat flux
Ren2 = zeros(2,SLen)         # 1st row is Renyi2 entropy, 2nd row is Renyi-2 entropy rate.
S_Exact = zeros(2,SLen)      # 2st row is exact entropy, 2nd row is exact entropy rate.
WdotChem = zeros(SLen)       # Find the total chemical work.
Sdoti = zeros(SLen)          # Store entropy rate of Shannon entropy
Sdot_Fac = zeros(SLen)       # Store entropoy rate due to combinatorial contribution
ExMol = zeros(L,SLen)        # Store average number of molecuels
SweepTimes = zeros(SLen,2)   # 1st column is time to calculate exact Shannon rate, 2nd column is approximation using Renyi-2 Rate

ITensors.space(::SiteType"S_Class") = M+1   # Space for dynamical variables
Operators(M)    # Define the second of second quantized operators to be used

sites = siteinds("S_Class",L)
psi = Initial_MPS(pokw,L,sites,MaxBD,noM)

orthogonalize!(psi,1)
H = SA_Mechanisms(sites,L,M,"Energy_Storage",c,nChemo)  # Build the MPO for the system Hamiltonian
HOperators = SA_Mechanisms(sites,L,M,"EStorage_NoEscRate",c,nChemo)
psi0 = deepcopy(psi)


# Save out the first data point --------------------------------------------------------------------
    psiOut0 = tdvp(H::MPO,dt,psi::MPS;cutoff=1e-12,outputlevel=0,nsite=1)
    psi .= psiOut0
    psiProb[st] = Contract_One_Vec(psi::MPS,n::Int64)
    ExMol[:,st] = Expected_Molecules(psi::MPS,sites::Vector{Index{Int64}},n::Int64)
    Qdot[:,st] = Heat_Flux(E[1:4,:],H::MPO,psi::MPS,sites::Vector{Index{Int64}},n::Int64)
    Ren2[:,st] = Renyi_Two_Entropy(psi::MPS,H::MPO,n::Int64)          
    WdotChem[st] = ChemicalWork(psi::MPS,HOperators,sites::Vector{Index{Int64}},UChemo::Matrix{Int64},mu::Vector{Float64})
    Sdot_Fac[st] = Combinatorial_Entropy(H::MPO,psi::MPS,n::Int64,sites::Vector{Index{Int64}})
    if ExactValues == 1
        # Exact Shannon entropy and rate
        psidot0 = apply(H::MPO,psi::MPS,cutoff=1e-15)
        S_Exact[:,st] = Exact_Entropy_VerII(psi::MPS,psidot0::MPS,sites::Vector{Index{Int64}},n::Int64,L::Int64)
    end
    st += 1

for t = to:tLen+1
    psiOut = tdvp(H::MPO,dt,psi::MPS;cutoff=1e-12,outputlevel=0,nsite=1)
    psi .= psiOut
    # We only need measure observables every dtObsStep
    
    if  mod(t,dtObsStep ) == 0  
        psiProb[st] = Contract_One_Vec(psi::MPS,n::Int64)

        # Calculate the expected number of molecules
        ExMol[:,st] = Expected_Molecules(psi::MPS,sites::Vector{Index{Int64}},n::Int64)

        # Calculate the heat flux ----
        Qdot[:,st] =  Heat_Flux(E[1:4,:],H::MPO,psi::MPS,sites::Vector{Index{Int64}},n::Int64)
        
        # Calculate the 2-Renyi Entropy
        Ren2Time = @elapsed begin
            Ren2[:,st] = Renyi_Two_Entropy(psi::MPS,H::MPO,n::Int64)
        end
        
        # Calculate the chemical work ---
        WdotChem[st] = ChemicalWork(psi::MPS,HOperators,sites::Vector{Index{Int64}},UChemo::Matrix{Int64},mu::Vector{Float64})

        # Combinatorial Entropy terms
        Sdot_Fac[st] = Combinatorial_Entropy(H::MPO,psi::MPS,n::Int64,sites::Vector{Index{Int64}})
        
        Ren1Time = 0
        if ExactValues == 1
            # Calculate the exact Entropy and Entropy rate
            Ren1Time = @elapsed begin
                psidot = apply(H::MPO,psi::MPS,cutoff=1e-8,maxdim=MaxBD)
                S_Exact[:,st] = Exact_Entropy(psi::MPS,psidot::MPS,sites::Vector{Index{Int64}},n::Int64,L::Int64)
            end

            # Calculate the exact dissipation
             if noM <= 6
               Sdoti[st] = Exact_EPR_VII(psi::MPS,sites::Vector{Index{Int64}},L::Int64,n::Int64,c::Vector{Float64},nChemo)
            end
        end
      
        SweepTimes[st,:] = [Ren1Time,Ren2Time]
        # Save out the MPS   
        global st += 1
    end
end

WChk = Sdot_Fac + S_Exact[2,:]+ WdotChem - sum(Qdot,dims=1)';
WRen2 = Sdot_Fac +Ren2[2,:] + WdotChem - sum(Qdot,dims=1)';

if SaveData == 1

   # PathOut= "Add path here "

    fo = h5open(PathOut*FileOut,"w")
    write(fo,"TotalProb",psiProb)
    write(fo,"ExpMol",ExMol)
    write(fo,"Qdot",Qdot)
    write(fo,"Renyi_Two",Ren2)
    write(fo,"WdotChem",WdotChem)
    write(fo,"Sdot_Fac",Sdot_Fac)
    write(fo,"TSave",TSave)
    write(fo,"LoopTimes",SweepTimes)
    write(fo,"dt",dt)
    if ExactValues == 1
        write(fo,"S_Exact",S_Exact)
        write(fo,"Sdoti",Sdoti)
    end
    close(fo)
end
# ===============================================================================


