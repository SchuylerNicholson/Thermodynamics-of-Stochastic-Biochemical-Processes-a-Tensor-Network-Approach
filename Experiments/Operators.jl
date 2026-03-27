function Operators(M)
 # S_Class operators -----------------
 @eval function ITensors.op!(Op::ITensor,
    ::OpName"NS",
    ::SiteType"S_Class",
    s::Index)
    for a=1:M
        Op[s'=>a+1,s=> a+1] = a
    end
end

  @eval function ITensors.op!(Op::ITensor,
    ::OpName"a",
    ::SiteType"S_Class",
    s::Index)
    for a=1:M
        Op[s'=>a,s=> a+1] = a
    end
end
@eval function ITensors.op!(Op::ITensor,
    ::OpName"adag",
    ::SiteType"S_Class",
    s::Index)
    for a=1:M
        Op[s'=>a+1,s=>a] =1
    end
end

@eval function ITensors.op!(Op::ITensor,
    ::OpName"ProjMc",
    ::SiteType"S_Class",
    s::Index)
    for a=1:M
        Op[s'=>a,s=>a] =1
    end
end

@eval function ITensors.op!(Op::ITensor,
    ::OpName"ProjMc2",
    ::SiteType"S_Class",
    s::Index)
    for a=1:M-1
        Op[s'=>a,s=>a] =1
    end
end

@eval function ITensors.op!(Op::ITensor,
    ::OpName"ProjM",
    ::SiteType"S_Class",
    s::Index)
    Op[s'=>M+1,s=>M+1] =1
end

# Build the log n! operator

@eval function ITensors.op!(Op::ITensor,
    ::OpName"lognFac",
    ::SiteType"S_Class",
    s::Index)
    if n <= 10
        for a=1:M+1
            Op[s'=>a,s=>a] =log(factorial(a-1))
        end
    else
        nVec = collect(10:M)
        Stir_nVec = collect(broadcast(log,2*pi*nVec)./2 + nVec.*broadcast(log,nVec) - nVec)
        OpDiag = vcat(collect(broadcast(log,broadcast(factorial,collect(0:9))) ), Stir_nVec)

        for a=1:M+1
            Op[s'=>a,s=>a] = OpDiag[a]
        end
    end
end
       # nVec = collect(10:M+1)
        #Stir_nVec = log(2*pi*nVec)./2 + nVec.*broadcast(log,nVec) - nVec;
      #  OpDiag = [broadcast(log,broadcast(factorial,collect(0:9))) Stir_nVec]
      #=
        for a = 1:M+1
            Op[s'=>a,s=>a] = log(factorial(a-1)) 
            #OpDiag[a]
        end
    end
    =#

# Chemostated species operators
# S_Class operators -----------------
# EXAMPLES-----------

#=@eval function ITensors.op!(Op::ITensor,
    ::OpName"c",
    ::SiteType"ChemStat",
    s::Index)
    for a=1:MCh
        Op[s'=>a,s=> a+1] = a
    end
end
@eval function ITensors.op!(Op::ITensor,
    ::OpName"cdag",
    ::SiteType"ChemStat",
    s::Index)
    for a=1:MCh
        Op[s'=>a+1,s=>a] =1
    end
end

@eval function ITensors.op!(Op::ITensor,
    ::OpName"ProjMChc",
    ::SiteType"ChemStat",
    s::Index)
    for a=1:MCh
        Op[s'=>a,s=>a] =1
    end
end

@eval function ITensors.op!(Op::ITensor,
    ::OpName"ProjMCh",
    ::SiteType"ChemStat",
    s::Index)
    Op[s'=>MCh+1,s=>MCh+1] =1
end

@eval function ITensors.op!(Op::ITensor,
    ::OpName"NCh",
    ::SiteType"ChemStat",
    s::Index)
    for a=1:MCh
        Op[s'=>a+1,s=> a+1] = a
    end
end
=#
end