module load
export poissonArrayLinear,imageRotations

using CSV
using DataFrames
using StatsBase
using Distributions
using EmpiricalDistributions
using Interpolations
using ImageFiltering


histSize = 13


function line(energy,m,b)
    return m .* energy .+ b 
end

function tanhFit(energy,m,b,c)
    return m .* tanh.(b .* energy ./ 300) .+ c
end

function square(energy,m,b,c)
    return m .* energy.^2 .+ b.*energy .+c
end
imagerPoissonFitsVector = []
for imager in ['A','B','C','D','E','F','G']
    file =  "/home/wyatt/Projects/BOOMS/calibration/Brady_Data/final-moving-source/$(imager)/poisson_params/poissonFit.csv"
    push!(
        imagerPoissonFitsVector,
        DataFrame(CSV.File(file))
    )
end
poissonArrayLinear = Array{Float64}(undef,7 ,4,5,3) #(imager,pmt,lorenzParam,fitParam)
pmts = ["PMT1","PMT2","PMT3","PMT4"]
params = ["fitParam$(i)" for i in 0:2]

for k=1:7 #imager
    for i in 1:4 #pmt
        for j in 1:3 #parameter
            linearCoeffs = imagerPoissonFitsVector[k][(imagerPoissonFitsVector[k].PMT .== pmts[i]) .&& (imagerPoissonFitsVector[k].param .== params[j]),:]
            poissonArrayLinear[k,i,j,:] = [( linearCoeffs.slope) (linearCoeffs.scale) (linearCoeffs.intercept) ]

        end
    end
end


struct poissonLinearModel   
    poissonArray :: Array{Float64}
end

const PLM  = poissonLinearModel(poissonArrayLinear)


function createParams(pmt,energy,imager)
    p = Vector{Float64}(undef , 3)
    
    p[1] = PLM.poissonArray[imager,pmt,1,1]
    p[2] = line(energy, PLM.poissonArray[imager,pmt,2,1], PLM.poissonArray[imager,pmt,2,2])
    p[3] = line(energy, PLM.poissonArray[imager,pmt,3,1], PLM.poissonArray[imager,pmt,3,2])#, PLM.poissonArray[imager,pmt,2,3])

    return p
end


function lorenzModel(x::Number,y::Number,p::Vector)
	return p[1] .* ( (x .- p[2]).^2  + (y .- p[3]).^2 .+ p[4] ).^(-1.5)
end

function lorenzIntegratedModel(x,y,p,pmt)
    h =p[1]
    t=p[2]
    eff=p[3]
    x0 = x ./ t
    y0 = y ./ t
    ht = h/t	

	if pmt == 1 
        return eff .* (atan.((1 .- (ht .- x0).* (-ht .+ x0 .+ sqrt.(1 .+ (ht .- x0).^2 .+ y0.^2)))./y0) .- 
        atan.((1 .+ x0 .* (x0 .+ sqrt.(1 .+ x0.^2 .+ y0.^2)))./y0) .- 
        atan.((1 .- (ht .- x0) .* (-ht .+ x0 .+ sqrt.(1 .+ 2 .* ht.^2 .- 2 .* ht .* x0 .+ x0.^2 .+ 2 .* ht .* y0 .+ y0.^2)))./(ht .+ y0)) .+ 
        atan.((1 .+ x0 .* (x0 .+ sqrt.(1 .+ x0.^2 .+ (ht .+ y0).^2)))./(ht .+ y0)))
    elseif 	pmt==2
        return eff .* (atan.((1 .+ x0 .*(x0 .+ sqrt.(1 .+ x0.^2 .+ (ht .- y0).^2)))./(ht .- y0)) .- 
        atan.((1 .- (ht .- x0).* (-ht .+ x0 .+ sqrt.(1 .+ (ht .- x0).^2 .+ y0.^2)))./y0) .+ 
        atan.((1 .+ x0 .*(x0 .+ sqrt.(1 .+ x0.^2 .+ y0.^2)))./y0) .- 
        atan.((1 .- (ht .- x0).* (-ht .+ x0 .+ sqrt.(1 .+ 2 .*ht.^2 .+ x0.^2 .+ y0.^2 .- 2 .* ht.* (x0 .+ y0))))./(ht .- y0))) 
	elseif 	pmt==3
        return eff .* (atan.((1 .+ x0.^2 .- x0.* sqrt.(1 .+ x0.^2 .+ (ht .- y0).^2))./(ht .- y0)) .+ 
        atan.((1 .+ x0.^2 .- x0.* sqrt.(1 .+ x0.^2 .+ y0.^2))./y0) .- 
        atan.((1 .- (ht .+ x0).* (-ht .- x0 .+ sqrt.(1 .+ (ht .+ x0).^2 .+ y0.^2)))./y0) .- 
        atan.((1 .- (ht .+ x0) .* (-ht .- x0 .+ sqrt.(1 .+ 2 .*ht.^2 .+ 2 .* ht.* x0 .+ x0.^2 .- 2 .*ht.* y0 .+ y0.^2)))./(ht .- y0)))
    elseif pmt==4
        return eff .* (atan.((1 .+ x0.^2 .- x0 .* sqrt.(1 .+ x0.^2 .+ (ht .+ y0).^2))./(ht .+ y0)) .+ 
									atan.((.-1 .+ x0 .* (-x0 .+ sqrt.(1 .+ x0.^2 .+ y0.^2)))./y0) .+ 
									atan.((1 .- (ht .+ x0) .* (-ht .- x0 .+ sqrt.(1 .+ (ht .+ x0).^2 .+ y0.^2)))./y0) .- 
									atan.((1 .- (ht .+ x0) .* (-ht .- x0 .+ sqrt.(1 .+ 2 .* ht.^2 .+ x0.^2 .+ y0.^2 .+ 2 .* ht .*(x0 .+ y0))))./(ht .+ y0)))	
    end
end

energies = 1:1:500
const lrfParams =  
( 
    A = [[createParams(i,energy,1) for i in 1:4] for energy in energies ],
    B = [[createParams(i,energy,2) for i in 1:4] for energy in energies ],
    C = [[createParams(i,energy,3) for i in 1:4] for energy in energies ],
    D = [[createParams(i,energy,4) for i in 1:4] for energy in energies ],
    E = [[createParams(i,energy,5) for i in 1:4] for energy in energies ],
    F = [[createParams(i,energy,6) for i in 1:4] for energy in energies ],
    G = [[createParams(i,energy,7) for i in 1:4] for energy in energies ]

    )

function getLRFVal(x,y,energy::Number,imager)
    params=lrfParams[imager][floor(Int64,energy)]
    lrfVals = [lorenzIntegratedModel(x,y,params[i],i) for i in 1:4]
    return lrfVals
end

function getLRFVal(x,y,energy::Vector,imager)
    params = []
    for i=1:4
        if energy[i]>0
            push!(params,lrfParams[imager][floor(Int64,energy[i])][i])
        else
            push!(params,lrfParams[imager][floor(Int64,1)][i])
        end
    end
    lrfVals = [lorenzIntegratedModel(x,y,params[i],i) for i in 1:4]
    return lrfVals
end

imageRotations = 
(
    # images are rotated so that 'up' corresponds to the B coords, as well as +Z in the mag coords
    A = x-> rotr90(x) , 
    B = x-> x ,
    C = x-> rotl90(x) ,
    D = x-> rot180(x) ,
    E = x-> rotl90(x) ,
    F = x-> rotr90(x) ,
    G = x-> rotl90(x) 
)


end
