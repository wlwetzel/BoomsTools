module julia_mcmc
using LinearAlgebra
using Revise
includet("load.jl")
using ..load
using AdaptiveMCMC
using Turing 
using PlotlyJS
using Random
using Distributions
using StatsPlots
using FillArrays
using Printf
using Optim
using SpecialFunctions

halfSize = 7.1
function logPrior(x::Number,y::Number)
    halfSize = 7.1
    if x<-halfSize || y<-halfSize
        return -Inf 
    end
    if x > halfSize || y> halfSize
        return -Inf
    end
    return 0
    
end

function logPrior(dat::Vector{Float64})
    logPrior(dat[1],dat[2])
end

function logPoisson(x::Number,y::Number,pmtVals::Vector,imager)
    result::Float64 = 0
    energy = sum(pmtVals)

    LRF = load.getLRFVal(x,y,energy,imager) #total Energy version
    # LRF = load.getLRFVal(x,y,pmtVals,imager) #fractional energy version
    mui = ( energy /sum(LRF) ) .* LRF
    # print(mui)
    for i in 1:4
        result += -.5 * (pmtVals[i] - mui[i])^2 / mui[i] - .5*log(mui[i])
    end
    # for i in 1:4
    #     gaussVal = load.gaussLinearFit(x,y,i,energy)
    #     result += pmtVals[i] * log( gaussVal) - gaussVal #- (SpecialFunctions.loggamma(pmtVals[i]))
    #     # result += -.5 * (pmtVals[i] - gaussVal)^2 / gaussVal - .5*log(gaussVal)
    # end
    return result
end


function logPoisson(x::Number,y::Number,pmtVals::Matrix,imager)
    rows = size(pmtVals)[1]
    results = Vector{Float64}(undef , rows)
    for i in 1:rows
        results[i] = logPoisson(x,y,pmtVals[i,:],imager)
    end
    return sum(results)
end


function logProbPoisson(pmtVals::Vector,imager)
    function log_p(x::Vector{Float64})
        # return  logPoisson(x[1],x[2],pmtVals)
        return logPrior(x[1],x[2]) + logPoisson(x[1],x[2],pmtVals,imager)
    end
end



function logProbPoisson(pmtVals::Matrix,imager)
    function log_p(x::Vector{Float64})
        return logPrior(x[1],x[2]) + logPoisson(x[1],x[2],pmtVals,imager)
    end
end

function MLE(data::Matrix{Float64},imager)
    
    total = size(data)[1]
    resultsArray = Array{Float64}(undef,total,3)
    bins = range(-7.1,7.1,load.histSize)
    acceptedIndices = []
    halfSize = 7.1
    lower = [-halfSize;-halfSize]
    upper = [halfSize;halfSize]
    for i=1 : total 
            # f = logProbPoissonHardCoded(data[i,:])
            f = logProbPoisson(data[i,:],imager)

            g(x) = -1 * f(x)
            progress= (i / total * 100)
            # print("Progress: $(@sprintf("%.2f",progress)) % \r")
            res = Optim.optimize(g,lower,upper,[0.0 ;0.0])
            # println(res)
            # res = Optim.optimize(g,[0.0 ;0.0])
            pos = Optim.minimizer(res)
            # print(res)
            # xInd = searchsortedfirst(bins,pos[1])
            # yInd = searchsortedfirst(bins,pos[2])
            # println(xInd)
            # println(yInd)
            likelihood = -1 * Optim.minimum(res)
            # println(likelihood)
            # println(load.likelihoodMatrix[xInd,yInd])
            # println()
            # if likelihood > load.likelihoodMatrix[xInd,yInd]
                resultsArray[i,1:2] = pos
                resultsArray[i,3] = likelihood
                # push!(acceptedIndices,i)
            # end
    end
    return resultsArray
    # return resultsArray
end


function likelihoodMatrix(data::Matrix{Float64})
    posSteps = -halfSize:.2:halfSize
    arraySize = size(posSteps)[1]
    resultsArray = Array{Float64}(undef,arraySize*arraySize)
    total = size(data)[1]
    for i=1:total
        progress= (i / total * 100)
        print("Progress: $(@sprintf("%.2f",progress)) % \r")

        prob = [exp(logPoisson(x,y,data[i,:])) for x in posSteps for y in posSteps]
        resultsArray .+= (prob / sum(prob))
    end
    return reshape(resultsArray,(arraySize,arraySize))
end



end