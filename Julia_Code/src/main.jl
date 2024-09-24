using Revise
includet("julia_mcmc.jl")
using Pkg 
using DataFrames
using CSV
# Pkg.activate(".")
using .julia_mcmc
using AdaptiveMCMC
using LinearAlgebra
using Printf
using StatsBase
using PlotlyJS
import Logging
includet("load.jl")
using ..load
# using ImageFiltering
using Tables
using Optim
# using Plots


Logging.disable_logging(Logging.Warn)
 
function dataLoad(file::String)
    df = DataFrame(CSV.File(file))
    PMTS = [:PMT1,:PMT2,:PMT3,:PMT4]
    df = df[sum.(eachrow(df[!,PMTS])) .> 0,:]
    df = df[sum.(eachrow(df[!,PMTS])) .< 500,:]
    df.gondola_time = (df.gondola_time .- df.gondola_time[1]) .* 10^-7
    return Float64.(df)
end

function dataLoad(file::CSV.File)
    df = DataFrame(file)  
    df = df[:,[2,3,4,5]]
    df = df[sum.(eachrow(df)) .> 0,:]
    # df = df ./ sum.(eachrow(df)) .*100
    mat = Matrix(df)
    return Float64.(mat)
end

function writeGifMLE(file,frameLength,saveName)
    data = DataFrame(CSV.File(file))  
    PMTS = [:PMT1,:PMT2,:PMT3,:PMT4]
    data = data[sum.(eachrow(data[!,PMTS])) .> 10,:]
    data = data[sum.(eachrow(data[!,PMTS])) .< 500,:]
    # data = data[(sum.(eachrow(data[!,PMTS])) .> 75) .& (sum.(eachrow(data[!,PMTS])) .< 125),:]

    edges = range(-7.1,7.1,load.histSize+1)

    duration = last(data[!,:Time]) - first(data[!,:Time])
    totalFrames = floor(duration / frameLength)
    # dark = gain.darkImg .* frameLength
    step = 14.2/load.histSize
    xVals = -7.1:step:7.1
    savePath = "/home/wyatt/Projects/BOOMS/imaging/julia_mcmc/dataToPlot/"*saveName
    if isfile(savePath)
        rm(savePath)
    end
    for j=1:totalFrames-1
        progress= (j / totalFrames * 100)
        print("Progress: $(@sprintf("%.2f",progress)) % \r")
        chunk = data[(data.Time .> ((j-1) * frameLength) ) .& (data.Time .< (j * frameLength) ) , :]
        pmtArray = Float64.(Array(chunk[!,[:PMT1,:PMT2,:PMT3,:PMT4]]))
        
        res = julia_mcmc.MLE(pmtArray)
        xs = res[:,1]
        ys = res[:,2]
        histFit =   StatsBase.fit(Histogram ,(xs,ys ),(edges,edges)) 
        imageArray = histFit.weights

        # imageArray = imageArray ./ load.getFlatFrame(mean(sum.(eachrow(data))))
        time =( j-1) * frameLength
        CSV.write( savePath,Tables.table(imageArray),append=true)
    end

end

function runMLE(file::String,imager::Int)
    data = DataFrame(CSV.File(file))  
    PMTS = [:PMT1,:PMT2,:PMT3,:PMT4]
    # data = data[(sum.(eachrow(data[!,PMTS])) .> 75) .& (sum.(eachrow(data[!,PMTS])) .< 125),:]
    time = data.Time[end] - data.Time[1]
    data = Float64.(Array(data[!,PMTS]))
    data = data[1:5:end,:]
    res = julia_mcmc.MLE(data,imager)
    xs = res[:,1]
    ys = res[:,2]
    return xs,ys,time,data
end

function plotMLE(file::String,save=false,title="",saveName="")
    xs,ys,time,data = runMLE(file)
    
    edges = range(-7.1,7.1,load.histSize+1)
    img =  StatsBase.fit(Histogram ,(xs,ys ),(edges,edges)).weights
    # img = (img .- gain.darkImg .*time ) .* gain.gainImg
    # img = img .* gain.gainImg 
    # img = img ./ load.getFlatFrame(mean(sum.(eachrow(data))))
    step = halfSize * 2 / (load.histSize)
    xs = -halfSize:step:halfSize
    xs = (xs .+ (xs[2]-xs[1]) / 2)
    trace = PlotlyJS.heatmap(x=xs,y=xs,z=img,colorscale="Cividis")
    
    # horiztraces = [PlotlyJS.scatter(y=[yVal,yVal],x=[-6,6],line_color="red",showlegend=false,mode="lines") for yVal in -6:2.0:6]
    traces = [trace]
    # append!(traces,horiztraces)
    p = PlotlyJS.plot(traces,Layout(title=title))
    # p = PlotlyJS.plot(PlotlyJS.histogram2d(x=xs,y=ys))
    if save
        print("asdfasdf")
        PlotlyJS.savefig(p,"/home/wyatt/Projects/BOOMS/imaging/plots/"*saveName*".png")
    end
    
    return p,img
end


function plotLikelihood(file::String)
    posSteps = -halfSize:.2:halfSize

    data = DataFrame(CSV.File(file))  
    PMTS = [:PMT1,:PMT2,:PMT3,:PMT4]
    data = data[(sum.(eachrow(data[!,PMTS])) .> 75) .& (sum.(eachrow(data[!,PMTS])) .< 125),:]
    data = Float64.(Array(data[!,PMTS]))
    img = julia_mcmc.likelihoodMatrix(data[1:10:end,:])
    trace = PlotlyJS.heatmap(x=posSteps,y=posSteps,z=img,colorscale="Cividis")
    plot(trace)
    return img
end

function plotMLE(files::Vector{String},title::String,imager::Int)
    xs = []
    ys = []
    for file in files
        x,y,time,data = runMLE(file,imager)
        append!(xs,x)
        append!(ys,y)
    end
    edges = range(-7.1,7.1,load.histSize+1)
    img =  StatsBase.fit(Histogram ,(xs,ys ),(edges,edges)).weights
    step = halfSize * 2 / (load.histSize)
    xs = -halfSize:step:halfSize
    xs = (xs .+ (xs[2]-xs[1]) / 2)
    trace = PlotlyJS.heatmap(x=xs,y=xs,z=img,colorscale="Cividis")
    p = PlotlyJS.plot([trace])
    PlotlyJS.savefig(p,"/home/wyatt/Projects/BOOMS/imaging/calibration_plots/"*title*".png")
    return p,img
end


function plotCalibrationData()
    for (id,imager) in enumerate(['A','B','C','D','E','F','G'])
        # ["americium","cadmium","cobalt2","cobalt1","sodium"]
        for source in ["americium","cadmium","cobalt2","cobalt1"]
            println(imager,source)
            calFiles = [ "/home/wyatt/Projects/BOOMS/calibration/Brady_Data/final-moving-source/$(imager)/$(source)_pixels/calibration_pixel_$(i).csv"
                         for i in 1:168]
            plotMLE(calFiles,"$(source)_$(imager)_calibration",id)
                
        end
    end
   

end

function plotAnger(file)
    data = DataFrame(CSV.File(file))  
    PMTS = [:PMT1,:PMT2,:PMT3,:PMT4]
    data = data[(sum.(eachrow(data[!,PMTS])) .> 0) .& (sum.(eachrow(data[!,PMTS])) .< 500),:]

    data = Float64.(Array(data[!,PMTS]))
    # xs = (data[:,1] + data[:,2]) - (data[:,3] + data[:,4])
    # ys = (data[:,2] + data[:,3]) - (data[:,1] + data[:,4])
    xWeights = [sum([3.55,-3.55,-3.55,3.55] .* row) for row in eachrow(data) ]
    yWeights = [sum([3.55,3.55,-3.55,-3.55] .* row) for row in eachrow(data) ]
    xs = xWeights ./ sum.(eachrow(data))
    ys = yWeights ./ sum.(eachrow(data))
    edges = range(-7.1,7.1,100)
    histFit =   StatsBase.fit(Histogram ,(xs,ys ),(edges,edges)) 
    PlotlyJS.plot(PlotlyJS.heatmap(z=histFit.weights))
end


function runDemos()

    basePath = "/home/wyatt/Projects/BOOMS/calibration/demos/"
    demoNames = ["americium_shadow","barium_distance","barium_shadow","barium_spiral","moving","sodium_shadow"]
    for demo in demoNames
        println(demo)
        writeGifMLE(basePath*demo*".csv",.2,demo*".csv")
    end
end    

struct Frame
    imagerResults::Vector
end

struct Simulation
    frames::Vector{Frame}
end

# struct Movie
#     timeRange


# end

function writeGifMLEAllImagers(fileNumber,frameLength,saveName,timeRange=[0,10^10])
    # basePath = "/home/wyatt/Projects/BOOMS/processed_data/imager_data/"
    basePath = "/home/wyatt/Projects/BOOMS/processed_data/imagerSubset/"
    
    allFiles = readdir(basePath;join=true)
    imagerFiles = [file for file in allFiles if occursin(string(fileNumber),file)]
    dataVec = Vector{}(undef,7)
    PMTS = [:PMT1,:PMT2,:PMT3,:PMT4]

    for i in 1:7
        tempdat = dataLoad(imagerFiles[i])
        tempdat = tempdat[(tempdat[!,:gondola_time] .> timeRange[1]) .& (tempdat[!,:gondola_time] .< timeRange[2]),:] 
        tempdat[!,:gondola_time] .-= tempdat[!,:gondola_time][1]
        dataVec[i] = tempdat
        println("Loaded $(i)")
    end


    duration = last(dataVec[1][!,:gondola_time]) - first(dataVec[1][!,:gondola_time])
    edges = range(-7.1,7.1,load.histSize+1)
    totalFrames = floor(duration / frameLength)
    savePath = "/home/wyatt/Projects/BOOMS/imaging/imageArrays/"*saveName
    
    if isfile(savePath)
        rm(savePath)
    end

    sim = Simulation(Vector{Frame}())
    for j=1:totalFrames-1
        progress= (j / totalFrames * 100)
        print("Progress: $(@sprintf("%.2f",progress)) % \r")
        writeArray = Array{Float64}(undef,load.histSize,load.histSize*7)
        resultVec = []
        for i in 1:7
            chunk = dataVec[i][(dataVec[i][!,:gondola_time] .> ((j-1) * frameLength) ) .& (dataVec[i][!,:gondola_time].< (j * frameLength) ) , :]
            pmtArray = Float64.(Array(chunk[!,[:PMT1,:PMT2,:PMT3,:PMT4]]))
            
            push!(resultVec,  
                   mean( 
                    pmtArray./(sum(pmtArray,dims=2)) , dims=1)
                    )

            res = julia_mcmc.MLE(pmtArray,i)
            xs = res[:,1]
            ys = res[:,2]
            histFit =   StatsBase.fit(Histogram ,(xs,ys ),(edges,edges)) 
            imageArray = rotl90(histFit.weights)

            imageArray = load.imageRotations[i](imageArray) #need to rotate imagers to match gondola config

            writeArray[:,1 + (i-1)*load.histSize:(i)*load.histSize] .= imageArray
        end
        frame = Frame(resultVec)
        push!(sim.frames,frame)
        CSV.write( savePath,Tables.table(writeArray),append=true)
    end
    return sim
end

function makeBackground(fileNumber,saveName)
    # basePath = "/home/wyatt/Projects/BOOMS/processed_data/imager_data/"
    basePath = "/home/wyatt/Projects/BOOMS/processed_data/imagerSubset/"
    
    allFiles = readdir(basePath;join=true)
    imagerFiles = [file for file in allFiles if occursin(fileNumber,file)]
    dataVec = Vector{}(undef,7)
    PMTS = [:PMT1,:PMT2,:PMT3,:PMT4]
    for i in 1:7
        tempdat = dataLoad(imagerFiles[i])
        tempdat[!,:gondola_time] .-= tempdat[!,:gondola_time][1]
        tempdat = tempdat[1:100000,:]
        dataVec[i] = tempdat
        println("Loaded $(i)")
    end
    # return plotVec

    edges = range(-7.1,7.1,load.histSize+1)
    savePath = "/home/wyatt/Projects/BOOMS/imaging/imageArrays/"*saveName
    if isfile(savePath)
        rm(savePath)
    end
    writeArray = Array{Float64}(undef,load.histSize,load.histSize*7)

    for i in 1:7
        chunk = dataVec[i]
        pmtArray = Float64.(Array(chunk[!,[:PMT1,:PMT2,:PMT3,:PMT4]]))
        res = julia_mcmc.MLE(pmtArray,i)
        xs = res[:,1]
        ys = res[:,2]
        histFit =   StatsBase.fit(Histogram ,(xs,ys ),(edges,edges))
        #julia is dumb, so the lower right part of an array corresponds to the I quadrant
        imageArray = rotl90(histFit.weights)
        imageArray = load.imageRotations[i](imageArray) #need to rotate imagers to match gondola config
        writeArray[:,1 + (i-1)*load.histSize:(i)*load.histSize] .= imageArray
    end
    CSV.write( savePath,Tables.table(writeArray),append=true)

end
#34777.568  ,  34835.000
#    # times = [2673]
