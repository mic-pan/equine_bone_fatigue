using DataFrames, CSV, CairoMakie
output_dir = "output/training_fatigue_life/"
include("makie_functions.jl")
set_theme!(makie_theme)

# Calculate the fatigue life for a speed
function fatigue_life(speed;scaled_load=90)
    stress = joint_stress(speed,scaled_load=scaled_load)
    fl = 10^(-(stress-134.2)/14.1)
    return fl
end

function joint_stress(speed;scaled_load=90)
    vertical_force = 2.778+2.137*speed-0.0535*speed^2 # Unit N/kg
    conversion_factor = scaled_load/24.13 # Unit MPa.kg/N
    stress = conversion_factor*vertical_force # Unit MPa
    return stress
end

# Calculate the stride length for a given speed
function stride_length(speed)
    stride_frequency = 1.7052+0.0305*speed+0.0004*speed^2 # Unit Hz
    stride_length = speed/stride_frequency # Unit m
    return stride_length
end

# Generate the distance per week for high-volume progressive training program
function high_volume_progressive()
    weeks = 1:8
    slow_distance = 2500 # Unit m
    fast_distance = 1850 # Unit m
    fast_training = fast_distance*[0,0,0,0,1,0,1,0]
    return DataFrame(week=weeks,slow_distance=slow_distance,fast_distance=fast_training)
end

# Generate the distance per week for moderate progressive training program
function moderate_progressive()
    weeks = 1:5
    slow_distance = 1500 # Unit m
    fast_distance = 1300 # Unit m
    fast_training = fast_distance*[0,0,0,1,2]
    return DataFrame(week=weeks,slow_distance=slow_distance,fast_distance=fast_training)
end

# Generate the distance per week for fast and light progressive training program
function fast_and_light_progressive()
    weeks = 1:6
    slow_distance = 1000 # Unit m
    fast_distance = 2300 # Unit m
    fast_training = fast_distance*[0,0,0,0,1,0]
    return DataFrame(week=weeks,slow_distance=slow_distance,fast_distance=fast_training)
end

# Generate progressive training program
function progressive_training(regime;duration=12)
    if regime == :high
        program = high_volume_progressive()
    elseif regime == :moderate
        program = moderate_progressive()
    elseif regime == :low
        program = fast_and_light_progressive()
    else
        error("Invalid regime")
    end

    weeks = 1:duration
    slow_distance = [zeros(duration-length(program.week));program.slow_distance]
    fast_distance = [zeros(duration-length(program.week));program.fast_distance]
    return DataFrame(week=weeks,slow_distance=slow_distance,fast_distance=fast_distance)
end

# Generate maintenance program from slow and fast distances
function maintenance_program(slow_distance,fast_distance,duration)
    weeks = 1:duration
    flag_training = @. (weeks%2==0)
    slow_training = slow_distance*flag_training
    fast_training = fast_distance*flag_training
    return DataFrame(week=weeks,slow_distance=slow_training,fast_distance=fast_training)
end

# Generate the distance per week for maintenance programs
low_maintenance(;duration=8) = maintenance_program(800,1200,duration)
medium_maintenance(;duration=8) = maintenance_program(2400,1600,duration)
medium_high_maintenance(;duration=8) = maintenance_program(4000,2400,duration)
high_volume_maintenance(;duration=8) = maintenance_program(6000,2400,duration)

function maintenance_training(regime;duration=8,shift=0)
    if regime == :low
        maintenance = low_maintenance(duration=duration)
    elseif regime == :moderate
        maintenance = medium_maintenance(duration=duration)
    elseif regime == :medium_high
        maintenance = medium_high_maintenance(duration=duration)
    elseif regime == :high
        maintenance = high_volume_maintenance(duration=duration)
    else
        error("Invalid regime")
    end

    maintenance.week .+= shift
    return maintenance
end

# Generate full training program
function full_training(progressive_group,maintenance_group;progressive_duration=12,racing_duration=8)
    progressive = progressive_training(progressive_group,duration=progressive_duration)
    maintenance = maintenance_training(maintenance_group,duration=racing_duration,shift=progressive_duration)
    df = vcat(progressive,maintenance)
    df.race = @. (df.week > progressive_duration) && (df.slow_distance == 0)
    return df
end

racing_fatigue_life() = Dict(
    54 => 0.0003,
    66 => 0.0020,
    78 => 0.0138,
    90 => 0.0934
)

# Add fatigue life to the training program
function add_fatigue_life!(df;slow_speed=13.8,fast_speed=16.1,scaled_load=90)
    # Add racing fatigue
    fatigue = racing_fatigue_life()[scaled_load]
    df.racing_fatigue = fatigue*df.race

    # Calculate slow gallop fatigue
    df.slow_strides = slow_strides = df.slow_distance./stride_length(slow_speed)
    df.slow_fatigue = slow_fatigue = slow_strides./fatigue_life(slow_speed,scaled_load=scaled_load)

    # Calculate fast gallop fatigue
    df.fast_strides = fast_strides = df.fast_distance./stride_length(fast_speed)
    df.fast_fatigue = fast_fatigue = fast_strides./fatigue_life(fast_speed,scaled_load=scaled_load)

    # Calculate total fatigue
    df.total_fatigue = slow_fatigue+fast_fatigue+df.racing_fatigue
    df.cumulative_fatigue = cumsum(df.total_fatigue)

    return df
end

# Generate full DataFrame with training programs and fatigue life
function training_racing_fatigue(progressive_group,maintenance_group;
    progressive_duration=12,racing_duration=8,slow_speed=13.8,fast_speed=16.1,scaled_load=90)
    df = full_training(progressive_group,maintenance_group,progressive_duration=progressive_duration,racing_duration=racing_duration)
    add_fatigue_life!(df,slow_speed=slow_speed,fast_speed=fast_speed,scaled_load=scaled_load)
    return df
end

# Generate DataFrames for fatigue life generated over all possible training programs
pgs = [:low,:moderate,:high]
mgs = [:low,:moderate,:medium_high,:high]
scaled_loads = [54,66,78,90]
dict_fatigue = Dict{Tuple{Symbol,Symbol,Float64},DataFrame}()

for pg in pgs, mg in mgs, sl in scaled_loads
    df = training_racing_fatigue(pg,mg,scaled_load=sl)
    dict_fatigue[(pg,mg,sl)] = df
    filename = output_dir*"fatigue_life_$(pg)_$(mg)_$(sl).csv"
    CSV.write(filename,df)
end

function summarise_fatigue(df,progressive_duration)
    training_fatigue = sum(df.slow_fatigue[1:progressive_duration] + df.fast_fatigue[1:progressive_duration])
    maintenance_fatigue = sum(df.slow_fatigue[progressive_duration+1:end] + df.fast_fatigue[progressive_duration+1:end])
    racing_fatigue = sum(df.racing_fatigue)
    return [training_fatigue,maintenance_fatigue,racing_fatigue]
end

function plot_damage_spread(dict_fatigue,idx)
    colors = Makie.wong_colors()[1:3]
    labels = ["Progressive","Maintenance","Racing"]
    fig = Figure(size=(800,800))
    sl = scaled_loads[idx]
    for (i,pg) in enumerate(pgs), (j,mg) in enumerate(mgs)
        df = dict_fatigue[(pg,mg,sl)]
        fls = summarise_fatigue(df,12)
        ax = Axis(fig[j,i],autolimitaspect=1,title = "Fatigue = $(round(df.cumulative_fatigue[end];digits=2))",
            titlesize=14,titlegap=0)
        pie!(ax,
            fls,["Training","Maintenance","Racing"],
            color=colors,
            strokecolor = :white,
            strokewidth = 2
        )
        hidedecorations!(ax)
        hidespines!(ax)
    end
    Legend(fig[1,4], [PolyElement(color=c) for c in colors], labels, framevisible=false)

    Label(fig[-2,:], "Scaled load = $sl MPa", tellwidth=false, halign=:center)
    Label(fig[-1,1:3], "Progressive program", tellwidth=false, halign=:center)
    Label(fig[0,1], "Fast and light", tellwidth=false, halign=:center, font=:regular)
    Label(fig[0,2], "Moderate", tellwidth=false, halign=:center, font=:regular)
    Label(fig[0,3], "High volume", tellwidth=false, halign=:center, font=:regular)

    Label(fig[:,-1], "Maintenance program", rotation=π/2, tellheight=false)
    Label(fig[1,0], "Low volume", rotation=π/2, tellheight=false, font=:regular)
    Label(fig[2,0], "Medium volume", rotation=π/2, tellheight=false, font=:regular)
    Label(fig[3,0], "Medium-high", rotation=π/2, tellheight=false, font=:regular)
    Label(fig[4,0], "High volume", rotation=π/2, tellheight=false, font=:regular)
    return fig
end

for i in eachindex(scaled_loads)
    fig = plot_damage_spread(dict_fatigue,i)
    savefig(output_dir*"fatigue_proportion_$(scaled_loads[i])",fig)
end

# Plot the relationship between fatigue accumulated and training program
sl = 90
progressive_fatigues = [dict_fatigue[(pg,:low,sl)].cumulative_fatigue[12] for pg in pgs]
maintenance_fatigues = [summarise_fatigue(dict_fatigue[(:low,mg,sl)],12)[2] for mg in mgs]
fig = Figure(size=(350,750))
ax = Axis(fig[1,1], xticks = (1:3, ["Fast and light", "Moderate", "High volume"]), xticklabelrotation=π/4
    ,ylabel="Damage", title="Progressive program (12 weeks)")
ax.rightspinevisible = false
ax.topspinevisible = false
barplot!(ax,1:3,progressive_fatigues, color=Makie.wong_colors()[1])
ax = Axis(fig[2,1], xticks = (1:4, ["Low volume", "Medium volume", "Medium-high", "High volume"]), xticklabelrotation=π/4
    ,ylabel="Damage", title="Maintenance program (8 weeks)")
ax.rightspinevisible = false
ax.topspinevisible = false
barplot!(ax,1:4,maintenance_fatigues, color=Makie.wong_colors()[2])
savefig(output_dir*"training_fatigue_comparison",fig)