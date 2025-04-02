using DifferentialEquations, Sundials
include("detailed_training_program.jl")
# include("bvtv_data.jl")
include("training_program_plots.jl")
include("makie_functions.jl")
set_theme!(makie_theme)

# Calculate the initial condition for bone volume fraction using the steady-state value at trot
ps = p_optimal()
fBM0 = fBM_ss(joint_stress(3.6),strain_rate_fitted(3.6),A_OBL_max=ps[1][2],A_OCL_max=ps[2][2])

## Define the bone adaptation system
tp = construct_training_program()
sys = ODESystem(tp)

# Run a simulation over multiple seasons
n_seasons = 4
prog_dur = total_program_duration(tp)
rest_periods = [(i*prog_dur,i*prog_dur+tp.rest_program.duration) for i in 0:(n_seasons-1)]
pretraining_periods = [(b,b+tp.pretraining_program.duration) for (a,b) in rest_periods]
progressive_periods = [(b,b+progressive_program_duration(tp)) for (a,b) in pretraining_periods]
racing_periods = [(b,b+tp.racefit_program.duration) for (a,b) in progressive_periods]

t_end = n_seasons*prog_dur
tspan = (0.0,t_end)
prob = ODEProblem(sys,[fBM=>fBM0],tspan,ps,reltol=1e-9,abstol=1e-12)
sol = solve(prob, CVODE_BDF())

# Calculate the distance and damage
Δt = 0.1
t_plot = ((n_seasons-1)*prog_dur) .+ (slow_progressive_start_time(tp):Δt:prog_dur)
slow_distance = multiseason(tp,slowworkout_distancefunc).(t_plot)
transition_distance = multiseason(tp,slowprogressive_distancefunc).(t_plot)
gallop_distance = multiseason(tp,gallop_distancefunc).(t_plot)
race_speed_distance = multiseason(tp,racespeed_distancefunc).(t_plot)
distances = [slow_distance,transition_distance,gallop_distance,race_speed_distance]

cumulative_slow_distance = cumsum(slow_distance)*Δt
cumulative_transition_distance = cumsum(transition_distance)*Δt
cumulative_gallop_distance = cumsum(gallop_distance)*Δt
cumulative_race_speed_distance = cumsum(race_speed_distance)*Δt
cumulative_distances = [cumulative_slow_distance,cumulative_transition_distance,cumulative_gallop_distance,cumulative_race_speed_distance]
cumulative_distances_km = @. cumulative_distances/1000

E_plot = sol(t_plot)[E]
scaling_factor = E_nominal_default./E_plot

slow_damage = multiseason(tp,canter_damagefunc).(t_plot) .* scaling_factor
transition_damage = multiseason(tp,transitionspeed_damagefunc).(t_plot) .* scaling_factor
gallop_damage = multiseason(tp,gallop_damagefunc).(t_plot) .* scaling_factor
race_speed_damage = multiseason(tp,racespeed_damagefunc).(t_plot) .* scaling_factor
damages = [slow_damage,transition_damage,gallop_damage,race_speed_damage]

cumulative_slow_damage = cumsum(slow_damage)*Δt
cumulative_transition_damage = cumsum(transition_damage)*Δt
cumulative_gallop_damage = cumsum(gallop_damage)*Δt
cumulative_race_speed_damage = cumsum(race_speed_damage)*Δt
cumulative_damages = [cumulative_slow_damage,cumulative_transition_damage,cumulative_gallop_damage,cumulative_race_speed_damage]

labels = ["7.5 m/s","11.8-13.8 m/s","13.8 m/s","16-16.7 m/s"]

function bandplot!(layout, t, ys, labels; xlabel="", ylabel="", fontsize=14)
    ax = Axis(layout, xlabel=xlabel, ylabel=ylabel,
        xlabelsize=fontsize,ylabelsize=fontsize,xticklabelsize=fontsize,yticklabelsize=fontsize,
        palette=(patchcolor = Makie.wong_colors()[[1,3,2,6]],))
    y1 = zeros(length(t))
    y2 = zeros(length(t))
    for (y,l) in zip(ys,labels)
        y2 = y1 + y
        band!(ax,t,y1,y2,label=l)
        y1 = y2
    end
    return ax
end

# Plot the bone volume fraction and damage
fig = Figure(size=(900,1000))
ga = fig[1,1] = GridLayout()
ax1 = Axis(ga[1,1], xlabel="", ylabel=L"$f_{BM}$", xticks=LinearTicks(9))
lines!(ax1, sol.t, sol[fBM], color=:black)
add_bg!(ax1,rest_periods,pretraining_periods,progressive_periods,racing_periods)
ax2 = Axis(ga[2,1], xlabel="Time (days)", ylabel="Damage", xticks=LinearTicks(9))
lines!(ax2, sol.t, sol[dmg], color=:black)
add_bg!(ax2,rest_periods,pretraining_periods,progressive_periods,racing_periods)
Label(ga[1,1,TopLeft()],"a")
Label(ga[2,1,TopLeft()],"b")
Legend(
    fig[1,2], [PolyElement(color=c) for c in bg_palette_programs], ["Rest", "Pre-training", "Progressive", "Race-fit"], "Program",
    framevisible=false, padding=(0,0,250,0)
)
rowgap!(ga,0)

# Plot the distance and damage accumulated over a single season
gb = fig[2,1] = GridLayout()
ax3 = bandplot!(gb[1,1], t_plot, distances, labels, xlabel="Time (days)", ylabel="Distance per day (m/day)")
add_bg!(ax3,[],[],[progressive_periods[end]],[racing_periods[end]])
ax4 = bandplot!(gb[1,2], t_plot, cumulative_distances_km, labels, xlabel="Time (days)", ylabel="Cumulative distance (km)")
add_bg!(ax4,[],[],[progressive_periods[end]],[racing_periods[end]])
ax5 = bandplot!(gb[2,1], t_plot, damages, labels, xlabel="Time (days)", ylabel="Damage per day")
add_bg!(ax5,[],[],[progressive_periods[end]],[racing_periods[end]])
ylims!(ax5, high=0.011)
ax6 = bandplot!(gb[2,2], t_plot, cumulative_damages, labels, xlabel="Time (days)", ylabel="Cumulative damage")
add_bg!(ax6,[],[],[progressive_periods[end]],[racing_periods[end]])

Label(gb[1,1,TopLeft()],"c")
Label(gb[1,2,TopLeft()],"d")
Label(gb[2,1,TopLeft()],"e")
Label(gb[2,2,TopLeft()],"f")
rowgap!(gb,1,0)

Legend(fig[2,2],ax4,"Speed",framevisible=false,tellheight=false,padding=(0,0,360,0))
rowsize!(fig.layout,1,Auto(0.7))
rowgap!(fig.layout,25)
savefig("output/training_program_simulations",fig)