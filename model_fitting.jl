using Sundials, Optimization, OptimizationBBO, OptimizationOptimJL, CairoMakie, Random
include("detailed_training_program.jl")
include("bvtv_data.jl")
include("makie_functions.jl")
set_theme!(makie_theme)

output_dir = "output/parameter_comparison/"
@parameters σ strain_rate n_strides
sys = bone_adaptation_system((nothing,nothing,nothing))

struct BoneParameterizationOptions
    prob_work::ODEProblem
    prob_rest::ODEProblem
    prob_race::ODEProblem
    t_work::Vector{Float64}
    fBM_work::Vector{Float64}
    t_rest::Vector{Float64}
    fBM_rest::Vector{Float64}
    median_fracture_time::Float64
end

function generate_ode_probs(t_work_end, t_rest_end, median_fBM_work, median_fBM_rest)
    speed_work = 16.0
    σ_work = joint_stress(speed_work)
    sr_work = strain_rate_fitted(speed_work)
    tspan_work = (0.0, t_work_end)
    prob_work = ODEProblem(sys,[fBM => median_fBM_rest],tspan_work,[σ=>σ_work, strain_rate=>sr_work, n_strides=>0],reltol=1e-9,abstol=1e-12)
    
    speed_rest = 3.6
    σ_rest = joint_stress(speed_rest)
    sr_rest = strain_rate_fitted(speed_rest)
    tspan_rest = (0.0, t_rest_end)
    prob_rest = ODEProblem(sys,[fBM => median_fBM_work],tspan_rest,[σ=>σ_rest, strain_rate=>sr_rest, n_strides=>0],reltol=1e-9,abstol=1e-12)

    tp = construct_training_program(
        rest_duration=0, pretraining_duration=0, racefit_duration=365,
        racefit_slow_gallop_distance_per_day=12000/30, racefit_fast_gallop_distance_per_day=4800/30
    )
    f_σ(t) = joint_stress(speedfunc(tp)(t))
    f_ϵ(t) = strain_rate_fitted(speedfunc(tp)(t))
    f_strides(t) = damagefunc(tp,σ0=σ0_default)(t) * fatigue_life(f_σ(t))
    sys_race = bone_adaptation_system((f_σ,f_ϵ,f_strides))
    tspan_race = (0.0, total_program_duration(tp))
    prob_race = ODEProblem(sys_race,[fBM => median_fBM_rest],tspan_race,[],reltol=1e-9,abstol=1e-12)

    return (prob_work,prob_rest,prob_race)
end

generate_p_input(p) = [
    A_OBL_max => p[1],
    A_OCL_max => p[2],
    σ0 => p[3]
]

function simulate_bvtv(prob,p_input)
    prob_new = remake(prob,p=p_input)
    sol = solve(prob_new, CVODE_BDF())
    return sol
end

function simulate_bvtv_plot(prob,p_input,tspan)
    prob_new = remake(prob,p=p_input,tspan=tspan)
    sol = solve(prob_new, CVODE_BDF())
    return sol
end

function objective_bvtv(p,bpo::BoneParameterizationOptions)
    p_input = generate_p_input(p)

    sol_work = simulate_bvtv(bpo.prob_work,p_input)
    fBM_work_model = sol_work.(bpo.t_work,idxs=fBM)
    loss_work = sum(abs2, bpo.fBM_work .- fBM_work_model)

    sol_rest = simulate_bvtv(bpo.prob_rest,p_input)
    fBM_rest_model = sol_rest(bpo.t_rest,idxs=fBM)
    loss_rest = sum(abs2, bpo.fBM_rest .- fBM_rest_model)

    return loss_work + (length(bpo.t_work)/length(bpo.t_rest))^2*loss_rest
end
function fracture_time(sol)
    idx_fracture = findfirst(sol[dmg] .> 1.0)
    if ~isnothing(idx_fracture)
        t0 = sol.t[idx_fracture-1]
        t1 = sol.t[idx_fracture]
        y0 = sol[dmg][idx_fracture-1]
        y1 = sol[dmg][idx_fracture]
        t_fracture = t0 + (1-y0)/(y1-y0)*(t1-t0)
    else
        t_fracture = sol.t[end]
    end
    return t_fracture
end
function objective_fracture(p,bpo::BoneParameterizationOptions)
    p_input = generate_p_input(p)
    sol = simulate_bvtv(bpo.prob_race,p_input)
    t_fracture = fracture_time(sol)
    return (t_fracture-bpo.median_fracture_time)^2
end

# regularisation_term(p,p0) = sum(abs2, (p .- p0)./p0)
objective_function_full(p,bpo::BoneParameterizationOptions) = objective_bvtv(p,bpo) + objective_fracture(p,bpo)

function fit_model_parameters(df;filterage=true)
    ((t_work, fBM_work, t_rest, fBM_rest),(median_fBM_work,median_fBM_rest)) = load_bv_tv_timeseries(df)
    if sum(df.fracture) == 0
        fracture_times = Float64[]
        median_fracture_time = 22*7
    else
        (fracture_times,median_fracture_time) = median_fracture_time_hitchens(df;filterage=filterage)
    end
    println("$(length(t_work)) work data points, $(length(t_rest)) rest data points, $(length(fracture_times)) fracture data points")
    (prob_work,prob_rest,prob_race) = generate_ode_probs(maximum(t_work),maximum(t_rest), median_fBM_work, median_fBM_rest)

    ## Stage 1: Global optimisation
    bpo = BoneParameterizationOptions(prob_work, prob_rest, prob_race, t_work, fBM_work, t_rest, fBM_rest, median_fracture_time)
    p0 = [A_OBL_max_default,A_OCL_max_default,σ0_default]
    lb = [A_OBL_min_default,A_OCL_min_default,σ0_default-30.0]
    ub = [A_OBL_max_default,A_OCL_max_default,σ0_default+30.0]
    optf = OptimizationFunction(objective_function_full, Optimization.AutoFiniteDiff())
    optprob = OptimizationProblem(optf, p0, bpo, lb=lb, ub=ub)
    popt1 = solve(optprob,BBO_adaptive_de_rand_1_bin_radiuslimited(),maxtime = 60.0)

    # Stage 2: Local optimisation
    optf = OptimizationFunction(objective_function_full, Optimization.AutoFiniteDiff())
    optprob = OptimizationProblem(optf, popt1.u, bpo, lb=lb, ub=ub)
    popt = solve(optprob,BFGS(),allow_f_increases=false)
    return popt, prob_work, prob_rest, prob_race
end

function plot_model_data_comparison(df, popt, prob_work, prob_rest, prob_race; filterage=true)
    p_input = generate_p_input(popt)
    sol_work = simulate_bvtv(prob_work,p_input)
    sol_rest = simulate_bvtv(prob_rest,p_input)
    sol_race = simulate_bvtv(prob_race,p_input)

    ((t_work, fBM_work, t_rest, fBM_rest),(median_fBM_work,median_fBM_rest)) = load_bv_tv_timeseries(df)
    (fracture_times,median_fracture_time) = median_fracture_time_hitchens(df;filterage=filterage)

    fig = Figure(size=(1200,350))
    Label(fig[0,1],"a",halign=:left,valign=:top,tellwidth=false,padding=(-60,0,-10,0))
    ax = Axis(fig[1,1], xlabel="Time (days)", ylabel="Bone volume fraction")
    scatter!(ax, t_work, fBM_work, markersize=7)
    lines!(ax, sol_work.t, sol_work[fBM], color=:black)
    Label(fig[0,2],"b",halign=:left,valign=:top,tellwidth=false,padding=(-60,0,-10,0))
    ax = Axis(fig[1,2], xlabel="Time (days)", ylabel="Bone volume fraction")
    scatter!(ax, t_rest, fBM_rest, markersize=7)
    lines!(ax, sol_rest.t, sol_rest[fBM], color=:black)
    Label(fig[0,3],"c",halign=:left,valign=:top,tellwidth=false,padding=(-60,0,-10,0))

    t_fracture = fracture_time(sol_race)
    t_race = LinRange(0,t_fracture,501)
    gc = fig[1,3] = GridLayout()
    ax3a = Axis(gc[1,1])
    hidedecorations!(ax3a)
    hidespines!(ax3a)
    hist!(ax3a, fracture_times, direction=:y, bins=0:25:250, normalization=:probability)
    vlines!(ax3a, median(fracture_times), color=:grey)
    ax3b = Axis(gc[2,1], xlabel="Time (days)", ylabel="Damage")
    lines!(ax3b, t_race, sol_race.(t_race,idxs=dmg), color=:black)
    hlines!(ax3b, 1.0, color=:grey)
    vlines!(ax3b, median_fracture_time, color=:grey)
    linkxaxes!(ax3a, ax3b)
    ylims!(ax3b,high=1.2)
    rowsize!(gc,1,Auto(0.25))
    rowgap!(gc,5)
    return fig
end

function plot_collated_comparison(dfs,feature_xlabel,feature_labels)
    outputs = []
    for (i,df) in enumerate(dfs)
        Random.seed!(4892)
        println("Fitting model to feature $feature_xlabel, group $(feature_labels[i])")
        outs = (popt, prob_work, prob_rest, prob_race) = fit_model_parameters(df;filterage=false)
        push!(outputs, outs)
    end

    feature_xticks = (1:length(feature_labels), feature_labels)

    fig = Figure(size=(1350,650))
    Label(fig[0,1],"a",halign=:left,valign=:top,tellwidth=false,padding=(-60,0,-10,0))
    ax1 = Axis(fig[1,1], xlabel="Time (days)", ylabel="Bone volume fraction")
    Label(fig[0,2],"b",halign=:left,valign=:top,tellwidth=false,padding=(-60,0,-10,0))
    ax2 = Axis(fig[1,2], xlabel="Time (days)", ylabel="Bone volume fraction")
    Label(fig[0,3],"c",halign=:left,valign=:top,tellwidth=false,padding=(-60,0,-10,0))
    ax3 = Axis(fig[1,3], xlabel="Time (days)", ylabel="Damage")
    hlines!(ax3, 1.0, color=:grey)
    ax4 = Axis(fig[3,1], xlabel=feature_xlabel, ylabel=L"A_\text{OBL}^\text{max}", xticks=feature_xticks)
    Label(fig[2,1],"d",halign=:left,valign=:top,tellwidth=false,padding=(-60,0,-10,0))
    ax5 = Axis(fig[3,2], xlabel=feature_xlabel, ylabel=L"A_\text{OCL}^\text{max}", xticks=feature_xticks)
    Label(fig[2,2],"e",halign=:left,valign=:top,tellwidth=false,padding=(-60,0,-10,0))
    ax6 = Axis(fig[3,3], xlabel=feature_xlabel, ylabel=L"σ_0", xticks=feature_xticks)
    Label(fig[2,3],"f",halign=:left,valign=:top,tellwidth=false,padding=(-60,0,-10,0))

    for (i,(outs,df)) in enumerate(zip(outputs,dfs))
        (popt, prob_work, prob_rest, prob_race) = outs

        p_input = generate_p_input(popt)
        sol_work = simulate_bvtv_plot(prob_work,p_input,(0.0,tmax_work))
        sol_rest = simulate_bvtv_plot(prob_rest,p_input,(0.0,tmax_rest))
        sol_race = simulate_bvtv(prob_race,p_input)

        ((t_work, fBM_work, t_rest, fBM_rest),(median_fBM_work,median_fBM_rest)) = load_bv_tv_timeseries(df)
        (fracture_times,median_fracture_time) = median_fracture_time_hitchens(df;filterage=false)

        c = Makie.wong_colors()[i]
        scatter!(ax1, t_work, fBM_work, markersize=7, color=c)
        l1 = lines!(ax1, sol_work.t, sol_work[fBM], color=c)
        translate!(l1,0,0,1)
        scatter!(ax2, t_rest, fBM_rest, markersize=7, color=c)
        l2 = lines!(ax2, sol_rest.t, sol_rest[fBM], color=c, label=feature_labels[i])
        translate!(l2,0,0,1)
        t_fracture = fracture_time(sol_race)
        t_race = LinRange(0,t_fracture,501)
        lines!(ax3, t_race, sol_race.(t_race,idxs=dmg), color=c)
        if ~isempty(fracture_times)
            scatter!(ax3, fracture_times, 0.95*ones(length(fracture_times))+0.1*rand(length(fracture_times)), color=c, strokecolor=:white, strokewidth=1)
        end
        scatter!(ax4, i, outs[1][1], color=c, markersize=12)
        scatter!(ax5, i, outs[1][2], color=c, markersize=12)
        scatter!(ax6, i, outs[1][3], color=c, markersize=12)
    end
    ylims!(ax3,0,1.2)
    fig[1,4] = Legend(fig, ax2, feature_xlabel, framevisible = false)

    A_OBL_max_vec = [outs[1][1] for outs in outputs]
    A_OCL_max_vec = [outs[1][2] for outs in outputs]
    σ0_vec = [outs[1][3] for outs in outputs]
    l4 = lines!(ax4, 1:length(outputs), A_OBL_max_vec, color=:black)
    translate!(l4,0,0,-1)
    l5 = lines!(ax5, 1:length(outputs), A_OCL_max_vec, color=:black)
    translate!(l5,0,0,-1)
    l6 = lines!(ax6, 1:length(outputs), σ0_vec, color=:black)
    translate!(l6,0,0,-1)
    xlims!(ax4,0.5,length(outputs)+0.5)
    xlims!(ax5,0.5,length(outputs)+0.5)
    xlims!(ax6,0.5,length(outputs)+0.5)
    return fig
end

# Load in data
df = load_bvtv_data()
tmax_rest = 7*maximum(-filter(r -> ~ismissing(r.timeinwork) && r.timeinwork < 0, df).timeinwork)
tmax_work = 7*maximum(filter(r -> ~ismissing(r.timeinwork) && r.timeinwork > 0, df).timeinwork)
# Fit model parameters to data
Random.seed!(4892)
(popt, prob_work, prob_rest, prob_race) = fit_model_parameters(df)
# Plot comparison between model and data
fig = plot_model_data_comparison(df, popt, prob_work, prob_rest, prob_race)
savefig("output/model_fit_comparison",fig)