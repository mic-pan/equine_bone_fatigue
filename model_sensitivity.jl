using CairoMakie, DifferentialEquations, QuasiMonteCarlo, GlobalSensitivity, Colors, Sundials, StatsBase
include("bone_adaptation_model.jl")
include("makie_functions.jl")
set_theme!(makie_theme)

# List out the parameters of the model
@parameters n_strides strain_rate σ
sys = bone_adaptation_system((nothing,nothing,nothing))
tspan = (0.0,70.0)
# Assuming racehorses cover 750m per week at a speed of 14.0 m/s
default_speed = 14.0
new_defaults = vcat(
    [
        σ => joint_stress(default_speed), 
        n_strides => 8000/30/stride_length(default_speed), 
        strain_rate => strain_rate_fitted(default_speed)
    ],
    p_optimal()
)
dict_defaults = Dict(new_defaults)
for (k,v) in new_defaults
    sys.defaults[k] = v
end
prob = ODEProblem(sys,[fBM => 0.7],tspan,new_defaults)

# Define appropriate ranges for each parameter 
# Default means 0.5-1.5 times the regular value
ps = ModelingToolkit.parameters(sys)
default_bounds(x,sys) = [0.5*sys.defaults[x],1.5*sys.defaults[x]]
bounds = Dict{Num,Vector{Float64}}(
    A_OCL_min => [0.0, 0.5*dict_defaults[A_OCL_max]],
    A_OBL_min => [0.0, 0.5*dict_defaults[A_OBL_max]],
    γB => [1, 5],
    γC => [1, 5],
    bSv => [-0.2, 0.2],
    σ1 => [0.9*σ1_default, 1.1*σ1_default],
    σ0 => [0.9*dict_defaults[σ0], 1.1*dict_defaults[σ0]],
    Fs => [1, 10],
    σ => [30, 102],
    n_strides => [25, 120],
    strain_rate => [0.05, 0.46]
)
_p_default_bounds = [A_OCL_max, A_OBL_max, δC, δB, γE, α, aSv, E0, E_nominal]
for p in _p_default_bounds
    bounds[p] = default_bounds(p,sys)
end

## Run a univariate sensitivity analysis on each of the parameters
function run_simulation(ps;prob=prob)
    prob = remake(prob,p=ps)
    sol = solve(prob, CVODE_BDF(),reltol=1e-9,abstol=1e-12)
    return [sol[fBM][end], sol[dmg][end]]
end

function run_simulation_multipoints(ps;prob=prob,ts=7:7:70)
    prob = remake(prob,p=ps)
    sol = solve(prob, CVODE_BDF(),reltol=1e-9,abstol=1e-12)
    fBM_vec = sol(ts, idxs=fBM).u
    dmg_vec = sol(ts, idxs=dmg).u
    return [fBM_vec; dmg_vec]
end

function univariate_sensitivity(p,bounds;simulations=101,prob=prob)
    p_values = range(bounds[1],bounds[2],length=simulations)
    _outputs = [run_simulation([p=>val];prob=prob) for val in p_values]
    outputs = [_outputs[i][j] for i in eachindex(_outputs), j in eachindex(_outputs[1])]
    return (p_values,outputs)
end

function plot_sensitivity!(gp,p,p_values,outputs;sys=sys,showxlabel=true)
    cols = Makie.wong_colors()[[1,4]]
    xlabel = showxlabel ? string(p) : ""
    out_defaults = run_simulation([p=>sys.defaults[p]])

    ax1 = Axis(gp, xlabel=xlabel, ylabel="Bone volume fraction", 
        yticklabelcolor=cols[1], ylabelcolor=cols[1], ytickcolor=cols[1])
    ax2 = Axis(gp, xlabel=xlabel, ylabel="Damage", yaxisposition=:right, 
        yticklabelcolor=cols[2], ylabelcolor=cols[2], ytickcolor=cols[2])
    if string(p) == "A_OBL_min"
        ax1.xticks = 0.0:0.002:0.006
        ax2.xticks = 0.0:0.002:0.006
    elseif string(p) == "A_OCL_min"
        ax1.xticks = [0.0,0.001]
        ax2.xticks = [0.0,0.001]
    end
    lines!(ax1,p_values,outputs[:,1],color=cols[1])
    scatter!(ax1,[sys.defaults[p]],[out_defaults[1]],color=cols[1])
    ylims!(ax1,0.5,1.0)
    lines!(ax2,p_values,outputs[:,2],color=cols[2])
    scatter!(ax2,[sys.defaults[p]],[out_defaults[2]],color=cols[2])
    ylims!(ax2,-0.2,2.2)
    return nothing
end

function plot_sensitivity(p,p_values,outputs;sys=sys,showxlabel=true)
    fig = Figure()
    plot_sensitivity!(fig[1,1],p,p_values,outputs,showxlabel=showxlabel)
    return fig
end

p_grid = reshape(ps,(4,5))
dims = Base.size(p_grid)
fig = Figure(size=(2000,1000))
for i in 1:dims[1], j in 1:dims[2]
    p = p_grid[i,j]
    p_values,outputs = univariate_sensitivity(p,bounds[p],simulations=101)
    plot_sensitivity!(fig[i,j],p,p_values,outputs)
end
savefig("output/univariate_sensitivity_analysis",fig)

## Run a bivariate sensitivity analysis on parameters
function bivariate_sensitivity(p1,bounds1,p2,bounds2;simulations=11,prob=prob)
    p1_values = range(bounds1[1],bounds1[2],length=simulations)
    p2_values = range(bounds2[1],bounds2[2],length=simulations)
    _outputs = [run_simulation([p1=>val1, p2=>val2];prob=prob) for val1 in p1_values, val2 in p2_values]
    outputs = [_outputs[i,j][k] for i in axes(_outputs,1), j in axes(_outputs,2), k in eachindex(_outputs[1])]
    return (p1_values,p2_values,outputs)
end

function plot_bivariate_sensitivity!(gp,p1,p1_values,p2,p2_values,outputs,levels;sys=sys,cmap=:viridis)
    ax = Axis(gp,aspect=1)
    contourf!(ax,p1_values,p2_values,outputs,levels=levels,extendhigh=:auto,extendlow=:auto,colormap=cmap)
    scatter!(ax,[sys.defaults[p1]],[sys.defaults[p2]],color=Makie.wong_colors()[6],markersize=20)
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

output_store = Dict(
    (i,j) => 
    bivariate_sensitivity(p1,bounds[p1],p2,bounds[p2],simulations=51) 
    for (i,p1) in enumerate(ps), (j,p2) in enumerate(ps) if i < j
)

fig = Figure(size=(8000,8000))
np = length(ps)
axs = Matrix{Axis}(undef,(np,np))
for (i,p1) in enumerate(ps)
    for (j,p2) in enumerate(ps) if i < j
            (p1_values,p2_values,outputs) = output_store[(i,j)]
            axs[i,j] = plot_bivariate_sensitivity!(fig[i,j],p2,p2_values,p1,p1_values,outputs[:,:,1]',range(0.5,1.0,length=11),cmap=:Blues) # BV/TV
            axs[j,i] = plot_bivariate_sensitivity!(fig[j,i],p1,p1_values,p2,p2_values,outputs[:,:,2],range(0.0,2.0,length=11),cmap=:RdPu) # Damage
        end
    end
end
for (i,p) in enumerate(ps)
    axs[i,i] = ax = Axis(fig[i,i],aspect=1)
    xlims!(ax, bounds[p])
    ylims!(ax, bounds[p])
    hidedecorations!(ax)
    hidespines!(ax)
    text!(
        ax, 0.5, 0.5,
        text = string(p),
        font = :bold,
        align = (:center, :center),
        space = :relative,
        fontsize = 36
    )
end
for i in eachindex(ps)
    linkyaxes!(axs[i,:]...)
    linkxaxes!(axs[:,i]...)
    axs[end,i].bottomspinevisible = true
    axs[end,i].xticksvisible = true
    axs[end,i].xticklabelsvisible = true
    axs[i,1].leftspinevisible = true
    axs[i,1].yticksvisible = true
    axs[i,1].yticklabelsvisible = true
end
Colorbar(fig[1,np+1], axs[1,2].scene.plots[1], label="Bone volume fraction")
Colorbar(fig[2,np+1], axs[2,1].scene.plots[1], label="Damage")
savefig("output/bivariate_sensitivity_analysis",fig)

## Conduct a sensitivity analysis using PRCC
ps_latex = Dict(
    A_OCL_max => L"A_{\text{OCL}}^\text{max}",
    A_OBL_max => L"A_{\text{OBL}}^\text{max}",
    A_OCL_min => L"A_{\text{OCL}}^\text{min}",
    A_OBL_min => L"A_{\text{OBL}}^\text{min}",
    δC => L"\delta_C",
    δB => L"\delta_B",
    γE => L"\gamma_E",
    α => L"\alpha",
    aSv => L"a",
    E0 => L"E_0",
    E_nominal => L"E_{\text{nom}}",
    γB => L"\gamma_B",
    γC => L"\gamma_C",
    bSv => L"b",
    σ1 => L"\sigma_1",
    σ0 => L"\sigma_0",
    Fs => L"F_s",
    σ => L"\sigma",
    n_strides => L"v_n",
    strain_rate => L"\dot{\varepsilon}"
)

N = 10^5
sampler = SobolSample()
lb = [[bounds[p][1] for p in ps];0]
ub = [[bounds[p][2] for p in ps];1]
p_sample_mat = QuasiMonteCarlo.sample(N,lb,ub,sampler)
p_sample = [p_sample_mat[1:end-1,i] for i in axes(p_sample_mat,2)]

outputs = run_simulation.(p_sample)
outputs_mat = [outputs[i][j] for j in eachindex(outputs[1]), i in eachindex(outputs)]

function partial_rank_correlation(X,Y)
    X_rank = vcat((tiedrank(view(X, i, :))' for i in axes(X, 1))...)
    Y_rank = [tiedrank(view(Y, i, :))' for i in axes(Y, 1)]
    prcc = vcat((GlobalSensitivity._calculate_partial_correlation_coefficients(X_rank, y) for y in Y_rank)...)
    # Y_rank = [tiedrank(Y[i,:]) for i in axes(Y, 1)]
    # prcc = [partialcor(X_rank[i,:],Y_rank[j],X_rank[1:end .!= i,:]') for i in axes(X_rank,1), j in eachindex(Y_rank)]
    return prcc
end
prcc = partial_rank_correlation(p_sample_mat,outputs_mat)

# Order based on PRCC values
idx_perm_fBM = sortperm(prcc[1,1:np])
_ps_fBM_sorted = [ps_latex[ps[i]] for i in idx_perm_fBM]
ps_fBM_sorted = [_ps_fBM_sorted;"Dummy"]
prcc_fBM_sorted = [prcc[1,idx_perm_fBM];prcc[1,end]]

idx_perm_dmg = sortperm(prcc[2,1:np])
_ps_dmg_sorted = [ps_latex[ps[i]] for i in idx_perm_dmg]
ps_dmg_sorted = [_ps_dmg_sorted;"Dummy"]
prcc_dmg_sorted = [prcc[2,idx_perm_dmg];prcc[2,end]]

fig = Figure(size=(900,500))
ax = Axis(
    fig[1,1],xlabel="PRCC",
    yticks = (1:np+1, ps_fBM_sorted),
    yticklabelsize = 14,
    yreversed=true,
    title="Bone volume fraction", titlefont=:bold
)
barplot!(ax,1:np+1,prcc_fBM_sorted,color=Makie.wong_colors()[1],direction=:x)
hidespines!(ax,:t,:r)
ax = Axis(
    fig[1,2],xlabel="PRCC",
    yticks = (1:np+1, ps_dmg_sorted),
    yticklabelsize = 14,
    yreversed=true,
    title="Damage", titlefont=:bold
)
barplot!(ax,1:np+1,prcc_dmg_sorted,color=Makie.wong_colors()[4],direction=:x)
hidespines!(ax,:t,:r)
Label(fig[1,1],"a",halign=:left,valign=:top,tellwidth=false,tellheight=false,padding=(-40,0,0,-40))
Label(fig[1,2],"b",halign=:left,valign=:top,tellwidth=false,tellheight=false,padding=(-40,0,0,-40))
savefig("output/prcc_sensitivity_analysis",fig)

# Run the same analysis with different time points
ts = 7:7:70
n_t = length(ts)
outputs = run_simulation_multipoints.(p_sample,ts=ts)
outputs_mat = [outputs[i][j] for j in eachindex(outputs[1]), i in eachindex(outputs)]
prcc = partial_rank_correlation(p_sample_mat,outputs_mat)

p_grid = reshape(ps,(4,5))
dims = Base.size(p_grid)
fig = Figure(size=(1650,1000))
for i in 1:dims[1], j in 1:dims[2]
    p = p_grid[i,j]
    xlabel = (i == dims[1]) ? "Time (days)" : ""
    ylabel = (j == 1) ? "PRCC" : ""
    ax = Axis(fig[i,j],title=ps_latex[p],xlabel=xlabel,ylabel=ylabel)
    lines!(ax,ts,prcc[1:10,i+4*(j-1)],label="Bone volume fraction",linewidth=3,color=Makie.wong_colors()[1])
    lines!(ax,ts,prcc[n_t+1:end,i+4*(j-1)],label="Damage",linewidth=3,color=Makie.wong_colors()[4])
    ylims!(ax,-1.05,1.05)
end
fig[1,6] = Legend(fig,fig.content[1],framevisible=false,titlesize=24,labelsize=24)
savefig("output/prcc_sensitivity_analysis_time",fig)

## Run sensitivity analysis using Sobol indices
N = 10^5
p_range = [[bounds[p][1],bounds[p][2]] for p in ps]
push!(p_range,[0,1])
run_simulation_excl_dummy(p) = run_simulation(p[1:end-1])
sobol_indices = gsa(run_simulation_excl_dummy, Sobol(order=[0,1,2]), p_range, samples=N)

S1 = sobol_indices.S1
ST = sobol_indices.ST

idx_perm_fBM = sortperm(ST[1,1:np],rev=true)
_ps_fBM_sorted = [ps_latex[ps[i]] for i in idx_perm_fBM]
ps_fBM_sorted = [_ps_fBM_sorted;"Dummy"]
S1_fBM_sorted = [S1[1,idx_perm_fBM];S1[1,np+1]]
ST_fBM_sorted = [ST[1,idx_perm_fBM];ST[1,np+1]]
cat_fBM = [1:np+1; 1:np+1]
heights_fBM = [S1_fBM_sorted; ST_fBM_sorted-S1_fBM_sorted]
output_cats_fBM = [ones(Int,np+1); 2*ones(Int,np+1)]

idx_perm_dmg = sortperm(ST[2,1:np],rev=true)
_ps_dmg_sorted = [ps_latex[ps[i]] for i in idx_perm_dmg]
ps_dmg_sorted = [_ps_dmg_sorted;"Dummy"]
S1_dmg_sorted = [S1[2,idx_perm_dmg];S1[2,np+1]]
ST_dmg_sorted = [ST[2,idx_perm_dmg];ST[2,np+1]]
heights_dmg = [S1_dmg_sorted; ST_dmg_sorted-S1_dmg_sorted]
output_cats_dmg = [ones(Int,np+1); 2*ones(Int,np+1)]

function convert_to_light(col,lightness=0.9)
    _col = convert(HSL,col)
    col_light = HSL(_col.h,_col.s,lightness)
    return col_light
end
col_fBM = Makie.wong_colors()[1]
col_fBM_light = convert_to_light(col_fBM,0.85)
col_dmg = Makie.wong_colors()[4]
col_dmg_light = convert_to_light(col_dmg,0.85)
colors_fBM = [col_fBM,col_fBM_light][output_cats_fBM]
colors_dmg = [col_dmg,col_dmg_light][output_cats_dmg]

fig = Figure(size=(900,1000))
ga = fig[1,1] = GridLayout()
ax = Axis(
    ga[1,1],xlabel="Sobol index",
    yticks = (1:np+1, ps_fBM_sorted),
    yticklabelsize = 14,
    yreversed=true,
    title="Bone volume fraction", titlefont=:bold
)
barplot!(ax,cat_fBM,heights_fBM,stack=output_cats_fBM,color=colors_fBM,direction=:x)
hidespines!(ax,:t,:r)
labels = ["First-order", "Total-order"]
elements = [PolyElement(polycolor=col_fBM), PolyElement(polycolor=col_fBM_light)]
axislegend(ax,elements,labels,framevisible=false,position=:rc)

gb = fig[1,2] = GridLayout()
ax = Axis(
    gb[1,1],xlabel="Sobol index",
    yticks = (1:np+1, ps_dmg_sorted),
    yticklabelsize = 14,
    yreversed=true,
    title="Damage", titlefont=:bold
)
barplot!(ax,cat_fBM,heights_dmg,stack=output_cats_dmg,color=colors_dmg,direction=:x)
hidespines!(ax,:t,:r)
labels = ["First-order", "Total-order"]
elements = [PolyElement(polycolor=col_dmg), PolyElement(polycolor=col_dmg_light)]
axislegend(ax,elements,labels,framevisible=false,position=:rc)

xs_fBM = Float64[]
ys_fBM = Float64[]
zs_fBM = Float64[]
xs_dmg = Float64[]
ys_dmg = Float64[]
zs_dmg = Float64[]
for i in 1:np+1, j in 1:np+1 
    if i < j
        push!(xs_fBM,i)
        push!(ys_fBM,j)
        push!(zs_fBM,sobol_indices.S2[i,j,1])
        push!(xs_dmg,j)
        push!(ys_dmg,i)
        push!(zs_dmg,sobol_indices.S2[i,j,2])
    end
end

ps_axis = [[ps_latex[ps[i]] for i in 1:np];"Dummy"]
gc = fig[2,1:2] = GridLayout()
ax = Axis(
    gc[1,1],aspect=1,
    title="Second-order Sobol indices",
    xticks = (1:21,ps_axis),
    yticks = (1:21,ps_axis),
    xticklabelsize=12,
    yticklabelsize=12,
    xticklabelrotation = π/2,
    xticksvisible = false,
    yticksvisible = false,
    yreversed = true
)
hm1 = heatmap!(ax,xs_fBM,ys_fBM,zs_fBM,colormap=:Blues,colorrange=(0.0,0.31))
hm2 = heatmap!(ax,xs_dmg,ys_dmg,zs_dmg,colormap=:RdPu,colorrange=(0.0,0.31))
hidespines!(ax)
Colorbar(gc[1,2],hm1,label="Bone volume fraction")
Colorbar(gc[1,3],hm2,label="Damage")
Label(fig[1,1],"a",halign=:left,valign=:top,tellwidth=false,tellheight=false,padding=(-40,0,0,-40))
Label(fig[1,2],"b",halign=:left,valign=:top,tellwidth=false,tellheight=false,padding=(-40,0,0,-40))
Label(fig[2,1],"c",halign=:left,valign=:top,tellwidth=false,tellheight=false,padding=(-40,0,0,-40))
rowgap!(fig.layout,30)
savefig("output/sobol_sensitivity_analysis",fig)

# Run analysis again at different time points
ts = 7:7:70
n_t = length(ts)
N = 10^5
run_simulation_excl_dummy_multipoints(p) = run_simulation_multipoints(p[1:end-1],ts=ts)
sobol_indices = gsa(run_simulation_excl_dummy_multipoints, Sobol(order=[0,1,2]), p_range, samples=N)
S1 = sobol_indices.S1
ST = sobol_indices.ST

p_grid = reshape(ps,(4,5))
dims = Base.size(p_grid)
fig = Figure(size=(1650,1000))
for i in 1:dims[1], j in 1:dims[2]
    p = p_grid[i,j]
    xlabel = (i == dims[1]) ? "Time (days)" : ""
    ylabel = (j == 1) ? "Sobol index" : ""
    ax = Axis(fig[i,j],title=ps_latex[p],xlabel=xlabel,ylabel=ylabel)
    lines!(ax,ts,S1[1:10,i+4*(j-1)],label="BV/TV (First-order)",linewidth=3,linestyle=:dash,color=Makie.wong_colors()[1])
    lines!(ax,ts,ST[1:10,i+4*(j-1)],label="BV/TV (Total)",linewidth=3,color=Makie.wong_colors()[1])
    lines!(ax,ts,S1[n_t+1:end,i+4*(j-1)],label="Damage (First-order)",linewidth=3,linestyle=:dash,color=Makie.wong_colors()[4])
    lines!(ax,ts,ST[n_t+1:end,i+4*(j-1)],label="Damage (Total)",linewidth=3,color=Makie.wong_colors()[4])
    ylims!(ax,-0.05,1.05)
end
fig[1,6] = Legend(fig,fig.content[1],framevisible=false,titlesize=24,labelsize=24)
savefig("output/sobol_sensitivity_analysis_time",fig)