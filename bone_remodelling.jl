using ModelingToolkit, DifferentialEquations, CairoMakie, Sundials, LaTeXStrings, SymbolicIndexingInterface
include("bone_adaptation_model.jl")
include("makie_functions.jl")
set_theme!(makie_theme)

sys = bone_adaptation_system((nothing,nothing,nothing))
cb = termination_callback(sys)

@parameters σ strain_rate n_strides
σs = [30, 42, 54, 66, 78, 90, 102]
srs = [0.02, 0.05, 0.12, 0.15, 0.26, 0.36, 0.46]
cycles_per_day = 40
n_states = length(σs)
tf_adapted_vec = zeros(n_states)
grad = ["#036ca9","#586caa","#816bab","#a268ac","#c163ae","#df5caf","#fc51b0"]
tspan = (0.0,200.0)

p1 = Figure(size=(1200,650))
ax11 = Axis(p1[1,1], xlabel="Time (days)", ylabel="Bone volume fraction")
ax12 = Axis(p1[1,2], xlabel="Time (days)", ylabel=L"\psi_\text{tissue}\ (\text{MPa})")
ax13 = Axis(p1[1,3], xlabel="Time (days)", ylabel="Damage")
for (i,(sigma,sr,c)) in enumerate(zip(σs,srs,grad))
    ps = vcat([σ=>sigma, strain_rate=>sr, n_strides=>cycles_per_day],p_optimal())
    prob = ODEProblem(sys,[fBM => 0.9],tspan,ps,reltol=1e-9,abstol=1e-12)
    sol = solve(prob, CVODE_BDF(), callback = cb)
    tf_adapted_vec[i] = sol.t[end]
    lines!(ax11, sol.t, sol[fBM], label="σ=$sigma MPa", color=c)
    lines!(ax12, sol.t, sol[ψ], label="σ=$sigma MPa", color=c)
    lines!(ax13, sol.t, sol[dmg], label="σ=$sigma MPa", color=c)
end
ax21 = Axis(p1[3,1], xlabel="Time (days)", ylabel="Bone volume fraction")
ax22 = Axis(p1[3,2], xlabel="Time (days)", ylabel=L"\psi_\text{tissue}\ (\text{MPa})")
ax23 = Axis(p1[3,3], xlabel="Time (days)", ylabel="Damage")
for (i,(sigma,sr,c)) in enumerate(zip(σs,srs,grad))
    ps = vcat([σ=>sigma, strain_rate=>sr, n_strides=>cycles_per_day],p_optimal())
    prob = ODEProblem(sys,[fBM => 0.7],tspan,ps,reltol=1e-9,abstol=1e-12)
    sol = solve(prob, CVODE_BDF(), callback = cb)
    tf_adapted_vec[i] = sol.t[end]
    lines!(ax21, sol.t, sol[fBM], label="σ=$sigma MPa", color=c)
    lines!(ax22, sol.t, sol[ψ], label="σ=$sigma MPa", color=c)
    lines!(ax23, sol.t, sol[dmg], label="σ=$sigma MPa", color=c)
end
p1[1, 4] = Legend(p1, ax11, "", framevisible = false)
Label(p1[0,1],"a",halign=:left,valign=:top,tellwidth=false,padding=(-50,0,0,0))
Label(p1[0,2],"b",halign=:left,valign=:top,tellwidth=false,padding=(-50,0,0,0))
Label(p1[0,3],"c",halign=:left,valign=:top,tellwidth=false,padding=(-50,0,0,0))
Label(p1[2,1],"d",halign=:left,valign=:top,tellwidth=false,padding=(-50,0,0,0))
Label(p1[2,2],"e",halign=:left,valign=:top,tellwidth=false,padding=(-50,0,0,0))
Label(p1[2,3],"f",halign=:left,valign=:top,tellwidth=false,padding=(-50,0,0,0))
savefig("output/damage_adaptation",p1)

## Run simulations to steady state
tspan_long = (0.0,1000.0)
fig = Figure(size=(500,300))
ax = Axis(fig[1,1], xlabel="Time (days)", ylabel="Bone volume fraction")
for (i,(sigma,sr,c)) in enumerate(zip(σs,srs,grad))
    ps = vcat([σ=>sigma, strain_rate=>sr, n_strides=>cycles_per_day],p_optimal())
    prob1 = ODEProblem(sys,[fBM => 0.9],tspan_long,ps,reltol=1e-9,abstol=1e-12)
    sol1 = solve(prob1, CVODE_BDF())
    prob2 = ODEProblem(sys,[fBM => 0.7],tspan_long,ps,reltol=1e-9,abstol=1e-12)
    sol2 = solve(prob2, CVODE_BDF())
    lines!(ax, sol1.t, sol1[fBM], label="σ=$sigma MPa", color=c)
    lines!(ax, sol2.t, sol2[fBM], label="σ=$sigma MPa", color=c)
end
fig[1, 2] = Legend(fig, ax, "", framevisible = false, merge=true)
savefig("output/adaptation_steady_state",fig)

## Run some simulations with varying cycles per day
cycles_per_day = 10:10:100
n_sims = length(cycles_per_day)
grad = ["#036ca9","#476caa","#676caa","#816bab","#9869ac","#ad67ad","#c163ae","#d55fae","#e959af","#fc51b0"]
tspan = (0.0,200.0)

fig = Figure(size=(900,350))
ax1 = Axis(fig[1,1], xlabel="Time (days)", ylabel="Bone volume fraction")
ax2 = Axis(fig[1,2], xlabel="Time (days)", ylabel="Damage")
for (i,(n,c)) in enumerate(zip(cycles_per_day,grad))
    prob = ODEProblem(sys,[fBM => 0.7],tspan,vcat([σ=>78, strain_rate=>0.26, n_strides=>n],p_optimal()),reltol=1e-9,abstol=1e-12)
    sol = solve(prob, CVODE_BDF(), callback = cb)
    lines!(ax1, sol.t, sol[fBM], label="$n", color=:black)
    lines!(ax2, sol.t, sol[dmg], label="$n", color=c)
end
fig[1, 3] = Legend(fig, ax2, "Cycles per day", framevisible = false)
Label(fig[0,1],"a",halign=:left,valign=:top,tellwidth=false,padding=(-50,0,0,0))
Label(fig[0,2],"b",halign=:left,valign=:top,tellwidth=false,padding=(-50,0,0,0))
savefig("output/damage_adaptation_cycles_per_day",fig)

## Run same simulations over a longer duration
cycles_per_day = 10:10:100
n_sims = length(cycles_per_day)
grad = ["#036ca9","#476caa","#676caa","#816bab","#9869ac","#ad67ad","#c163ae","#d55fae","#e959af","#fc51b0"]
tspan_long = (0.0,5000.0)

fig = Figure(size=(900,350))
ax1 = Axis(fig[1,1], xlabel="Time (days)", ylabel="Bone volume fraction")
ax2 = Axis(fig[1,2], xlabel="Time (days)", ylabel="Damage")
for (i,(n,c)) in enumerate(zip(cycles_per_day,grad))
    prob = ODEProblem(sys,[fBM => 0.7],tspan_long,vcat([σ=>78, strain_rate=>0.26, n_strides=>n],p_optimal()),reltol=1e-9,abstol=1e-12)
    sol = solve(prob, CVODE_BDF(), callback = cb)
    lines!(ax1, sol.t, sol[fBM], label="$n", color=:black)
    lines!(ax2, sol.t, sol[dmg], label="$n", color=c)
end
fig[1, 3] = Legend(fig, ax2, "Cycles per day", framevisible = false)
Label(fig[0,1],"a",halign=:left,valign=:top,tellwidth=false,padding=(-50,0,0,0))
Label(fig[0,2],"b",halign=:left,valign=:top,tellwidth=false,padding=(-50,0,0,0))
savefig("output/damage_adaptation_cycles_per_day_long",fig)

## Run simulations with high stress and low stride number
prob = ODEProblem(sys,[fBM=>0.5],tspan,vcat([σ=>90, strain_rate=>0.36, n_strides=>3],p_optimal()),reltol=1e-9,abstol=1e-12)
sol = solve(prob, CVODE_BDF(), callback = cb)
fig = Figure(size=(700,500))
ax1 = Axis(fig[1,1], xlabel="Time (days)", ylabel="Bone volume fraction")
ax2 = Axis(fig[1,2], xlabel="Time (days)", ylabel="ψ")
ax3 = Axis(fig[2,1], xlabel="Time (days)", ylabel="E")
ax4 = Axis(fig[2,2], xlabel="Time (days)", ylabel="Damage")
lines!(ax1, sol.t, sol[fBM])
lines!(ax2, sol.t, sol[ψ])
lines!(ax3, sol.t, sol[E])
lines!(ax4, sol.t, sol[dmg])
savefig("output/low_stride_high_stress",fig)