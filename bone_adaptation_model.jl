using ModelingToolkit, Polynomials
@parameters t
D = Differential(t)

"""
Function to create an ODE system for bone adaptation. The input is a tuple containing functions for the stress, strain rate, and strides per day.
If the input is `nothing`, the corresponding variable is treated as a parameter.
If the keyword argument damage_input is set to true, the damage rate is treated as the third input, in place of the n_strides input. The for bone damage dynamics is adjusted accordingly.
"""
function bone_adaptation_system(inputs; damage_input=false)
    eqs = bone_adaptation_system_eqs(inputs,damage_input)
    vs = ModelingToolkit.vars(eqs)
    ps = [v for v in vs if ModelingToolkit.isparameter(v) && ~isequal(v,t)]
    sts = [v for v in vs if ~ModelingToolkit.isparameter(v) && ~ModelingToolkit.isdifferential(v)]
    _sys = ODESystem(eqs, t, sts, ps, name=:bone)
    sys = structural_simplify(_sys)
    return sys
end

function create_parameter(name)
    return (ModelingToolkit.toparam)((Symbolics.wrap)((SymbolicUtils.setmetadata)((Sym){Real}(name), Symbolics.VariableSource, (:parameters, name))))
end
function create_variable(name)
    return (identity)((Symbolics.wrap)((SymbolicUtils.setmetadata)(((Sym){(SymbolicUtils.FnType){NTuple{1, Any}, Real}}(name))((Symbolics.value)(t)), Symbolics.VariableSource, (:variables, name))))
end

function bone_adaptation_system_eqs(inputs,damage_input)
    σ = isnothing(inputs[1]) ? create_parameter(:σ) : create_variable(:σ)
    strain_rate = isnothing(inputs[2]) ? create_parameter(:strain_rate) : create_variable(:strain_rate)
    if damage_input == false
        n_strides = isnothing(inputs[3]) ? create_parameter(:n_strides) : create_variable(:n_strides)
    else
        n_strides = create_variable(:damage_formation_rate)
    end

    eqs = [
        bone_adaptation_inputs(inputs...,σ,strain_rate,n_strides);
        surface_density_eq();
        bone_density_eq();
        strain_energy_density_eq(σ=σ);
        bone_stiffness_eqs(strain_rate=strain_rate);
        osteoblast_activity_eq();
        osteoclast_activity_eq();
        damage_rate_eq(σ=σ);
        damage_accumulation_eq(n_strides=n_strides,damage_input=damage_input)
    ]
    return eqs
end

# Define default parameters, variables and values
aSv_default = 11.42
bSv_default = -0.02
E0_default = 2500
γE_default = 0.06
A_OBL_min_default = 0.001094
A_OBL_max_default = 0.0127
δB_default = 7
γB_default = 1
A_OCL_min_default = 0.001
A_OCL_max_default = 0.011
δC_default = 1
γC_default = 2
E_nominal_default = 2500*0.9^3*0.36^0.06
σ0_default = 134.2
σ1_default = 14.1
α_default = 0.19
Fs_default = 5
@parameters aSv=aSv_default bSv=bSv_default E0=E0_default γE=γE_default α=α_default Fs=Fs_default
@parameters A_OBL_min=A_OBL_min_default A_OBL_max=A_OBL_max_default δB=δB_default γB=γB_default
@parameters A_OCL_min=A_OCL_min_default A_OCL_max=A_OCL_max_default δC=δC_default γC=γC_default
@parameters E_nominal=E_nominal_default σ0=σ0_default σ1=σ1_default
@variables fBM(t)=0.5 Sv(t) E(t) ψ(t) σ(t) E(t) strain_rate(t) A_OBL(t) A_OCL(t) dmg_rate(t) n_strides(t) dmg(t)=0.0

# Functions for relationship between bone volume fraction and surface area
# Equations from Hitchens et al. (2018; https://doi.org/10.1007/s10237-017-0998-z)
surface_density(fBM,a=aSv_default,b=bSv_default) = a*sqrt(max(1-fBM,0))*(b*fBM + 1-b)
surface_density_eq(Sv=Sv,fBM=fBM,a=aSv,b=bSv) = Sv ~ surface_density(fBM,a,b)

# Functions for the strain energy density. Assumes a linear relationship between stress and strain in 1D.
strain_energy_density(σ,E) = 1/(2*E) * σ^2
strain_energy_density_eq(;ψ=ψ,σ=σ,E=E) = ψ ~ strain_energy_density(σ,E)

# Function for the relationship for bone stiffness
loading_stiffness(fBM,strain_rate,E0=E0_default,γ=γE_default) = E0 * fBM^3 * strain_rate^γ
bone_stiffness_eqs(;E=E,fBM=fBM,strain_rate=strain_rate,E0=E0,γ=γE) = [
    E ~ loading_stiffness(fBM,strain_rate,E0,γ)
]

# Function for osteoblast activity
function osteoblast_activity(ψ,A_min=A_OBL_min_default,A_max=A_OBL_max_default,δ=δB_default,γ=γB_default) 
    return A_min + (A_max - A_min)*ψ^γ / (δ^γ + ψ^γ)
end
function osteoblast_activity_eq(A_OBL=A_OBL,ψ=ψ,A_min=A_OBL_min,A_max=A_OBL_max,δ=δB,γ=γB)
    return A_OBL ~ osteoblast_activity(ψ,A_min,A_max,δ,γ)
end

# Function for osteoclast activity
function osteoclast_activity(ψ,A_min=A_OCL_min_default,A_max=A_OCL_max_default,δ=δC_default,γ=γC_default) 
    return A_min + (A_max - A_min)*δ^γ / (δ^γ + ψ^γ)
end
function osteoclast_activity_eq(A_OCL=A_OCL,ψ=ψ,A_min=A_OCL_min,A_max=A_OCL_max,δ=δC,γ=γC)
    return A_OCL ~ osteoclast_activity(ψ,A_min,A_max,δ,γ)
end

# Function for fatigue life
fatigue_life(σ;σ0=σ0_default,σ1=σ1_default) = 10^(-(σ-σ0)/σ1)
adjusted_fatigue_life(σ;E=E_nominal_default,σ0=σ0_default,σ1=σ1_default,E_nominal=E_nominal_default) = fatigue_life(σ;σ0=σ0,σ1=σ1)*E/E_nominal
function damage_rate_eq(;dmg_rate=dmg_rate,σ=σ,E=E,E_nominal=E_nominal,σ0=σ0,σ1=σ1)
    return dmg_rate ~ 1/fatigue_life(σ,σ0=σ0,σ1=σ1)
end

# Equations for inputs
function bone_adaptation_inputs(f_σ,f_ϵ,f_strides,σ=σ,strain_rate=strain_rate,n_strides=n_strides)
    eqs = Equation[]
    if ~isnothing(f_σ)
        push!(eqs, σ ~ f_σ(t))
    end
    if ~isnothing(f_ϵ)
        push!(eqs, strain_rate ~ f_ϵ(t))
    end
    if ~isnothing(f_strides)
        push!(eqs, n_strides ~ f_strides(t))
    end
    return eqs
end

# Equation for bone density evolution
bone_density_eq(;fBM=fBM,A_OBL=A_OBL,A_OCL=A_OCL,α=α,Sv=Sv) = D(fBM) ~ (A_OBL - A_OCL)*α*Sv

# Equation for damage evolution
function damage_accumulation_eq(;dmg=dmg,A_OBL=A_OBL,A_OCL=A_OCL,α=α,Sv=Sv,fBM=fBM,Fs=Fs,n_strides=n_strides,E=E,E_nominal=E_nominal,damage_input=false)
    if damage_input == false 
        return D(dmg) ~ n_strides*dmg_rate*E_nominal/E - A_OBL*α*Sv*dmg/fBM - (Fs-1)*A_OCL*α*Sv*dmg/fBM
    else
        return D(dmg) ~ n_strides*E_nominal/E - A_OBL*α*Sv*dmg/fBM - (Fs-1)*A_OCL*α*Sv*dmg/fBM
    end
end

# Equation for the relationship between speed and joint stress (Witte et al., 2006)
function joint_stress(speed;scaled_load=90)
    scaling_factor = scaled_load/24.13
    return scaling_factor*vertical_force(speed)
end
function vertical_force(speed)
    return 2.778 + 2.1376*speed - 0.0535*speed^2
end

# Equation for the relationship between speed and maximum strain
function strain_rate_fitted(speed)
    return 0.015481010197084148*speed + 0.0004955110898565426*speed^2
end

# Calculate the stride length for a given speed
function stride_length(speed)
    stride_frequency = 1.7052+0.0305*speed+0.0004*speed^2 # Unit Hz
    stride_length = speed/stride_frequency # Unit m
    return stride_length
end

# Callback to end simulation when damage reaches 1
function termination_callback(sys)
    idx_dmg = variable_index(sys, dmg)
    condition(u,t,integrator) = u[idx_dmg] - 1
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition,affect!)
    return cb
end

"""Calculate the steady-state strain energy density"""
function SED_ss(;
    A_OBL_min=A_OBL_min_default,A_OBL_max=A_OBL_max_default,δB=δB_default,
    A_OCL_min=A_OCL_min_default,A_OCL_max=A_OCL_max_default,δC=δC_default)
    pB = Polynomial([δB,1])
    pC = Polynomial([δC^2,0,1])
    p = A_OBL_min*pB*pC + (A_OBL_max - A_OBL_min)*Polynomial([0,1])*pC - A_OCL_min*pB*pC -(A_OCL_max - A_OCL_min)*δC^2*pB
    ψ_ss = roots(p)[3].re
    return ψ_ss
end
"""Calculate the osteoblast and osteoclast activities"""
A_ss() = osteoblast_activity(SED_ss())
"""Calculate the steady-state bone volume fraction"""
function fBM_ss(σ,strain_rate;E0=E0_default,γE=γE_default,A_OBL_max=A_OBL_max_default,A_OCL_max=A_OCL_max_default) 
    return (σ^2/(2*E0*strain_rate^γE*SED_ss(A_OBL_max=A_OBL_max,A_OCL_max=A_OCL_max)))^(1/3)
end

function damage(distance,speed;σ0=σ0_default)
    n_strides = distance/stride_length(speed)
    return n_strides/fatigue_life(joint_stress(speed),σ0=σ0)
end

"""
Parameters obtained from fitting to data
"""
p_optimal() = [
    A_OBL_max => 0.006028609291464905
    A_OCL_max => 0.003582042256313746
    σ0 => 139.02301150354072
]