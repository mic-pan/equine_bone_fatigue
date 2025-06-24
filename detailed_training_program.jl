include("bone_adaptation_model.jl")

pulse(x,a,b) = (a<x)*(x<=b)
ramp(x,a,b) = (x-a)*pulse(x,a,b)
σ0_fitted = p_optimal()[3][2]
adjusted_damage(distance,speed,σ0=σ0_fitted) = damage(distance,max(1.6,speed),σ0=σ0)

abstract type TrainingPhase end

struct HomegeneousProgram <: TrainingPhase
    duration::Float64
    distance_per_day::Float64
    speed::Float64
end
speedfunc(hp::HomegeneousProgram) = t -> hp.speed*pulse(t,0,hp.duration)
distancefunc(hp::HomegeneousProgram) = t -> hp.distance_per_day*pulse(t,0,hp.duration)
damagefunc(hp::TrainingPhase;σ0=σ0_fitted) = t -> adjusted_damage(distancefunc(hp)(t),speedfunc(hp)(t),σ0)

struct SlowProgressiveProgram <: TrainingPhase
    duration::Float64
    distance_per_day_start::Float64
    distance_per_day_end::Float64
    speed_start::Float64
    speed_end::Float64
end
speedfunc(spp::SlowProgressiveProgram) = t -> (
    spp.speed_start*pulse(t,0,spp.duration) + 
    (spp.speed_end-spp.speed_start)/spp.duration*ramp(t,0,spp.duration)
)
distancefunc(spp::SlowProgressiveProgram) = t -> (
    spp.distance_per_day_start*pulse(t,0,spp.duration) + 
    (spp.distance_per_day_end-spp.distance_per_day_start)/spp.duration*ramp(t,0,spp.duration)
)

struct FastProgressiveProgram <: TrainingPhase
    duration::Float64
    cumulative_distance1::Float64
    cumulative_distance2::Float64
    speed1::Float64
    speed2::Float64
end
speedfunc1(fpp::FastProgressiveProgram) = t -> fpp.speed1*pulse(t,0,fpp.duration)
speedfunc2(fpp::FastProgressiveProgram) = t -> fpp.speed2*pulse(t,0,fpp.duration)
speedfunc(fpp::FastProgressiveProgram) = t -> max(speedfunc1(fpp)(t), speedfunc2(fpp)(t))
distancefunc1(fpp::FastProgressiveProgram) = t -> (
    (fpp.cumulative_distance1+fpp.cumulative_distance2)/fpp.duration*pulse(t,0,fpp.duration) - 
    distancefunc2(fpp)(t)
)
distancefunc2(fpp::FastProgressiveProgram) = t -> fpp.cumulative_distance2*2/fpp.duration^2*ramp(t,0,fpp.duration)
distancefunc(fpp::FastProgressiveProgram) = t -> distancefunc1(fpp)(t) + distancefunc2(fpp)(t)
damagefunc1(fpp::FastProgressiveProgram;σ0=σ0_fitted) = t -> adjusted_damage(distancefunc1(fpp)(t),speedfunc1(fpp)(t),σ0)
damagefunc2(fpp::FastProgressiveProgram;σ0=σ0_fitted) = t -> adjusted_damage(distancefunc2(fpp)(t),speedfunc2(fpp)(t),σ0)
damagefunc(fpp::FastProgressiveProgram;σ0=σ0_fitted) = t -> damagefunc1(fpp,σ0=σ0)(t) + damagefunc2(fpp,σ0=σ0)(t)

struct RacefitProgram <: TrainingPhase
    duration::Float64
    slow_gallop_distance_per_day::Float64
    fast_gallop_distance_per_day::Float64
    race_distance_per_day::Float64
    slow_gallop_speed::Float64
    fast_gallop_speed::Float64
    race_speed::Float64
end
slowgallop_program(rp::RacefitProgram) = HomegeneousProgram(rp.duration,rp.slow_gallop_distance_per_day,rp.slow_gallop_speed)
fastgallop_program(rp::RacefitProgram) = HomegeneousProgram(rp.duration,rp.fast_gallop_distance_per_day,rp.fast_gallop_speed)
race_program(rp::RacefitProgram) = HomegeneousProgram(rp.duration,rp.race_distance_per_day,rp.race_speed)
speedfunc(rp::RacefitProgram) = t -> rp.race_speed*pulse(t,0,rp.duration)
slowgallop_distancefunc(rp::RacefitProgram) = distancefunc(slowgallop_program(rp))
fastgallop_distancefunc(rp::RacefitProgram) = distancefunc(fastgallop_program(rp))
race_distancefunc(rp::RacefitProgram) = distancefunc(race_program(rp))
distancefunc(rp::RacefitProgram) = t -> slowgallop_distancefunc(rp)(t) + fastgallop_distancefunc(rp)(t) + race_distancefunc(rp)(t)
slowgallop_damagefunc(rp::RacefitProgram;σ0=σ0_fitted) = damagefunc(slowgallop_program(rp),σ0=σ0)
fastgallop_damagefunc(rp::RacefitProgram;σ0=σ0_fitted) = damagefunc(fastgallop_program(rp),σ0=σ0)
race_damagefunc(rp::RacefitProgram;σ0=σ0_fitted) = damagefunc(race_program(rp),σ0=σ0)
damagefunc(rp::RacefitProgram;σ0=σ0_fitted) = t -> slowgallop_damagefunc(rp,σ0=σ0)(t) + fastgallop_damagefunc(rp,σ0=σ0)(t) + race_damagefunc(rp,σ0=σ0)(t)

struct TrainingProgram
    rest_program::HomegeneousProgram
    pretraining_program::HomegeneousProgram
    slow_progressive_program::SlowProgressiveProgram
    fast_progressive_program::FastProgressiveProgram
    racefit_program::RacefitProgram
    slow_workout::HomegeneousProgram
end
progressive_program_duration(tp::TrainingProgram) = tp.slow_progressive_program.duration + tp.fast_progressive_program.duration
total_program_duration(tp::TrainingProgram) = tp.rest_program.duration + tp.pretraining_program.duration + progressive_program_duration(tp) + tp.racefit_program.duration
pretraining_start_time(tp::TrainingProgram) = tp.rest_program.duration
slow_progressive_start_time(tp::TrainingProgram) = pretraining_start_time(tp) + tp.pretraining_program.duration
fast_progressive_start_time(tp::TrainingProgram) = slow_progressive_start_time(tp) + tp.slow_progressive_program.duration
racefit_start_time(tp::TrainingProgram) = fast_progressive_start_time(tp) + tp.fast_progressive_program.duration
speedfunc(tp::TrainingProgram) = t -> (
    tp.rest_program.speed*pulse(t,-Inf,0) +
    speedfunc(tp.rest_program)(t) + 
    speedfunc(tp.pretraining_program)(t-pretraining_start_time(tp)) + 
    speedfunc(tp.slow_progressive_program)(t-slow_progressive_start_time(tp)) + 
    speedfunc(tp.fast_progressive_program)(t-fast_progressive_start_time(tp)) + 
    speedfunc(tp.racefit_program)(t-racefit_start_time(tp))
)
rest_distancefunc(tp::TrainingProgram) = distancefunc(tp.rest_program)
pretraining_distancefunc(tp::TrainingProgram) = t -> distancefunc(tp.pretraining_program)(t-pretraining_start_time(tp))
slowprogressive_distancefunc(tp::TrainingProgram) = t -> distancefunc(tp.slow_progressive_program)(t-slow_progressive_start_time(tp))
fastprogressive_distancefunc(tp::TrainingProgram) = t -> distancefunc(tp.fast_progressive_program)(t-fast_progressive_start_time(tp))
racefit_distancefunc(tp::TrainingProgram) = t -> distancefunc(tp.racefit_program)(t-racefit_start_time(tp))
slowworkout_distancefunc(tp::TrainingProgram) = t -> distancefunc(tp.slow_workout)(t-slow_progressive_start_time(tp))
distancefunc(tp::TrainingProgram) = t -> (
    tp.rest_program.distance_per_day*pulse(t,-Inf,0) + rest_distancefunc(tp)(t) + pretraining_distancefunc(tp)(t) + 
    slowprogressive_distancefunc(tp)(t) + fastprogressive_distancefunc(tp)(t) + racefit_distancefunc(tp)(t) +
    slowworkout_distancefunc(tp)(t)
)
gallop_distancefunc(tp::TrainingProgram) = t -> (
    distancefunc1(tp.fast_progressive_program)(t-fast_progressive_start_time(tp)) +
    slowgallop_distancefunc(tp.racefit_program)(t-racefit_start_time(tp))
)
racespeed_distancefunc(tp::TrainingProgram) = t -> (
    distancefunc2(tp.fast_progressive_program)(t-fast_progressive_start_time(tp)) +
    fastgallop_distancefunc(tp.racefit_program)(t-racefit_start_time(tp)) +
    race_distancefunc(tp.racefit_program)(t-racefit_start_time(tp))
)
rest_damagefunc(tp::TrainingProgram) = (t,σ0) -> damagefunc(tp.rest_program,σ0=σ0)(t)
pretraining_damagefunc(tp::TrainingProgram) = (t,σ0) -> damagefunc(tp.pretraining_program,σ0=σ0)(t-pretraining_start_time(tp))
slowprogressive_damagefunc(tp::TrainingProgram) = (t,σ0) -> damagefunc(tp.slow_progressive_program,σ0=σ0)(t-slow_progressive_start_time(tp))
fastprogressive_damagefunc(tp::TrainingProgram) = (t,σ0) -> damagefunc(tp.fast_progressive_program,σ0=σ0)(t-fast_progressive_start_time(tp))
racefit_damagefunc(tp::TrainingProgram) = (t,σ0) -> damagefunc(tp.racefit_program,σ0=σ0)(t-racefit_start_time(tp))
slowworkout_damagefunc(tp::TrainingProgram) = (t,σ0) -> damagefunc(tp.slow_workout,σ0=σ0)(t-slow_progressive_start_time(tp))
damagefunc(tp::TrainingProgram;σ0=σ0_fitted) = t -> (
    adjusted_damage(tp.rest_program.distance_per_day,tp.rest_program.speed,σ0)*pulse(t,-Inf,0) +
    rest_damagefunc(tp)(t,σ0) + pretraining_damagefunc(tp)(t,σ0) + slowprogressive_damagefunc(tp)(t,σ0) +
    fastprogressive_damagefunc(tp)(t,σ0) + racefit_damagefunc(tp)(t,σ0) + slowworkout_damagefunc(tp)(t,σ0)
)
canter_damagefunc(tp) = t -> slowworkout_damagefunc(tp)(t,σ0_fitted)
transitionspeed_damagefunc(tp) = t -> slowprogressive_damagefunc(tp)(t,σ0_fitted)
gallop_damagefunc(tp::TrainingProgram;σ0=σ0_fitted) = t -> (
    damagefunc1(tp.fast_progressive_program,σ0=σ0)(t-fast_progressive_start_time(tp)) +
    slowgallop_damagefunc(tp.racefit_program,σ0=σ0)(t-racefit_start_time(tp))
)
racespeed_damagefunc(tp::TrainingProgram;σ0=σ0_fitted) = t -> (
    damagefunc2(tp.fast_progressive_program,σ0=σ0)(t-fast_progressive_start_time(tp)) +
    fastgallop_damagefunc(tp.racefit_program,σ0=σ0)(t-racefit_start_time(tp)) +
    race_damagefunc(tp.racefit_program,σ0=σ0)(t-racefit_start_time(tp))
)
multiseason(tp::TrainingProgram,f::Function) = t -> f(tp)(rem(t,total_program_duration(tp)))

function construct_training_program(;
    rest_duration=44, rest_distance_per_day=4000, rest_speed=3.6,
    pretraining_duration=4*7, pretraining_distance_per_day=2000, pretraining_speed=7.5,
    slow_progressive_duration=4*7,slow_progressive_distance_per_day_start=900/7, slow_progressive_distance_per_day_end=1800/7,
    slow_progressive_speed_start=11.8,slow_progressive_speed_end=13.8,
    fast_progressive_duration=38, fast_progressive_cumulative_distance1=6600, fast_progressive_cumulative_distance2=3200,
    fast_progressive_speed1=13.8, fast_progressive_speed2=16.0,
    racefit_duration=4*2*7, racefit_slow_gallop_distance_per_day=4800/30, racefit_fast_gallop_distance_per_day=3200/30, racefit_race_distance_per_day=800/7,
    racefit_slow_gallop_speed=13.8, racefit_fast_gallop_speed=16.0, racefit_race_speed=16.7,
    slow_workout_distance_per_day=2000, slow_workout_speed=7.5
    )

    rest = HomegeneousProgram(rest_duration,rest_distance_per_day,rest_speed)
    pretraining = HomegeneousProgram(pretraining_duration,pretraining_distance_per_day,pretraining_speed)
    slow_progressive = SlowProgressiveProgram(
        slow_progressive_duration,
        slow_progressive_distance_per_day_start,
        slow_progressive_distance_per_day_end,
        slow_progressive_speed_start,
        slow_progressive_speed_end
    )
    fast_progressive = FastProgressiveProgram(
        fast_progressive_duration,
        fast_progressive_cumulative_distance1,
        fast_progressive_cumulative_distance2,
        fast_progressive_speed1,
        fast_progressive_speed2
    )
    racefit = RacefitProgram(
        racefit_duration,
        racefit_slow_gallop_distance_per_day,
        racefit_fast_gallop_distance_per_day,
        racefit_race_distance_per_day,
        racefit_slow_gallop_speed,
        racefit_fast_gallop_speed,
        racefit_race_speed
    )
    slow_workout_duration = slow_progressive_duration + fast_progressive_duration + racefit_duration
    slow_workout = HomegeneousProgram(slow_workout_duration,slow_workout_distance_per_day,slow_workout_speed)
    tp = TrainingProgram(rest,pretraining,slow_progressive,fast_progressive,racefit,slow_workout)
    return tp
end

function ModelingToolkit.ODESystem(tp::TrainingProgram)
    multiseason_speed = multiseason(tp,speedfunc)
    multiseason_damage = multiseason(tp,damagefunc)
    f_σ(t) = joint_stress(multiseason_speed(t))
    f_ϵ(t) = strain_rate_fitted(multiseason_speed(t))
    f_Dr(t) = multiseason_damage(t)
    sys = bone_adaptation_system((f_σ,f_ϵ,f_Dr),damage_input=true)
    return sys
end