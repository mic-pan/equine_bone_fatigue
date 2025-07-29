# Supplementary code for "A mathematical model of metacarpal subchondral bone adaptation, microdamage, and repair in racehorses"

## Scripts
This repository contains the Julia code required to generate the figures for the study *A mathematical model of metacarpal subchondral bone adaptation, microdamage, and repair in racehorses*. The following files will generate the results figures as indicated below:
| Script                                 | Figures              |
|----------------------------------------|----------------------|
|`model_fitting.jl`                      | Fig. 3               |
|`model_sensitivity.jl`                  | Figs. 4,S7,S9,S10    |
|`detailed_training_program_results.jl`  | Fig. 6               |
|`strain_speed.jl`                       | Fig. S2              |
|`bone_remodelling.jl`                   | Figs. S3-S6          |
|`adaptation_hill_figures.jl`            | Fig. S8              |

## Instructions for running code
The code can be run using a Julia installation, using the steps below. The code has been tested on Julia 1.11.1.

1. Install Julia (https://julialang.org/downloads/) and Git (https://git-scm.com/downloads)
2. Clone this repository locally by running the following in the terminal: `git clone https://github.com/mic-pan/equine_bone_fatigue`
3. Start Julia: `julia`
4. Change into the code directory: `cd("equine_bone_fatigue")`
5. Press the `]` key to enter the package manager
6. Activate the environment: `activate env`
7. Install the required packages: `instantiate`
8. Exit the package manager by pressing the backspace key
9. The scripts can be run by using the command `include("script_name.jl")`
