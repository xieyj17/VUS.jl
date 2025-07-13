module VUS

using DataFrames
using CSV
using Statistics
using StatsBase
using GLM
using MLBase
using Random
using Plots
using ProgressMeter

export clean_nba_data, generate_binned_df, get_vus, plot_vus, 
       fit_dynamic_benchmark, prepare_benchmark_data,
       bootstrap_vus, plot_auc_surface_with_confidence

# Constants
const QUARTER_IN_SEC = 12 * 60
const TEAM_NAME_MAP = Dict(
    "ATL" => "atlanta-hawks",
    "BOS" => "boston-celtics", 
    "BRK" => "brooklyn-nets",
    "CHO" => "charlotte-hornets",
    "CHI" => "chicago-bulls",
    "CLE" => "cleveland-cavaliers",
    "DAL" => "dallas-mavericks",
    "DEN" => "denver-nuggets",
    "DET" => "detroit-pistons",
    "GSW" => "golden-state-warriors",
    "HOU" => "houston-rockets",
    "IND" => "indiana-pacers",
    "LAC" => "los-angeles-clippers",
    "LAL" => "los-angeles-lakers",
    "MEM" => "memphis-grizzlies",
    "MIA" => "miami-heat",
    "MIL" => "milwaukee-bucks",
    "MIN" => "minnesota-timberwolves",
    "NOP" => "new-orleans-pelicans",
    "NYK" => "new-york-knicks",
    "OKC" => "oklahoma-city-thunder",
    "ORL" => "orlando-magic",
    "PHI" => "philadelphia-76ers",
    "PHO" => "phoenix-suns",
    "POR" => "portland-trail-blazers",
    "SAC" => "sacramento-kings",
    "SAS" => "san-antonio-spurs",
    "TOR" => "toronto-raptors",
    "UTA" => "utah-jazz",
    "WAS" => "washington-wizards"
)

# Utils functions
function time_to_seconds(time_str)
    if isa(time_str, Real)
        return Float64(time_str)
    end
    
    time_str = string(time_str)
    m = match(r"(\d+):(\d+)", time_str)
    if m !== nothing
        minutes = parse(Int, m.captures[1])
        seconds = parse(Int, m.captures[2])
        return Float64(minutes * 60 + seconds)
    end
    
    return parse(Float64, time_str)
end

function time_left_in_quarter(second)
    return QUARTER_IN_SEC - second
end

function normalize_time!(df::DataFrame)
    # Handle missing values in clock column
    df.current_quarter_seconds_passed = [ismissing(x) ? missing : time_left_in_quarter(time_to_seconds(x)) for x in df.clock]
    
    # Calculate seconds passed, handling missing values
    df.second_passed = [(ismissing(q) || ismissing(s)) ? missing : (q - 1) * QUARTER_IN_SEC + s 
                       for (q, s) in zip(df.Quarter, df.current_quarter_seconds_passed)]
    
    # Calculate normalized time, handling missing values
    df.normalized_time = [ismissing(s) ? missing : s / (4 * QUARTER_IN_SEC) for s in df.second_passed]
    
    # Remove rows with missing normalized_time
    dropmissing!(df, :normalized_time)
    
    return df
end

function actual_results!(df::DataFrame)
    # Convert play_id to numeric, handling missing values
    df.play_id = [ismissing(x) ? missing : tryparse(Float64, string(x)) for x in df.play_id]
    dropmissing!(df, :play_id)
    
    # Find final game states
    final_indices = combine(groupby(df, :game_num), :play_id => argmax => :idx)
    final_game_states = df[final_indices.idx, :]
    
    # Calculate actual results
    final_game_states.actual_result = [ismissing(row.home) || ismissing(row.away) ? missing : 
                                      (row.home > row.away ? 1 : 0) for row in eachrow(final_game_states)]
    
    # Create mapping and apply to full dataframe
    result_map = Dict(zip(final_game_states.game_num, final_game_states.actual_result))
    df.actual_result = [get(result_map, game_num, missing) for game_num in df.game_num]
    
    # Remove rows with missing actual_result
    dropmissing!(df, :actual_result)
    
    # Sort by game and time
    sort!(df, [:game_num, :normalized_time])
    return df
end

function clean_nba_data(input_df::DataFrame)
    df = copy(input_df)
    
    # Check if required columns exist
    required_cols = [:game_num, :Quarter, :clock, :home_WP, :home, :away]
    for col in required_cols
        if !hasproperty(df, col)
            error("Required column $col not found in DataFrame")
        end
    end
    
    normalize_time!(df)
    actual_results!(df)
    
    # Filter and select columns
    df = df[df.Quarter .<= 4, :]  # Remove overtime
    
    # Ensure all required columns exist before selecting
    available_cols = [:game_num, :normalized_time, :home_WP, :actual_result, :home, :away]
    df = df[:, available_cols]
    df = unique(df)
    
    # Final cleanup of missing values
    dropmissing!(df, [:normalized_time, :actual_result])
    
    return df
end

# VUS functions
function generate_binned_df(df::DataFrame; num_bins::Int=100)
    # Remove any rows with missing normalized_time
    df_clean = dropmissing(df, :normalized_time)
    
    if nrow(df_clean) == 0
        error("No valid data after removing missing values")
    end
    
    # Create quantile-based bins using StatsBase.quantile
    quantiles = quantile(df_clean.normalized_time, range(0, 1, length=num_bins+1))
    
    # Assign bins based on quantiles
    df_clean.time_bin = [findfirst(x -> val <= x, quantiles[2:end]) for val in df_clean.normalized_time]
    
    # Handle edge case where value equals maximum
    df_clean.time_bin = [isnothing(bin) ? num_bins : bin for bin in df_clean.time_bin]
    
    # DON'T group by game_num - keep all individual observations
    # Just add the time_bin column and return the data
    binned_df = df_clean[:, [:game_num, :time_bin, :home_WP, :actual_result]]
    rename!(binned_df, :home_WP => :avg_home_WP)  # Rename for consistency
    
    # Calculate bin widths 
    bin_widths = diff(quantiles)
    bin_width_map = Dict(i => bin_widths[i] for i in 1:length(bin_widths))
    
    return binned_df, bin_width_map
end

# ROC/AUC calculation using ROC.jl package
function calculate_roc_and_auc(binned_df::DataFrame, bin_index::Int)
    slice_df = binned_df[binned_df.time_bin .== bin_index, :]
    
    if nrow(slice_df) < 2
        return NaN, nothing, nothing
    end
    
    try
        # Convert to vectors and handle missing values
        valid_rows = completecases(slice_df[:, [:mode_actual_result, :avg_home_WP]])
        if sum(valid_rows) < 2
            return NaN, nothing, nothing
        end
        
        y_true = slice_df.mode_actual_result[valid_rows]
        y_scores = slice_df.avg_home_WP[valid_rows]
        
        # Check for variability in both true labels and scores
        if length(unique(y_true)) < 2
            return NaN, nothing, nothing
        end
        
        if length(unique(y_scores)) < 2
            return 0.5, [0.0, 1.0], [0.0, 1.0]  # Random performance
        end
        
        # Convert to Boolean for ROC.jl (assumes 1 = positive, 0 = negative)
        y_true_bool = y_true .== 1
        
        # Calculate ROC using ROC.jl
        roc_result = ROC.roc(y_true_bool, y_scores)
        auc_val = ROC.AUC(roc_result)
        
        # Extract FPR and TPR for plotting
        fpr = [pt.FPR for pt in roc_result.points]
        tpr = [pt.TPR for pt in roc_result.points]
        
        return auc_val, fpr, tpr
        
    catch e
        println("Error in ROC calculation for bin $bin_index: $e")
        return NaN, nothing, nothing
    end
end

function get_vus(binned_df::DataFrame, bin_width_map::Dict; num_bins::Int=100)
    vus = 0.0
    auc_list = Float64[]
    valid_aucs = 0
    
    for bin_index in 1:num_bins
        if bin_index in binned_df.time_bin
            roc_auc, _, _ = calculate_roc_and_auc(binned_df, bin_index)
            if !isnan(roc_auc)
                bin_width = get(bin_width_map, bin_index, 0.0)
                vus += roc_auc * bin_width
                valid_aucs += 1
            end
            push!(auc_list, roc_auc)
        else
            push!(auc_list, NaN)
        end
    end
    
    return vus, auc_list
end

function plot_vus(binned_df::DataFrame, bin_width_map::Dict; num_bins::Int=100)
    println("Generating 3D plot...")
    
    time_bins = sort(unique(binned_df.time_bin))
    
    # Create 3D plot
    p = plot3d(xlabel="Time Bin", ylabel="False Positive Rate", zlabel="True Positive Rate",
              title="ROC Curves by Time Bin", size=(800, 600))
    
    for bin_index in time_bins
        _, fpr, tpr = calculate_roc_and_auc(binned_df, bin_index)
        
        if fpr !== nothing && tpr !== nothing
            x_coords = fill(bin_index, length(fpr))
            plot3d!(p, x_coords, fpr, tpr, label="Bin $bin_index")
        end
    end
    
    # Add reference plane for random chance
    x_range = 0:num_bins
    y_range = 0:0.1:1
    surface!(p, x_range, y_range, (x,y) -> y, alpha=0.2, color=:navy)
    
    display(p)
end

# Benchmark functions
function fit_dynamic_benchmark(df::DataFrame, model_features::Vector{String})
    model_df = copy(df)
    model_df.home_WP = fill(0.5, nrow(model_df))  # Initialize with default
    
    time_bins = sort(unique(model_df.time_bin))
    
    println("Fitting model with features: $model_features across $(length(time_bins)) time bins...")
    
    for bin_index in time_bins
        bin_data = model_df[model_df.time_bin .== bin_index, :]
        
        # Check for sufficient data
        if nrow(bin_data) < 10 || length(unique(bin_data.actual_result)) < 2
            continue
        end
        
        # Prepare data for logistic regression
        X = Matrix(bin_data[:, model_features])
        y = bin_data.actual_result
        
        # Fit logistic regression
        try
            model = glm(hcat(ones(size(X, 1)), X), y, Binomial(), LogitLink())
            predicted_probs = predict(model)
            
            # Update predictions for this bin
            mask = model_df.time_bin .== bin_index
            model_df.home_WP[mask] = predicted_probs
        catch e
            println("Error fitting model for bin $bin_index: $e")
        end
    end
    
    return model_df
end

function prepare_benchmark_data(year::String="2018"; num_bins::Int=100)
    # Load and clean game data
    game_df = CSV.read("nba_$(year).csv", DataFrame)
    cleaned_df = clean_nba_data(game_df)
    
    # Prepare lookup columns
    cols_to_relink = [:game_num]
    for col in [:game_date, :home_team, :away_team]
        if hasproperty(game_df, col)
            push!(cols_to_relink, col)
        end
    end
    
    lookup_df = unique(game_df[:, cols_to_relink], :game_num)
    full_game_df = leftjoin(cleaned_df, lookup_df, on=:game_num)
    
    # Load and merge ELO data
    elo_df = CSV.read("nba_elo.csv", DataFrame)
    elo_df = elo_df[elo_df.season .== parse(Int, year), :]
    elo_df.home_team = [get(TEAM_NAME_MAP, team, team) for team in elo_df.team1]
    elo_df.away_team = [get(TEAM_NAME_MAP, team, team) for team in elo_df.team2]
    
    # Merge ELO data (simplified date matching)
    merged_df = leftjoin(full_game_df, elo_df[:, [:home_team, :away_team, :date, :elo_prob1]], 
                        on=[:home_team, :away_team])
    
    # Fill missing ELO probabilities
    merged_df.elo_prob1 = coalesce.(merged_df.elo_prob1, 0.5)
    
    # Create time bins and score difference using the same binning logic
    quantiles = quantile(merged_df.normalized_time, range(0, 1, length=num_bins+1))
    merged_df.time_bin = [findfirst(x -> val <= x, quantiles[2:end]) for val in merged_df.normalized_time]
    merged_df.time_bin = [isnothing(bin) ? num_bins : bin for bin in merged_df.time_bin]
    
    merged_df.score_diff = merged_df.home .- merged_df.away
    
    return merged_df
end

# Bootstrap functions
function _bootstrap_iteration(original_df::DataFrame, game_numbers::Vector, num_bins::Int, seed::Int)
    """Single bootstrap iteration - worker function for parallel execution"""
    # Set local random seed for reproducibility
    Random.seed!(seed)
    
    # Bootstrap sample
    boot_game_nums = sample(game_numbers, length(game_numbers), replace=true)
    boot_df = vcat([original_df[original_df.game_num .== g, :] for g in boot_game_nums]...)
    
    # Generate binned data and calculate VUS
    binned_boot_df, bin_width_map = generate_binned_df(boot_df, num_bins=num_bins)
    boot_vus, auc_list = get_vus(binned_boot_df, bin_width_map, num_bins=num_bins)
    
    return boot_vus, auc_list
end

function bootstrap_vus(original_df::DataFrame; n_bootstraps::Int=1000, alpha::Float64=0.05, num_bins::Int=100, parallel::Bool=true)
    game_numbers = unique(original_df.game_num)
    
    if parallel
        println("Starting parallel bootstrapping with $n_bootstraps samples using $(Threads.nthreads()) threads...")
        
        # Generate seeds for reproducibility
        Random.seed!(42)
        seeds = rand(1:10000, n_bootstraps)
        
        # Parallel execution
        results = Vector{Tuple{Float64, Vector{Float64}}}(undef, n_bootstraps)
        
        Threads.@threads for i in 1:n_bootstraps
            results[i] = _bootstrap_iteration(original_df, game_numbers, num_bins, seeds[i])
        end
        
        # Extract VUS values and AUC lists
        vus_distribution = [r[1] for r in results]
        auc_estimates_matrix = hcat([r[2] for r in results]...)
        
    else
        # Sequential execution with progress bar
        Random.seed!(42)
        vus_distribution = Float64[]
        auc_estimates_per_bin = [Float64[] for _ in 1:num_bins]
        
        println("Starting sequential bootstrapping with $n_bootstraps samples...")
        
        @showprogress for i in 1:n_bootstraps
            boot_vus, auc_list = _bootstrap_iteration(original_df, game_numbers, num_bins, 42 + i)
            push!(vus_distribution, boot_vus)
            
            # Store AUC values for each bin
            for (bin_idx, auc_val) in enumerate(auc_list)
                push!(auc_estimates_per_bin[bin_idx], auc_val)
            end
        end
        
        auc_estimates_matrix = hcat(auc_estimates_per_bin...)
    end
    
    # Calculate confidence intervals
    vus_clean = filter(!isnan, vus_distribution)
    lower_bound = quantile(vus_clean, alpha/2)
    upper_bound = quantile(vus_clean, 1-alpha/2)
    
    println("Bootstrap analysis complete.")
    return vus_distribution, lower_bound, upper_bound, auc_estimates_matrix
end

function plot_auc_surface_with_confidence(auc_estimates_matrix::Matrix; alpha::Float64=0.05)
    # Handle NaN values by computing statistics only on valid data
    mean_aucs = vec([mean(filter(!isnan, auc_estimates_matrix[i, :])) for i in 1:size(auc_estimates_matrix, 1)])
    lower_aucs = vec([quantile(filter(!isnan, auc_estimates_matrix[i, :]), alpha/2) for i in 1:size(auc_estimates_matrix, 1)])
    upper_aucs = vec([quantile(filter(!isnan, auc_estimates_matrix[i, :]), 1-alpha/2) for i in 1:size(auc_estimates_matrix, 1)])
    
    time_bins = 1:length(mean_aucs)
    
    p = plot(time_bins, mean_aucs, label="Mean AUC", lw=2, color=:navy)
    plot!(p, time_bins, [lower_aucs upper_aucs], fillrange=[upper_aucs lower_aucs], 
          fillalpha=0.3, color=:skyblue, label="$(round((1-alpha)*100))% CI")
    
    xlabel!(p, "Time Bin")
    ylabel!(p, "Area Under Curve (AUC)")
    title!(p, "AUC by Time Bin with Bootstrapped Confidence Intervals")
    ylims!(p, (0.4, 1.0))
    
    display(p)
end

end # module