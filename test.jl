using Pkg
Pkg.activate(".")  
Pkg.instantiate()
# Example usage of VUS.jl package
using VUS
using CSV
using DataFrames
using Revise 
# Load and process data (equivalent to main.py)
raw_data = CSV.read("nba_2018.csv", DataFrame);
df = clean_nba_data(raw_data);

# Generate binned data and calculate VUS
binned_df, bin_width_map = generate_binned_df(df)
vus_value, auc_list = get_vus(binned_df, bin_width_map)

println("VUS: ", vus_value)

# Plot VUS
plot_vus(binned_df, bin_width_map)

# Run bootstrap analysis (parallel by default)
vus_dist, lower_ci, upper_ci, auc_matrix = bootstrap_vus(df, n_bootstraps=100)
println("VUS 95% CI: [$lower_ci, $upper_ci]")

# Plot AUC confidence bands
plot_auc_surface_with_confidence(auc_matrix)

# For sequential execution (if preferred)
# vus_dist, lower_ci, upper_ci, auc_matrix = bootstrap_vus(df, n_bootstraps=100, parallel=false)

# Benchmark example
year = "2018"
merged_df = prepare_benchmark_data(year)
model_features = ["score_diff", "elo_prob1"]  # Example features
model_df = fit_dynamic_benchmark(merged_df, model_features)

