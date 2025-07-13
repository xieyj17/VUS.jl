using Pkg
Pkg.activate(".")  
Pkg.instantiate()

using VUS, Test, DataFrames, CSV, Revise

include("utils.jl")
using .TestUtils

raw_data = CSV.read("./test/nba_2018.csv", DataFrame)

df = clean_nba_data(raw_data)
println("Data preparation complete. The DataFrame has ", nrow(df), " rows.")

# --- Step 3: Calculate the VUS Score ---
println("\nGenerating binned data (100 bins)...")
binned_df, bin_width_map = VUS.generate_binned_df(df, num_bins=100)
println("Binned data created.")

println("\nCalculating Volume Under the Surface (VUS)...")
vus, auc_list = VUS.get_vus(binned_df, bin_width_map)

# --- Step 4: Display the Final Result ---
println("\n------------------------------------")
println("  VUS Score: ", round(vus, digits=4))
println("------------------------------------")

# --- Step 5: Generate the 3D Plot ---
println("\nGenerating 3D VUS plot...")
VUS.plot_vus(binned_df, num_bins=100)
println("Plot generation complete.")


vus_dist, lower, upper = VUS.bootstrap_vus(df, n_bootstraps=1000)

println("\n--- Bootstrap Results ---")
println("95% Confidence Interval: ($(round(lower, digits=4)), $(round(upper, digits=4)))")
