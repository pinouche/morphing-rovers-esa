# Imitation learning

For each scenario, run a simulation, then for each data point/date (i.e., rover's position) in the simulation, identify the closest point from a chosen computed
arc. We use this point to build the "imitation" training dataset (current state, angle to get to nearest arc point).

The goal is that for a given scenario, we should identify which arc is best (yields best fitness) and keep this arc in the
to compute the final dataset across all 30 scenarios.