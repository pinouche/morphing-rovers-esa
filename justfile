# Run `just` to list the recipes. Variables can be overridden before the recipe name:
#   just output_dir=runs/test solution=runs/test/solution.json solve -s imitation.n_runs=1

set positional-arguments

config := "configs/default.yml"
output_dir := "runs/default"
solution := "solution.json"

default:
    @just --list

# Install the dependencies
sync:
    uv sync

# Run the whole optimisation and save the final solution; extra arguments go to the CLI (-s key=value)
solve *args:
    uv run morphing-rovers run -c {{config}} -s output_dir={{output_dir}} "$@"
    cp {{output_dir}}/submission.json {{solution}}

# Save the best chromosome so far as the solution, e.g. from an interrupted run
export:
    uv run morphing-rovers submit {{output_dir}}/best_chromosome.npy -o {{solution}} --name morphing-rovers
    uv run morphing-rovers evaluate {{output_dir}}/best_chromosome.npy
