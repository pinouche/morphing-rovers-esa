"""Command line interface: `morphing-rovers {run,sweep,evaluate,submit}`."""

import argparse
import logging
import sys

import numpy as np

from morphing_rovers import pipeline
from morphing_rovers.chromosome import clip_to_bounds, load_chromosome
from morphing_rovers.config import load_config
from morphing_rovers.submission import write_submission
from morphing_rovers.udp import morphing_rover_UDP

logger = logging.getLogger("morphing_rovers")

OPTIMISATIONS = {
    "run": (pipeline.run, "end-to-end optimisation: masks by terrain clustering, controller by imitation learning"),
    "sweep": (pipeline.sweep, "imitation learning on single scenarios to pick their arc radius factor"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="morphing-rovers", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for name, (_, help) in OPTIMISATIONS.items():
        command = commands.add_parser(name, help=help)
        command.add_argument("-c", "--config", help="YAML config file (default: built-in defaults, as in "
                                                   "configs/default.yml)")
        command.add_argument("-s", "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                             help="override a config value, e.g. -s imitation.n_runs=1 (repeatable)")

    evaluate = commands.add_parser("evaluate", help="print the fitness of a chromosome on each scenario")
    evaluate.add_argument("chromosome", help=".npy file, or legacy pickle")
    evaluate.add_argument("--scenarios", type=int, nargs="+", help="scenario indices (0-29), default all")
    evaluate.add_argument("--plot", action="store_true", help="plot the masks, trajectories, modes and velocities")

    submit = commands.add_parser("submit", help="write a submission file for optimise.esa.int")
    submit.add_argument("chromosome", help=".npy file, or legacy pickle")
    submit.add_argument("-o", "--output", default="submission.json")
    submit.add_argument("--name", default="morphing-rovers")
    submit.add_argument("--description", default="")
    submit.add_argument("--clip", action="store_true", help="clip the chromosome to the problem bounds first")

    args = parser.parse_args(argv)
    # stdout, like the fitness tables printed by the challenge code, so that both stay in order.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)

    if args.command in OPTIMISATIONS:
        optimise, _ = OPTIMISATIONS[args.command]
        optimise(load_config(args.config, args.overrides))
    elif args.command == "evaluate":
        udp = morphing_rover_UDP()
        chromosome = load_chromosome(args.chromosome)
        udp.pretty(chromosome, args.scenarios)
        if args.plot:
            udp.plot(chromosome, args.scenarios, plot_modes=True, plot_mode_efficiency=True)
    elif args.command == "submit":
        chromosome = load_chromosome(args.chromosome)
        if args.clip:
            clipped = clip_to_bounds(chromosome)
            logger.info("Clipped %d values to the bounds", np.count_nonzero(clipped != chromosome))
            chromosome = clipped
        write_submission(chromosome, args.output, args.name, args.description)
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
