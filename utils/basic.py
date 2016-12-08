import logging as log
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action="count",
                        help="increase output verbosity (e.g., -vv is more than -v)")

args = parser.parse_args()

if args.verbose:
    log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    log.info("Verbose output.")
else:
    log.basicConfig(format="%(levelname)s: %(message)s")
