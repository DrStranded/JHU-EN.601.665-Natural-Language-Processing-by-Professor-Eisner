#!/usr/bin/env python3

import argparse
import logging
import math
from pathlib import Path
import torch
import numpy as np

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model1",
        type=Path,
        help="path to the first language model",
    )
    parser.add_argument(
        "model2",
        type=Path,
        help="path to the second language model",
    )
    parser.add_argument(
        "prior_prob1",
        type=float,
        help="prior probability of the first language model",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu', 'cuda', 'mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()

def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0

    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        # If the factor p(z | xy) = 0, then it will drive our cumulative file
        # probability to 0 and our cumulative log_prob to -infinity.  In
        # this case we can stop early, since the file probability will stay
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)

    return log_prob


def classify_text(file: Path, lm1: LanguageModel, lm2: LanguageModel, prior_prob1: float) -> float:
    ll1 = file_log_prob(file, lm1)
    ll2 = file_log_prob(file, lm2)
    log_p1 = np.log(prior_prob1)
    log_p2 = np.log(1 - prior_prob1)
    log_post1 = (log_p1 + ll1) - np.logaddexp(log_p1 + ll1, log_p2 + ll2)
    return np.exp(log_post1)


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device
    # (e.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration).
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                                 "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                                 "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)

    log.info("Testing...")
    lm1 = LanguageModel.load(args.model1, device=args.device)
    lm2 = LanguageModel.load(args.model2, device=args.device)

    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.

    log.info("Per-file log-probabilities:")
    total_log_prob = 0.0
    lm1_cnt = 0
    tot_cnt = 0
    for file in args.test_files:
        prob_lm1 = classify_text(file, lm1, lm2, args.prior_prob1)
        if prob_lm1 >= 0.5:
            which_lm = args.model1
            lm1_cnt += 1
        else:
            which_lm = args.model2
        tot_cnt += 1
        print(f"{which_lm}\t{file}")

    print(f"{lm1_cnt} files were more probably from {args.model1} ({(lm1_cnt / tot_cnt * 100):.2f}%)")
    print(f"{tot_cnt - lm1_cnt} files were more probably from {args.model2} ({(1 - lm1_cnt / tot_cnt) * 100:.2f}%)")


if __name__ == "__main__":
    main()

