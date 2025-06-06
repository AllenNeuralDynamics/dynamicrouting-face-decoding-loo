from __future__ import annotations

import os

# os.environ["RUST_BACKTRACE"] = "1"
# os.environ['POLARS_MAX_THREADS'] = '1'
# os.environ["TOKIO_WORKER_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["RAYON_NUM_THREADS"] = "1"


# stdlib imports --------------------------------------------------- #
import json
import logging
import pathlib
import time

import decoding_utils

# 3rd-party imports necessary for processing ----------------------- #
import matplotlib
import upath

# local modules ---------------------------------------------------- #
import utils
from decoding_utils import Params

# logging configuration -------------------------------------------- #
# use `logger.info(msg)` instead of `print(msg)` so we get timestamps and origin of log messages
logger = logging.getLogger(
    pathlib.Path(__file__).stem
    if __name__.endswith("_main__")
    else __name__
    # multiprocessing gives name '__mp_main__'
)

# general configuration -------------------------------------------- #
matplotlib.rcParams["pdf.fonttype"] = 42
logging.getLogger("matplotlib.font_manager").setLevel(
    logging.ERROR
)  # suppress matplotlib font warnings on linux


# processing function ---------------------------------------------- #


def main():
    t0 = time.time()

    utils.setup_logging()
    params = Params()  # reads from CLI args
    logger.setLevel(params.logging_level)

    if params.override_params_json:
        logger.info(f"Overriding parameters with {params.override_params_json}")
        params = Params(**json.loads(params.override_params_json))

    if params.test:
        params = Params(
            result_prefix=f"test/{params.result_prefix}",
            input_data=['facial_features'],
        )
        logger.info("Test mode: using modified set of parameters")

    upath.UPath("/results/params.json").write_text(params.model_dump_json(indent=4))
    if params.json_path.exists():
        existing_params = json.loads(params.json_path.read_text())
        if existing_params != params.model_dump():
            raise ValueError(
                f"Params file already exists and does not match current params:\n{existing_params=}\n{params.model_dump()=}"
            )
    else:
        logger.info(f"Writing params file: {params.json_path}")
        params.json_path.write_text(params.model_dump_json(indent=4))

    logger.info(f"starting decode_context_with_linear_shift with {params!r}")
    decoding_utils.decode_context(params=params)

    utils.ensure_nonempty_results_dir()
    logger.info(f"Time elapsed: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
