from __future__ import annotations

# stdlib imports --------------------------------------------------- #
import argparse
import dataclasses
import gc
import json
import functools
import logging
import pathlib
import time
import types
import typing
import uuid
from typing import Any, Literal, Union

# 3rd-party imports necessary for processing ----------------------- #
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pynwb
import upath
import zarr
from dynamic_routing_analysis import spike_utils, decoding_utils, data_utils, path_utils

import utils

# logging configuration -------------------------------------------- #
# use `logger.info(msg)` instead of `print(msg)` so we get timestamps and origin of log messages
logger = logging.getLogger(
    pathlib.Path(__file__).stem if __name__.endswith("_main__") else __name__
    # multiprocessing gives name '__mp_main__'
)

# general configuration -------------------------------------------- #
matplotlib.rcParams['pdf.fonttype'] = 42
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR) # suppress matplotlib font warnings on linux


# utility functions ------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_id', type=str, default=None)
    parser.add_argument('--logging_level', type=str, default='INFO')
    parser.add_argument('--skip_existing', type=int, default=1)
    parser.add_argument('--update_packages_from_source', type=int, default=1)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--session_table_query', type=str, default="is_ephys & is_task & is_annotated & is_production & issues=='[]'")
    parser.add_argument('--override_params_json', type=str, default="{}")
    for field in dataclasses.fields(Params):
        if field.name in [getattr(action, 'dest') for action in parser._actions]:
            # already added field above
            continue
        logger.debug(f"adding argparse argument {field}")
        kwargs = {}
        if isinstance(field.type, str):
            kwargs = {'type': eval(field.type)}
        else:
            kwargs = {'type': field.type}
        if kwargs['type'] in (list, tuple):
            logger.debug(f"Cannot correctly parse list-type arguments from App Builder: skipping {field.name}")
        if isinstance(field.type, str) and field.type.startswith('Literal'):
            kwargs['type'] = str
        if isinstance(kwargs['type'], (types.UnionType, typing._UnionGenericAlias)):
            kwargs['type'] = typing.get_args(kwargs['type'])[0]
            logger.info(f"setting argparse type for union type {field.name!r} ({field.type}) as first component {kwargs['type']!r}")
        parser.add_argument(f'--{field.name}', **kwargs)
    args = parser.parse_args()
    list_args = [k for k,v in vars(args).items() if type(v) in (list, tuple)]
    if list_args:
        raise NotImplementedError(f"Cannot correctly parse list-type arguments from App Builder: remove {list_args} parameter and provide values via `override_params_json` instead")
    logger.info(f"{args=}")
    return args


# processing function ---------------------------------------------- #
# modify the body of this function, but keep the same signature

def process_session(session_id: str, params: "Params", test: int = 0, skip_existing: bool = False) -> None:
    """Process a single session with parameters defined in `params` and save results + params to
    /results.
    
    A test mode should be implemented to allow for quick testing of the capsule (required every time
    a change is made if the capsule is in a pipeline) 
    """
    # Get nwb file
    # Currently this can fail for two reasons: 
    # - the file is missing from the datacube, or we have the path to the datacube wrong (raises a FileNotFoundError)
    # - the file is corrupted due to a bad write (raises a RecursionError)
    # Choose how to handle these as appropriate for your capsule

    try:
        session = utils.get_nwb(session_id, raise_on_missing=True, raise_on_bad_file=True) 
    except (FileNotFoundError, RecursionError) as exc:
        logger.info(f"Skipping {session_id}: {exc!r}")
        return
    
    if test:
        params.folder_name = f"test/{params.folder_name}"
        params.only_use_all_units = True
        params.n_units = ["all"]
        params.keep_n_SVDs = 5
        params.LP_parts_to_keep = ["ear_base_l"]
        params.n_repeats = 1
        params.n_unit_threshold = 5
        logger.info(f"Test mode: using modified set of parameters")

    if skip_existing and params.file_path.exists():
        logger.info(f"{params.file_path} exists: processing skipped")
        return
    
    # Get components from the nwb file:
    trials = session.trials[:]
    if 'recalc' in params.unit_criteria:
        recalc_path='/data/dynamicrouting_unit_metrics_recalculated/units_with_recalc_metrics.csv'
        recalculated_unit_metrics = pd.read_csv(recalc_path)
        sel_units = recalculated_unit_metrics.query(params.units_query)['unit_id'].tolist()

        units = session.units[:].query("unit_id in @sel_units")

    else:
        units = session.units[:].query(params.units_query)

    if params.select_single_area is not None:
        units = units.query('structure==@params.select_single_area')
        if len(units)==0:
            logger.error(f"Structure does not exist in session; skipping")
            return

    if test:
        logger.info(f"Test mode: using reduced set of units")
        units = units.sort_values('location').head(20)

    #units: pd.DataFrame =  utils.remove_pynwb_containers_from_table(units[:])
    units['session_id'] = session_id
    units.drop(columns=['waveform_sd','waveform_mean'], inplace=True, errors='ignore')

    cols_to_keep=['unit_id','session_id','structure','electrode_group_name','spike_times','ccf_ap', 'ccf_dv', 'ccf_ml']
    units=units[cols_to_keep]

    logger.info(f'starting decode_context_with_linear_shift for {session_id} with {params.to_json()}')

    decoding_utils.decode_context_with_linear_shift(session=session,params=params.to_dict(),trials=trials,units=units)

    del units
    del trials
    del session
    gc.collect()

    logger.info(f'making summary tables of decoding results for {session_id}')
    decoding_results = decoding_utils.concat_decoder_results(
        files=[params.file_path],
        savepath=params.savepath,
        return_table=True,
        single_session=True,
    )
    #find n_units to loop through for next step
    #print(decoding_results)
    if decoding_results is not None:
        n_units = []
        for col in decoding_results.filter(like='true_accuracy_').columns.values:
            if len(col.split('_'))==3:
                temp_n_units=col.split('_')[2]
                try:
                    n_units.append(int(temp_n_units))
                except:
                    n_units.append(temp_n_units)
            else:
                n_units.append(None)

        decoding_results = []
        for nu in n_units:
            decoding_utils.concat_trialwise_decoder_results(
                files=[params.file_path],
                savepath=params.savepath,
                return_table=False,
                n_units=nu,
                single_session=True,
            )

    logger.info(f'{session_id} | Writing params file')
    params.write_json(params.file_path.with_suffix('.json'))
    
    
# define run params here ------------------------------------------- #

# The `Params` class is used to store parameters for the run, for passing to the processing function.
# @property methods (like `savepath` below) are computed from other parameters on-demand as required:
# this way, we can separate the parameters dumped to json from larger arrays etc. required for
# processing. We can also update other fields during test mode, and the updated values will be incoporated 
# into these fields.

# - if needed, we can get parameters from the command line and pass them to the dataclass (see `main()` below):
#   just add a field to the App Builder parameters with the same `Parameter Name`

# this is an example from Sam's processing code, replace with your own parameters as needed:
@dataclasses.dataclass
class Params:
    # ----------------------------------------------------------------------------------
    # defaults don't matter for these parameters, they will be updated later:
    session_id: str = ""
    run_id: str = ""
    """A unique string that should be attached to all decoding runs in the same batch"""
    # ----------------------------------------------------------------------------------

    folder_name: str = "test"
    unit_criteria: str = 'medium'
    n_units: list = dataclasses.field(default_factory=lambda: [5, 10, 20, 30, 40, 50, 'all'])
    """number of units to sample for each area"""
    n_repeats: int = 25
    """number of times to repeat decoding with different randomly sampled units"""
    input_data_type: str | Literal['spikes', 'facemap', 'LP'] = 'spikes'
    vid_angle_facemotion: str | Literal['behavior', 'face', 'eye'] = 'face'
    vid_angle_LP: str | Literal['behavior', 'face', 'eye'] = 'behavior'
    central_section: str = '4_blocks_plus'
    """for linear shift decoding, how many trials to use for the shift. '4_blocks_plus' is best"""
    exclude_cue_trials: bool = False
    """option to totally exclude autorewarded trials"""
    n_unit_threshold: int = 5
    """minimum number of units to include an area in the analysis"""
    keep_n_SVDs: int = 500
    """number of SVD components to keep for facemap data"""
    LP_parts_to_keep: list = dataclasses.field(default_factory=lambda: ['ear_base_l', 'eye_bottom_l', 'jaw', 'nose_tip', 'whisker_pad_l_side'])
    spikes_binsize: float = 0.2
    spikes_time_before: float = 0.2
    spikes_time_after: float = 0.01
    use_structure_probe: bool = True
    """if True, append probe name to area name when multiple probes in the same area"""
    crossval: Literal['5_fold', 'blockwise'] = '5_fold'
    """blockwise untested with linear shift"""
    labels_as_index: bool = True
    """convert labels (context names) to index [0,1]"""
    decoder_type: str | Literal['linearSVC', 'LDA', 'RandomForest', 'LogisticRegression'] = 'LogisticRegression'
    only_use_all_units: bool = False
    """if True, do not run decoding with different areas, only with all areas -- for debugging"""
    predict: str = 'context'
    """ 'context' = predict context; 'vis_appropriate_response' = predict whether the mouse's response was appropriate for a visual context block """
    regularization: float | None = None
    """ set regularization (C) for the decoder. Setting to None reverts to the default value (usually 1.0) """
    penalty: str | None = None
    """ set penalty for the decoder. Setting to None reverts to default """
    solver: str | None = None
    """ set solver for the decoder. Setting to None reverts to default """
    select_single_area: str | None = None
    """ select a single area to run decoding analysis on. If None, run on all areas """
    split_area_by_probe: int = 1
    """ splits area units by probe if recorded by more than one probe"""

    @property
    def savepath(self) -> upath.UPath:
        return path_utils.DECODING_ROOT_PATH / f"{self.folder_name}_{self.run_id}" 

    @property
    def filename(self) -> str:
        return f"{self.session_id}_{self.run_id}.pkl"

    @property
    def file_path(self) -> upath.UPath:
        return self.savepath / self.filename
    
    @property
    def units_query(self) -> str:
        if self.unit_criteria == 'medium':
            return 'isi_violations_ratio<=0.5 and presence_ratio>=0.9 and amplitude_cutoff<=0.1'
        elif self.unit_criteria == 'strict':
            return 'isi_violations_ratio<=0.1 and presence_ratio>=0.99 and amplitude_cutoff<=0.1'
        elif self.unit_criteria == 'use_sliding_rp':
            return 'sliding_rp_violation<=0.1 and presence_ratio>=0.99 and amplitude_cutoff<=0.1'
        elif self.unit_criteria == 'recalc_presence_ratio':
            return 'sliding_rp_violation<=0.1 and presence_ratio_task>=0.99 and amplitude_cutoff<=0.1'
        elif self.unit_criteria == 'no_drift':
            return 'decoder_label!="noise" and isi_violations_ratio<=0.5 and presence_ratio>=0.7 and amplitude_cutoff<=0.1'
        elif self.unit_criteria == 'loose_drift':
            return 'activity_drift<=0.2 and decoder_label!="noise" and isi_violations_ratio<=0.5 and presence_ratio>=0.7 and amplitude_cutoff<=0.1'
        elif self.unit_criteria == 'medium_drift':
            return 'activity_drift<=0.15 and decoder_label!="noise" and isi_violations_ratio<=0.5 and presence_ratio>=0.7 and amplitude_cutoff<=0.1'
        elif self.unit_criteria == 'strict_drift':
            return 'activity_drift<=0.1 and decoder_label!="noise" and isi_violations_ratio<=0.5 and presence_ratio>=0.7 and amplitude_cutoff<=0.1'
        else:
            raise ValueError(f"No units query available for {self.unit_criteria=!r}")

    def to_json(self, **dumps_kwargs) -> str:
        """json string of field name: value pairs, excluding values from property getters (which may be large)"""
        return json.dumps(dataclasses.asdict(self), **dumps_kwargs)

    def write_json(self, path: str | upath.UPath = '/results/params.json') -> None:
        path = upath.UPath(path)
        logger.info(f"Writing params to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=2))

    def to_dict(self) -> dict[str, Any]:
        """dict of field name: value pairs, including values from property getters"""
        return dataclasses.asdict(self) | {k: getattr(self, k) for k in dir(self.__class__) if isinstance(getattr(self.__class__, k), property)}

# ------------------------------------------------------------------ #


def main():
    t0 = time.time()
    
    utils.setup_logging()

    # get arguments passed from command line (or "AppBuilder" interface):
    args = parse_args()
    logger.setLevel(args.logging_level)

    # if any of the parameters required for processing are passed as command line arguments, we can
    # get a new params object with these values in place of the defaults:
    params = {}
    for field in dataclasses.fields(Params):
        if (val := getattr(args, field.name, None)) is not None:
            params[field.name] = val
    
    override_params = json.loads(args.override_params_json)
    if override_params:
        for k, v in override_params.items():
            if k in params:
                logger.info(f"Overriding value of {k!r} from command line arg with value specified in `override_params_json`")
            params[k] = v
    
    # if session_id is passed as a command line argument, we will only process that session,
    # otherwise we process all session IDs that match filtering criteria:    
    session_table = pd.read_parquet(utils.get_datacube_dir() / 'session_table.parquet')
    session_table['issues']=session_table['issues'].astype(str)
    session_ids: list[str] = session_table.query(args.session_table_query)['session_id'].values.tolist()
    logger.debug(f"Found {len(session_ids)} session_ids available for use after filtering")
    
    if args.session_id is not None:
        if args.session_id not in session_ids:
            logger.warning(f"{args.session_id!r} not in filtered session_ids: exiting")
            exit()
        logger.info(f"Using single session_id {args.session_id} provided via command line argument")
        session_ids = [args.session_id]
    elif utils.is_pipeline(): 
        # only one nwb will be available 
        session_ids = set(session_ids) & set(p.stem for p in utils.get_nwb_paths())
    else:
        logger.info(f"Using list of {len(session_ids)} session_ids after filtering")
    
    # run processing function for each session, with test mode implemented:
    for session_id in session_ids:
        try:
            process_session(session_id, params=Params(**params | {'session_id': session_id}), test=args.test, skip_existing=args.skip_existing)
        except Exception as e:
            logger.exception(f'{session_id} | Failed:')
        else:
            logger.info(f'{session_id} | Completed')

        if args.test:
            logger.info("Test mode: exiting after first session")
            break
    utils.ensure_nonempty_results_dir()
    logger.info(f"Time elapsed: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
