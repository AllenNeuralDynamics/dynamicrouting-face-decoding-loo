import concurrent.futures as cf
import logging
import math
import multiprocessing
from typing import Iterable

import numpy as np
import polars as pl
import polars._typing
import tqdm
import upath
from dynamic_routing_analysis.decoding_utils import decoder_helper, NotEnoughBlocksError

import utils

logger = logging.getLogger(__name__)


    
def group_structures(frame: polars._typing.FrameType, keep_originals=False) -> polars._typing.FrameType:
    grouping = {
        'SCop': 'SCs',
        'SCsg': 'SCs',
        'SCzo': 'SCs',
        'SCig': 'SCm',
        'SCiw': 'SCm',
        'SCdg': 'SCm',
        'SCdw': 'SCm',
    }
    n_repeats = 2 if keep_originals else 1
    frame = (
        frame
        .with_columns(
            pl.when(pl.col('structure').is_in(grouping))
            .then(pl.col('structure').repeat_by(n_repeats))
            .otherwise(pl.col('structure').repeat_by(1))
        )
        .explode('structure')
        .with_columns(
            pl.when(pl.col('structure').is_in(grouping).is_first_distinct().over('unit_id'))
            .then(pl.col('structure').replace(grouping))
            .otherwise(pl.col('structure'))
        )
    )
    return frame 

def get_filtered_units(params) -> pl.LazyFrame:
    frame = (
        utils.get_df('units', lazy=True)
        .filter(
            params.units_query,
        )
        .pipe(group_structures, keep_originals=True)
        .filter(
            pl.col('unit_id').n_unique().ge(params.n_units).over(params.units_group_by)
        )
    )
    return frame

def decode_context_with_linear_shift(
    session_ids: str | Iterable[str],
    params,
) -> None:
    if isinstance(session_ids, str):
        session_ids = [session_ids]

    areas = (
        get_filtered_units(params)
        .filter(
            pl.col('session_id').is_in(session_ids),
        )
        .select(params.units_group_by)
        .unique(params.units_group_by)
        .collect()
    )
    logger.info(f"Processing {len(areas)} unique session/area/probe combinations")
    if params.use_process_pool:
        session_results: dict[str, list[cf.Future]] = {}
        future_to_session = {}
        with cf.ProcessPoolExecutor(max_workers=params.max_workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            for row in areas.iter_rows(named=True):
                future = executor.submit(
                    wrap_decoder_helper,
                    params=params,
                    **row,
                )
                session_results.setdefault(row['session_id'], []).append(future)
                future_to_session[future] = row['session_id']
                logger.debug(f"Submitted decoding to process pool for session {row['session_id']}, structure {row['structure']}")
                if params.test:
                    logger.info("Test mode: exiting after first session")
                    break
            for future in tqdm.tqdm(cf.as_completed(future_to_session), total=len(future_to_session), unit='structure', desc=f'Decoding'):
                session_id = future_to_session[future]
                if all(future.done() for future in session_results[session_id]):
                    logger.debug(f"Decoding completed for session {session_id}")
                    for f in session_results[session_id]:
                        try:
                            _ = f.result()
                        except Exception:
                            logger.exception(f'{session_id} | Failed:')
                    logger.info(f'{session_id} | Completed')
    else: # single-process mode
        for row in tqdm.tqdm(areas.iter_rows(named=True), total=len(areas), unit='row', desc=f'decoding {session_id}'):
            try:
                wrap_decoder_helper(
                    params=params,
                    **row,
                )
            except Exception:
                logger.exception(f'{row["session_id"]} | Failed:')
            if params.test:
                logger.info("Test mode: exiting after first session")
                break

def wrap_decoder_helper(
    params,
    session_id: str,
    structure: str,
    electrode_group_name: str | None = None,
):
    if params.skip_existing and params.data_path.exists():
        delta_df = (
            pl.scan_delta(params.data_path.as_posix())
            .filter(
                pl.col('session_id') == session_id,
                pl.col('structure') == structure,
            )
        )
        if electrode_group_name is not None:
            delta_df = delta_df.filter(pl.col('electrode_group_name') == electrode_group_name)
        session_results = delta_df.collect()
        if not session_results.is_empty():
            logger.info(f"Skipping {session_id} {structure} {electrode_group_name} - results already exist")
            return 
    
    logger.debug(f"Getting units and trials for {session_id} {structure}")
    spike_counts_df = utils.get_per_trial_spike_times(
        intervals={
            'n_spikes_baseline': (pl.col('stim_start_time') - 0.2, pl.col('stim_start_time')),
        },
        as_counts=True,
        unit_ids=(
            get_filtered_units(params)
            .filter(
                pl.col('session_id') == session_id,
                pl.col('structure') == structure,
                pl.lit(True) if electrode_group_name is None else pl.col('electrode_group_name').eq(electrode_group_name),
            )
            .select('unit_id')
            .collect()
            ['unit_id']
            .unique()
        ),
    ).sort('trial_index', 'unit_id') # len == n_units x n_trials, with spike counts in a column
    # sequence of unit_ids is used later: don't re-sort!
    
    logger.debug(f"Got spike counts: {spike_counts_df.shape} rows")
    
    spike_counts_array = (
        spike_counts_df
        .select('n_spikes_baseline')
        .to_numpy()
        .squeeze()
        .reshape(spike_counts_df.n_unique('trial_index'), spike_counts_df.n_unique('unit_id'))
    )
    logger.debug(f"Reshaped spike counts array: {spike_counts_array.shape}")
    
    unit_ids = spike_counts_df['unit_id'].unique()
    trials = (
        utils.get_df('trials', lazy=True)
        .filter(pl.col('session_id') == session_id)
        .sort('trial_index')
        .select('context_name', 'trial_index', 'block_index')
        .collect()
    )
    logger.debug(f"Got {len(trials)} trials")
    
    if trials.n_unique('block_index') != 6:
        raise NotEnoughBlocksError(f'Expecting 6 blocks: {session_id} has {trials.n_unique("block_index")} blocks')

    neg = math.ceil(len(trials.filter(pl.col('block_index')==0))/2)
    pos = math.floor(len(trials.filter(pl.col('block_index')==5))/2)
    labels = trials.sort('trial_index')['context_name'].to_numpy().squeeze()[neg: -pos]
    shifts = tuple(range(-neg, pos+1))
    logger.debug(f"Using shifts from {shifts[0]} to {shifts[-1]}")
    
    results = []
    # if we specify n_units == 20 and have 20 units, there are no repeats to do - 
    # get the min number of repeats possible to avoid unnecessary work:
    n_possible_samples = math.comb(len(unit_ids), params.n_units)
    if params.n_repeats > n_possible_samples:
        n_repeats = n_possible_samples
        logger.warning(f"Reducing number of repeats from {params.n_repeats} to {n_repeats} to avoid unnecessary work ({params.n_units=}, {len(unit_ids)=})")
    else:
        n_repeats = params.n_repeats
    unit_samples: list[set[int]] = []
    for repeat_idx in tqdm.tqdm(range(n_repeats), total=n_repeats, unit='repeat', desc=f'repeating {structure}|{session_id}'):
        
        # ensure we don't sample the same set of units twice when we have few units:
        while True:
            sel_units = set(np.random.choice(np.arange(0, len(unit_ids)), params.n_units, replace=False))
            if sel_units not in unit_samples:
                unit_samples.append(sel_units)
                break
            
        logger.debug(f"Repeat {repeat_idx}: selected {len(sel_units)} units")
        
        for shift in shifts:
            first_trial_index = neg + shift
            last_trial_index = len(trials) - pos + shift
            logger.debug(f"Shift {shift}: using trials {first_trial_index} to {last_trial_index} out of {len(trials)}")
            assert first_trial_index >= 0, f"{first_trial_index=}"
            assert last_trial_index > first_trial_index, f"{last_trial_index=}, {first_trial_index=}"
            assert last_trial_index <= spike_counts_array.shape[0], f"{last_trial_index=}, {spike_counts_array.shape[0]=}"
            data = spike_counts_array[first_trial_index: last_trial_index, sorted(sel_units)]
            assert data.shape == (len(labels), len(sel_units)), f"{data.shape=}, {len(labels)=}, {len(sel_units)=}"
            logger.debug(f"Shift {shift}: using data shape {data.shape} with {len(labels)} labels")
            
            _result = decoder_helper(
                data,
                labels,
                decoder_type=params.decoder_type,
                crossval=params.crossval,
                crossval_index=None,
                labels_as_index=params.labels_as_index,
                train_test_split_input=None,
                regularization=params.regularization,
                penalty=params.penalty,
                solver=params.solver,
                n_jobs=params.n_jobs
            )
            result = {}
            result['balanced_accuracy_test'] = _result['balanced_accuracy_test'].item()
            result['shift_idx'] = shift
            result['repeat_idx'] = repeat_idx
            result['predict_proba'] = _result['predict_proba'][:, np.where(_result['label_names'] == 'vis')[0][0]].tolist()
            result['unit_ids'] = unit_ids.to_numpy()[sorted(sel_units)].tolist()
            results.append(result)
            if params.test:
                break
        if params.test:
            break
    import pickle
    upath.UPath('/results/result.pkl').write_bytes(pickle.dumps(_result))
    (
        pl.DataFrame(results)
        .with_columns(
            pl.lit(session_id).alias('session_id'),
            pl.lit(structure).alias('structure'),
            pl.lit(electrode_group_name).alias('electrode_group_name'),
            pl.lit(params.n_units).alias('n_units'),
        )
        .write_delta(params.data_path.as_posix(), mode='append')
    )
    logger.info(f"Completed decoding for session {session_id}, structure {structure}")
    # return results