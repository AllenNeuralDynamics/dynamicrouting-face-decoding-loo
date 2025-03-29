import os
import random
os.environ['RUST_BACKTRACE'] = '1'
#os.environ['POLARS_MAX_THREADS'] = '1'
os.environ['TOKIO_WORKER_THREADS'] = '1' 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['RAYON_NUM_THREADS'] = '1'

import concurrent.futures as cf
import logging
import math
import multiprocessing
import uuid
from typing import Iterable, Sequence

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

def repeat_multi_probe_areas(frame: polars._typing.FrameType) -> polars._typing.FrameType:
    """"If an area is recorded on multiple probes, transform the dataframe so it has rows for each
    probe and a row for both probes combined ('electrode_group_names': List[String])"""
    return (
        frame
        # create list of probe names per session-structure:
        .group_by('session_id', 'structure')
        .agg(
            pl.all().exclude('electrode_group_name'),
            pl.col('electrode_group_name').unique().alias('electrode_group_names'),
        )
        # duplicate those with multiple probes (col becomes List[List[String]]):
        .with_columns(
            pl.when(pl.col('electrode_group_names').list.n_unique().gt(1))
            .then(pl.col('electrode_group_names').repeat_by(2))
            .otherwise(pl.col('electrode_group_names').repeat_by(1))
        )
        # # .explode(pl.all().exclude('session_id', 'structure', 'electrode_group_names'))
        .explode('electrode_group_names')

        # convert the duplicated rows to joined 'probeA_probeB' format:
        .with_columns(
            pl.when(pl.col('electrode_group_names').is_first_distinct().over('session_id', 'structure'))
            .then(pl.col('electrode_group_names'))
            .otherwise(pl.col('electrode_group_names').list.join('_').cast(pl.List(pl.String)))
        )
        # create individual lists vs single list for duplicate/non-duplicate rows:
        # [probeA, probeB] vs [[probeA], [probeB]]
        .with_columns(
            pl.col('electrode_group_names').list.eval(pl.element().str.split('_'))
        )
         .explode(pl.all().exclude('session_id', 'structure', 'electrode_group_names'))
        .explode('electrode_group_names')
    )

def decode_context_with_linear_shift(
    session_ids: str | Iterable[str],
    params,
) -> None:
    if isinstance(session_ids, str):
        session_ids = [session_ids]

    combinations_df = (
        utils.get_df('units', lazy=True)
        .filter(
            pl.col('session_id').is_in(session_ids),
            params.units_query,
        )
        .pipe(group_structures, keep_originals=True)
        .pipe(repeat_multi_probe_areas)
        .filter(
            pl.col('unit_id').n_unique().ge(params.n_units).over(params.units_group_by)
        )
        .select(params.units_group_by)
        .unique(params.units_group_by)
        .collect()
    )
    if params.skip_existing and (params.data_path / '_delta_log').exists():
        existing = pl.scan_delta(params.data_path.as_posix()).select(params.units_group_by).unique(params.units_group_by).collect().to_dicts()
    else:
        existing = []

    logger.info(f"Processing {len(combinations_df)} unique session/area/probe combinations")
    if params.use_process_pool:
        session_results: dict[str, list[cf.Future]] = {}
        future_to_session = {}
        with cf.ProcessPoolExecutor(max_workers=params.max_workers, mp_context=multiprocessing.get_context('forkserver')) as executor:
            for row in combinations_df.iter_rows(named=True):
                if params.skip_existing and row in existing:
                    logger.info(f"Skipping {row} - results already exist")
                    continue
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
        for row in tqdm.tqdm(combinations_df.iter_rows(named=True), total=len(combinations_df), unit='row', desc=f'decoding'):
            if params.skip_existing and row in existing:
                logger.info(f"Skipping {row} - results already exist")
                continue
            try:
                wrap_decoder_helper(
                    params=params,
                    **row,
                )
            except NotEnoughBlocksError:
                logger.warning(f'{row["session_id"]} | NotEnoughBlocksError')
            except Exception:
                logger.exception(f'{row["session_id"]} | Failed:')
            if params.test:
                logger.info("Test mode: exiting after first session")
                break

def wrap_decoder_helper(
    params,
    session_id: str,
    structure: str,
    electrode_group_names: Sequence[str] | None = None,
):
    logger.debug(f"Getting units and trials for {session_id} {structure}")
    spike_counts_df = utils.get_per_trial_spike_times(
        intervals={
            'n_spikes_baseline': (pl.col('stim_start_time') - 0.2, pl.col('stim_start_time')),
        },
        as_counts=True,
        unit_ids=(
            utils.get_df('units', lazy=True)
            .filter(
                params.units_query,
                pl.col('session_id') == session_id,
                pl.col('structure') == structure,
                pl.lit(True) if electrode_group_names is None else pl.col('electrode_group_name').is_in(electrode_group_names),
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
    if (
        trials.n_unique('block_index') == 1
        and utils.get_df('session').filter(pl.col('session_id') == trials['session_id'][0])['keywords'].list.contains('templeton')
    ):
        logger.info(f'Adding dummy block labels for Templeton session {session_id}')
        trials = (
            trials
            .with_columns(
                pl.col('start_time').sub(pl.col('start_time').min().over('session_id')).truediv(10*60).floor().alias('ten_min_block_index')
            )
            .with_columns(
                pl.when(pl.col('ten_min_block_index').mod(2).eq(random.choice([0, 1])))
                .then(pl.lit('vis'))
                .otherwise(pl.lit('aud'))
                .alias('context_name')
            )
            .drop('ten_min_block_index')
        )
    elif trials.n_unique('block_index') != 6:
        raise NotEnoughBlocksError(f'Expecting 6 blocks: {session_id} has {trials.n_unique("block_index")} blocks')
    logger.debug(f"Got {len(trials)} trials")

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
                n_jobs=None,
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
    (
        pl.DataFrame(results)
        .with_columns(
            pl.lit(session_id).alias('session_id'),
            pl.lit(structure).alias('structure'),
            pl.lit(list(electrode_group_names)).alias('electrode_group_names'),
            pl.lit(params.n_units).alias('n_units'),
        )
        #.write_parquet((params.data_path / f"{uuid.uuid4()}.parquet").as_posix())
        .write_delta(params.data_path.as_posix(), mode='append')

    )
    logger.info(f"Completed decoding for session {session_id}, structure {structure}")
    # return results