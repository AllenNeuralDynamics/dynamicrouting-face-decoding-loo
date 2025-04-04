import os
import random
os.environ['RUST_BACKTRACE'] = '1'
#os.environ['POLARS_MAX_THREADS'] = '1'
os.environ['TOKIO_WORKER_THREADS'] = '1' 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['RAYON_NUM_THREADS'] = '1'

import concurrent.futures as cf
import contextlib
import itertools
import logging
import math
import multiprocessing
import random
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


    
def group_structures(frame: polars._typing.FrameType, keep_originals=True) -> polars._typing.FrameType:
    grouping = {
        'SCop': 'SCs',
        'SCsg': 'SCs',
        'SCzo': 'SCs',
        'SCig': 'SCm',
        'SCiw': 'SCm',
        'SCdg': 'SCm',
        'SCdw': 'SCm',
        "ECT1": 'ECT',
        "ECT2/3": 'ECT',    
        "ECT6b": 'ECT',
        "ECT5": 'ECT',
        "ECT6a": 'ECT', 
        "ECT4": 'ECT',
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
    duplicates =  (
        frame
        .clone()
        .with_columns(
             pl.col('electrode_group_name').unique().over('session_id', 'structure', mapping_strategy='join').alias('electrode_group_names')
         )
        .filter(
             pl.col('electrode_group_names').list.len().ge(2)
         )
    )
    return (
        pl.concat(
            [
                frame.with_columns(pl.col('electrode_group_name').cast(pl.List(pl.String)).alias('electrode_group_names')),
                duplicates,
            ],
        )
        .drop('electrode_group_name')
    )

def decode_context_with_linear_shift(
    session_ids: str | Iterable[str],
    params,
) -> None:
    if isinstance(session_ids, str):
        session_ids = [session_ids]

    combinations_df = (
        utils.get_df('units', lazy=True)
        .drop_nulls('structure')
        .filter(
            pl.col('session_id').is_in(session_ids),
            params.units_query,
        )
        .pipe(group_structures, keep_originals=True)
        .pipe(repeat_multi_probe_areas)
        .filter(params.min_n_units_query)
        .select(params.units_group_by)
        .unique(params.units_group_by)
        .collect()
    )
    if params.skip_existing and params.data_path.exists():
        existing = (
            pl.scan_parquet(params.data_path.as_posix().removesuffix('/') + '/')
            .filter(
                pl.col('min_n_units').is_null() if params.min_n_units is None else pl.col('min_n_units').eq(params.min_n_units),
                pl.col('unit_criteria') == params.unit_criteria,
            )
            .select(params.units_group_by)
            .unique(params.units_group_by)
            .collect()
            .to_dicts()
        )
    else:
        existing = []
    def is_row_in_existing(row):
        """Regular dict comparison doesn't work with list field?"""
        return any(
            x for x in existing 
            if all(x[k] == row[k] for k in ['session_id', 'structure'])
            and set(x['electrode_group_names']) == set(row['electrode_group_names'])
        )
        
    logger.info(f"Processing {len(combinations_df)} unique session/area/probe combinations")
    if params.use_process_pool:
        session_results: dict[str, list[cf.Future]] = {}
        future_to_session = {}
        lock = None # multiprocessing.Manager().Lock() # or None
        with cf.ProcessPoolExecutor(max_workers=params.max_workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            for row in combinations_df.iter_rows(named=True):
                if params.skip_existing and is_row_in_existing(row):
                    logger.info(f"Skipping {row} - results already exist")
                    continue
                future = executor.submit(
                    wrap_decoder_helper,
                    params=params,
                    **row,
                    lock=lock,
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
            if params.skip_existing and is_row_in_existing(row):
                logger.info(f"Skipping {row} - results already exist")
                continue
            try:
                wrap_decoder_helper(
                    params=params,
                    **row,
                )
            except NotEnoughBlocksError as exc:
                logger.warning(f'{row["session_id"]} | {exc!r}')
            except Exception:
                logger.exception(f'{row["session_id"]} | Failed:')
            if params.test:
                logger.info("Test mode: exiting after first session")
                break

def wrap_decoder_helper(
    params,
    session_id: str,
    structure: str,
    electrode_group_names: Sequence[str],
    lock = None,
):
    logger.debug(f"Getting units and trials for {session_id} {structure}")
    spike_counts_df = (
        utils.get_per_trial_spike_times(
            intervals={
                'n_spikes_baseline': (pl.col('stim_start_time') - params.spikes_time_before, pl.col('stim_start_time')),
            },
            as_counts=True,
            unit_ids=(
                utils.get_df('units', lazy=True)
                .pipe(group_structures)
                .filter(
                    params.units_query,
                    pl.col('session_id') == session_id,
                    pl.col('structure') == structure,
                    pl.col('electrode_group_name').is_in(electrode_group_names),
                )
                .select('unit_id')
                .collect()
                ['unit_id']
                .unique()
            ),
        )
        .filter(
            pl.col('n_spikes_baseline').is_not_null(),
            # only keep observed trials
        )
        .sort('trial_index', 'unit_id') 
    )
    # len == n_units x n_trials, with spike counts in a column
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
        .filter(
            pl.col('session_id') == session_id,
            pl.col('trial_index').is_in(spike_counts_df['trial_index'].unique()),
            # obs_intervals may affect number of trials available
        )
        .sort('trial_index')
        .select('context_name', 'start_time', 'trial_index', 'block_index', 'session_id')
        .collect()
    )
    if (
        trials['block_index'].n_unique() == 1
        and not (
            utils.get_df('session')
            .filter(
                pl.col('session_id') == trials['session_id'][0],
                pl.col('keywords').list.contains('templeton'),
            )
        ).is_empty()
    ):
        logger.info(f'Adding dummy context labels for Templeton session {session_id}')
        trials = (
            trials
            .with_columns(
                pl.col('start_time').sub(pl.col('start_time').min().over('session_id')).truediv(10*60).floor().clip(0, 5).alias('block_index')
                # short 7th block will sometimes be present: merge into 6th with clip
            )
            .with_columns(
                pl.when(pl.col('block_index').mod(2).eq(random.choice([0, 1])))
                .then(pl.lit('vis'))
                .otherwise(pl.lit('aud'))
                .alias('context_name')
            )
        )
    if trials.n_unique('block_index') != 6:
        raise NotEnoughBlocksError(f'Expecting 6 blocks: {session_id} has {trials.n_unique("block_index")} blocks of observed ephys data')
    logger.debug(f"Got {len(trials)} trials")

    context_labels = trials.sort('trial_index')['context_name'].to_numpy().squeeze()

    max_neg_shift = math.ceil(len(trials.filter(pl.col('block_index')==0))/2)
    max_pos_shift = math.floor(len(trials.filter(pl.col('block_index')==5))/2)
    shifts = tuple(range(-max_neg_shift, max_pos_shift + 1))
    logger.debug(f"Using shifts from {shifts[0]} to {shifts[-1]}")
    
    n_units_to_use = params.min_n_units or len(unit_ids) # if min_n_units is None, use all available
    
    unit_idx = list(range(0, len(unit_ids)))

    results = []
    for repeat_idx in tqdm.tqdm(range(params.n_repeats), total=params.n_repeats, unit='repeat', desc=f'repeating {structure} | {session_id}'):
        
        sel_unit_idx = random.sample(unit_idx, n_units_to_use)
            
        logger.debug(f"Repeat {repeat_idx}: selected {len(sel_unit_idx)} units")
        
        for shift in (*shifts, None): # None will be a special case using all trials, with no shift
            
            is_all_trials = shift is None
            if not is_all_trials:
                labels = context_labels[max_neg_shift: -max_pos_shift]
                first_trial_index = max_neg_shift + shift
                last_trial_index = len(trials) - max_pos_shift + shift
                logger.debug(f"Shift {shift}: using trials {first_trial_index} to {last_trial_index} out of {len(trials)}")
                assert first_trial_index >= 0, f"{first_trial_index=}"
                assert last_trial_index > first_trial_index, f"{last_trial_index=}, {first_trial_index=}"
                assert last_trial_index <= spike_counts_array.shape[0], f"{last_trial_index=}, {spike_counts_array.shape[0]=}"
                data = spike_counts_array[first_trial_index: last_trial_index, sorted(sel_unit_idx)]
            else:
                labels = context_labels
                data = spike_counts_array[:, sorted(sel_unit_idx)]

            assert data.shape == (len(labels), len(sel_unit_idx)), f"{data.shape=}, {len(labels)=}, {len(sel_unit_idx)=}"
            logger.debug(f"Shift {shift}: using data shape {data.shape} with {len(labels)} context labels")
            
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
            if shift in (0, None):  # don't save probabilities from shifts which we won't use 
                result['predict_proba'] = _result['predict_proba'][:, np.where(_result['label_names'] == 'vis')[0][0]].tolist()
            else:
                result['predict_proba'] = None 
            result['unit_ids'] = unit_ids.to_numpy()[sorted(sel_unit_idx)].tolist()
            result['is_all_trials'] = is_all_trials
            results.append(result)
            if params.test:
                break
        if params.test:
            break
    with lock or contextlib.nullcontext():
        logger.info('Writing data')
        (
            pl.DataFrame(results)
            .with_columns(
                pl.lit(session_id).alias('session_id'),
                pl.lit(structure).alias('structure'),
                pl.lit(sorted(electrode_group_names)).alias('electrode_group_names'),
                pl.lit(params.min_n_units).alias('min_n_units').cast(pl.UInt8),
                pl.lit(params.unit_criteria).alias('unit_criteria'),
            )
            .cast(
                {
                    'shift_idx': pl.Int8,
                    'repeat_idx': pl.UInt8,
                }
            )
            .write_parquet(
                (params.data_path / f"{uuid.uuid4()}.parquet").as_posix(),
                compression_level=18,
                statistics='full',    
            )
            # .write_delta(params.data_path.as_posix(), mode='append')
        )
    logger.info(f"Completed decoding for session {session_id}, structure {structure}")
    # return results