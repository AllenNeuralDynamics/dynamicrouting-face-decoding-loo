import concurrent.futures as cf
import logging
import math
import multiprocessing

import numpy as np
import polars as pl
import tqdm
import utils
from dynamic_routing_analysis.decoding_utils import decoder_helper

logger = logging.getLogger(__name__)

def decode_context_with_linear_shift(
    session_id: str,
    params,
):
    structure_to_results = {}
    parallel = True
    # TODO add option to work on area + probe 
    # TODO add SC groupings
    units = (
        utils.get_df('units', lazy=True)
        .filter(
            pl.col('session_id') == session_id,
            # only use areas with at least n_units (cannot random sample without replacement 
            # if we have less than n_units):
            pl.col('unit_id').n_unique().ge(params.n_units).over('session_id', 'structure'),    
        )
    )
    structures = units.select('structure').collect()['structure'].unique().sort()
    if parallel:
        with cf.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn')) as executor:
            future_to_structure = {}
            for structure in structures:
                future = executor.submit(
                    wrap_decoder_helper,
                    session_id=session_id,
                    params=params,
                    structure=structure,
                )
                future_to_structure[future] = structure
                logger.info(f"Submitted decoding to process pool for session {session_id}, structure {structure}")
                if params.test:
                    break
            for future in tqdm.tqdm(cf.as_completed(future_to_structure), total=len(future_to_structure), unit='structure', desc=f'decoding {session_id}'):
                structure = future_to_structure[future]
                structure_to_results[structure] = future.result()
    else:
        
        for structure in structures:
            result = wrap_decoder_helper(
                session_id=session_id,
                params=params,
                structure=structure,
            )
            structure_to_results[structure] = result
            if params.test:
                break
    return structure_to_results

def wrap_decoder_helper(
    session_id: str,
    params,
    structure: str,
):
    logger.debug(f"Getting units and trials for {session_id} {structure}")
    
    spike_counts_df = utils.get_per_trial_spike_times(
        starts=pl.col('stim_start_time') - 0.2,
        ends=pl.col('stim_start_time'),
        col_names='n_spikes',
        unit_ids=(
            utils.get_df('units', lazy=True)
            .filter(pl.col('session_id') == session_id)
            .filter(pl.col('structure') == structure)
            .select('unit_id')
            .collect()
            ['unit_id']
            .unique()
        ),
    ) # len == n_units x n_trials, with spike counts in a column
    logger.debug(f"Got spike counts: {spike_counts_df.shape} rows")
    
    spike_counts_array = (
        spike_counts_df
        .sort('trial_index', 'unit_id')
        .select('n_spikes')
        .to_numpy()
        .squeeze()
        .reshape(spike_counts_df.n_unique('trial_index'), spike_counts_df.n_unique('unit_id'))
    )
    logger.debug(f"Reshaped spike counts array: {spike_counts_array.shape}")
    
    unit_ids = spike_counts_df['unit_id'].unique().sort()
    trials = (
        utils.get_df('trials', lazy=True)
        .filter(pl.col('session_id') == session_id)
        .sort('trial_index')
        .select('context_name', 'trial_index', 'block_index')
        .collect()
    )
    logger.debug(f"Got {len(trials)} trials")
    
    if trials.n_unique('block_index') != 6:
        raise ValueError(f'Expecting 6 blocks for {session_id}: got {trials.n_unique("block_index")}')

    neg = math.ceil(len(trials.filter(pl.col('block_index')==0))/2)
    pos = math.floor(len(trials.filter(pl.col('block_index')==5))/2)
    labels = trials.sort('trial_index')['context_name'].to_numpy().squeeze()[neg: -pos]
    shifts = tuple(range(-neg, pos+1))
    logger.debug(f"Using shifts from {shifts[0]} to {shifts[-1]}")
    
    repeat_idx_to_results = {}
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
        shift_to_results = {}
        
        # ensure we don't sample the same set of units twice when we have few units:
        while True:
            sel_units = set(np.random.choice(np.arange(0, len(unit_ids)), params.n_units, replace=False))
            if sel_units not in unit_samples:
                unit_samples.append(sel_units)
                break
            
        logger.debug(f"Repeat {repeat_idx}: selected {len(sel_units)} units")
        
        for shift in shifts:
            first_trial_index = neg + shift
            assert first_trial_index >= 0
            last_trial_index = -pos + shift
            assert last_trial_index <= spike_counts_array.shape[0]
            data = spike_counts_array[first_trial_index: last_trial_index, sorted(sel_units)]
            assert data.shape == (len(labels), len(sel_units))
            logger.debug(f"Shift {shift}: using data shape {data.shape} with {len(labels)} labels")
            
            shift_to_results[shift] = decoder_helper(
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
            )
            if params.test:
                break
        repeat_idx_to_results[repeat_idx] = shift_to_results
        if params.test:
            break
    
    logger.info(f"Completed decoding for session {session_id}, structure {structure}")
    return repeat_idx_to_results