import concurrent.futures as cf
import multiprocessing

import math
import polars as pl
import tqdm
import utils
from dynamic_routing_analysis.decoding_utils import decoder_helper
import numpy as np


def decode_context_with_linear_shift(
    session_id: str,
    params,
):
    future_to_structure = {}
    structure_to_results = {}
    parallel = True
    # TODO add option to work on area + probe 
    # TODO add SC groupings
    if parallel:
        with cf.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn')) as executor:
            for structure in (
                utils.get_df('units', lazy=True)
                .filter(pl.col('session_id') == session_id)
                .select('structure')
                .collect()
                ['structure']
                .unique()
            ):
                executor.submit(
                    wrap_decoder_helper,
                    session_id=session_id,
                    params=params,
                    structure=structure,
                )
                if params.test:
                    break
            for future in tqdm.tqdm(cf.as_completed(future_to_structure), total=len(future_to_structure), unit='structure', desc=f'decoding {session_id}'):
                structure = future_to_structure[future]
                structure_to_results[structure] = future.result()
    else:
        
        for structure in (
            utils.get_df('units', lazy=True)
            .filter(pl.col('session_id') == session_id)
            .select('structure')
            .collect()
            ['structure']
            .unique()
        ):
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
    # TODO add option to do no shift/selection of middle blocks
    # get df len == n_units x n_trials, with spike counts in a column
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
    )
    spike_counts_array = (
        spike_counts_df
        .sort('trial_index', 'unit_id')
        .select('n_spikes')
        .to_numpy()
        .squeeze()
        .reshape(spike_counts_df.n_unique('trial_index'), spike_counts_df.n_unique('unit_id'))
    ) # n_trials x n_units
    unit_ids = spike_counts_df['unit_id'].sort()
    trials: pl.DataFrame = (
        utils.get_df('trials', lazy=True)
        .filter(pl.col('session_id') == session_id)
        .sort('trial_index')
        .select('context_name', 'trial_index', 'block_index')
        .collect()
    )
    if trials.n_unique('block_index') != 6:
        raise ValueError(f'Expecting 6 blocks for {session_id}: got {trials.n_unique("block_index")}')
    # TODO templeton context labels
    
    #find middle 4 block labels
    neg = math.ceil(len(trials.filter(pl.col('block_index')==0))/2)
    pos = math.floor(len(trials.filter(pl.col('block_index')==5))/2)
    labels = trials.sort('trial_index')['context_name'].to_numpy().squeeze()[neg: -pos]
    shifts= tuple(range(-neg, pos+1))
    
    repeat_idx_to_results = {}
    for repeat_idx in tqdm.tqdm(range(params.n_repeats), total=params.n_repeats, unit='repeat', desc=f'repeating {structure}|{session_id}'):
        shift_to_results = {}
        sel_units=np.random.choice(np.arange(unit_ids), params.n_units, replace=False)
        for shift in shifts:
            shift_to_results[shift] = decoder_helper(
                spike_counts_array[neg+shift: -pos+shift, sel_units],
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
    return repeat_idx_to_results