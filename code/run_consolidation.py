import tqdm
import polars as pl

results_path = 's3://aind-scratch-data/dynamic-routing/decoding/results/v265_0/'
print(f"Getting unique session IDs from parquet files in {results_path}")
lf = pl.scan_parquet(results_path)
session_ids = lf.select('session_id').collect()['session_id'].unique()

session_partition_path = 's3://aind-scratch-data/dynamic-routing/decoding/results/v265_0_part/'
print(f"Writing parquet files partitioned by session ID to {session_partition_path}")
for session_id in tqdm.tqdm(session_ids):
    (
        lf.filter(pl.col('session_id') == session_id)
        .collect()
        # sorting will impact data access speed - should be tailored to most common analysis pattern:
        .sort('is_all_trials', 'structure', 'unit_criteria', 'unit_subsample_size', 'repeat_idx')
        .write_parquet(
            f'{session_partition_path}{session_id}.parquet',
            compression_level=18,
        )
    )
    
single_path = 's3://aind-scratch-data/dynamic-routing/decoding/results/v265_0_consolidated.parquet'
print(f'Consolidating all session parquet files at {single_path}')
(
    pl.scan_parquet(session_partition_path)
    .sink_parquet(single_path, compression_level=18)
)