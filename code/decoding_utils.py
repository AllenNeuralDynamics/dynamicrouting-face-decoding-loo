from ast import alias
import datetime
import os
import random

os.environ["RUST_BACKTRACE"] = "1"
# os.environ['POLARS_MAX_THREADS'] = '1'
os.environ["TOKIO_WORKER_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["RAYON_NUM_THREADS"] = "1"

import concurrent.futures as cf
import contextlib
import logging
import math
import multiprocessing
import uuid
from typing import Annotated, Iterable, Literal

import lazynwb
import numpy as np
import polars as pl
import polars._typing
import pydantic.functional_serializers
import pydantic_settings
import pydantic_settings.sources
import tqdm
import upath
import utils
from dynamic_routing_analysis.decoding_utils import (NotEnoughBlocksError,
                                                     decoder_helper)

logger = logging.getLogger(__name__)

# define run params here ------------------------------------------- #
Expr = Annotated[
    pl.Expr,
    pydantic.functional_serializers.PlainSerializer(
        lambda expr: expr.meta.serialize(format="json"), return_type=str
    ),
]


class BinnedRelativeIntervalConfig(pydantic.BaseModel):
    event_column_name: str
    start_time: float
    stop_time: float
    bin_size: float

    @property
    def intervals(self) -> list[tuple[float, float]]:
        start_times = np.arange(self.start_time, self.stop_time, self.bin_size)
        stop_times = start_times + self.bin_size
        return list(zip(start_times, stop_times))


class FeatureConfig(pydantic.BaseModel):
    model_label: str
    table_path: str
    features: list[str | int]
    """List of features to use for the model. Can be a list of strings (names of cols in DynamicTable) or
    integers (for indexing into TimeSeries.data array)"""

    @property
    def is_table(self) -> bool:
        return not isinstance(self.features[0], int)


feature_config_map: dict[str, FeatureConfig] = {
    "ear": FeatureConfig(
        model_label="ear",
        table_path="processing/behavior/lp_side_camera",
        features=["ear_base_l"],
    ),
    "jaw": FeatureConfig(
        model_label="jaw",
        table_path="processing/behavior/lp_side_camera",
        features=["jaw"],
    ),
    "nose": FeatureConfig(
        model_label="nose",
        table_path="processing/behavior/lp_side_camera",
        features=["nose_tip"],
    ),
    "whisker_pad": FeatureConfig(
        model_label="whisker_pad",
        table_path="processing/behavior/lp_side_camera",
        features=["whisker_pad_l_side"],
    ),
    "facial_features": FeatureConfig(
        model_label="facial_features",
        table_path="processing/behavior/lp_side_camera",
        features=["ear_base_l", "jaw", "nose_tip", "whisker_pad_l_side"],
    ),
    "facemap": FeatureConfig(
        model_label="facemap",
        table_path="processing/behavior/facemap_side_camera",
        features=list(range(10)),
    ),
}


class Params(pydantic_settings.BaseSettings):
    model_config  = pydantic.ConfigDict(protected_namespaces=()) # allow fields that start with `model_`

    # ----------------------------------------------------------------------------------
    # Required parameters
    result_prefix: str
    "An identifier for the decoding run, used to name the output files (can have duplicates with different run_id)"
    # ----------------------------------------------------------------------------------

    # Capsule-specific parameters -------------------------------------- #
    session_id: str | None = pydantic.Field(None, exclude=True, repr=True)
    """If provided, only process this session_id. Otherwise, process all sessions that match the filtering criteria"""
    run_id: str = pydantic.Field(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )  # created at runtime: same for all Params instances
    """A unique string that should be attached to all decoding runs in the same batch"""
    skip_existing: bool = pydantic.Field(True, exclude=True, repr=True)
    test: bool = pydantic.Field(False, exclude=True)
    logging_level: str | int = pydantic.Field("INFO", exclude=True)
    update_packages_from_source: bool = pydantic.Field(False, exclude=True)
    override_params_json: str | None = pydantic.Field("{}", exclude=True)
    use_process_pool: bool = pydantic.Field(True, exclude=True, repr=True)
    max_workers: int | None = pydantic.Field(None, exclude=True, repr=True)
    """For process pool"""

    # Decoding parameters ----------------------------------------------- #
    session_table_query: str = (
        "is_ephys & is_task & is_annotated & is_production & issues=='[]'"
    )
    input_data: list[str] = pydantic.Field(
        default_factory=lambda: [
            "facial_features",
            "ear",
            "nose",
            "jaw",
            "whisker_pad",
            "facemap",
        ]
    )
    n_repeats: int = 1
    crossval: Literal["5_fold", "blockwise"] = "5_fold"
    """blockwise untested with linear shift"""
    labels_as_index: bool = True
    """convert labels (context names) to index [0,1]"""
    decoder_type: Literal["linearSVC", "LDA", "RandomForest", "LogisticRegression"] = (
        "LogisticRegression"
    )
    regularization: float | None = None
    """ set regularization (C) for the decoder. Setting to None reverts to the default value (usually 1.0) """
    penalty: str | None = None
    """ set penalty for the decoder. Setting to None reverts to default """
    solver: str | None = None
    """ set solver for the decoder. Setting to None reverts to default """

    trials_filter: str | Expr = pydantic.Field(default_factory=lambda: pl.lit(True))
    """ filter trials table input to decoder by boolean column or polars expression"""

    feature_intervals: Literal["quiescent_stim_start_response"] = (
        "quiescent_stim_start_response"
    )

    @property
    def data_path(self) -> upath.UPath:
        """Path to delta lake on S3"""
        return (
            upath.UPath("s3://aind-scratch-data/dynamic-routing/face-decoding/results")
            / f"{'_'.join([self.result_prefix, self.run_id])}"
        )

    @property
    def json_path(self) -> upath.UPath:
        """Path to params json on S3"""
        return self.data_path.with_suffix(".json")

    @pydantic.computed_field(repr=False)
    @property
    def feature_interval_configs(self) -> list[BinnedRelativeIntervalConfig]:
        return {
            "quiescent_stim_start_response": [
                BinnedRelativeIntervalConfig(
                    event_column_name="quiescent_stop_time",
                    start_time=-1.5,
                    stop_time=0,
                    bin_size=1.5,
                ),
                BinnedRelativeIntervalConfig(
                    event_column_name="stim_start_time",
                    start_time=-2,
                    stop_time=7,
                    bin_size=0.250,
                ),
                BinnedRelativeIntervalConfig(
                    event_column_name="response_time",
                    start_time=-3,
                    stop_time=7,
                    bin_size=0.250,
                ),
            ],
        }[self.feature_intervals]

    @pydantic.computed_field(repr=False)
    def datacube_version(self) -> str:
        return utils.get_datacube_dir().name.split("_")[-1]

    # set the priority of the sources:
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        *args,
        **kwargs,
    ):
        # the order of the sources is what defines the priority:
        # - first source is highest priority
        # - for each field in the class, the first source that contains a value will be used
        return (
            init_settings,
            pydantic_settings.sources.JsonConfigSettingsSource(
                settings_cls, json_file="parameters.json"
            ),
            pydantic_settings.CliSettingsSource(settings_cls, cli_parse_args=True),
        )


# end of run params ------------------------------------------------ #


def decode_context_with_linear_shift(
    session_ids: str | Iterable[str],
    params: Params,
) -> None:
    if isinstance(session_ids, str):
        session_ids = [session_ids]
    session_ids = sorted(set(session_ids))

    if params.skip_existing and params.data_path.exists():
        logger.warning("Skipping existing is ignored!")
        existing = []
    else:
        existing = []

    def is_row_in_existing(row):
        return False

    if params.test: 
        params.input_data = ['facial_features', 'facemap']
        combinations_df = pl.DataFrame(
            {
                "session_id": [session_ids[0]] * len(params.input_data),
                "model_label": np.concatenate([[v] * len([session_ids[0]]) for v in params.input_data]),
            }
        )

    else: 
        combinations_df = pl.DataFrame(
            {
                "session_id": list(session_ids) * len(params.input_data),
                "model_label": np.concatenate([[v] * len(session_ids) for v in params.input_data]),
            }
        )

    logger.info(f"Processing {len(combinations_df)} unique session/model combinations")
    if params.use_process_pool:
        session_results: dict[str, list[cf.Future]] = {}
        future_to_session = {}
        lock = None  # multiprocessing.Manager().Lock() # or None
        with cf.ProcessPoolExecutor(
            max_workers=params.max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
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
                session_results.setdefault(row["session_id"], []).append(future)
                future_to_session[future] = row["session_id"]
                logger.debug(
                    f"Submitted decoding to process pool for session {row['session_id']}, {row['model_label']}"
                )
            for future in tqdm.tqdm(
                cf.as_completed(future_to_session),
                total=len(future_to_session),
                unit="model",
                desc="Decoding",
            ):
                session_id = future_to_session[future]
                if all(future.done() for future in session_results[session_id]):
                    logger.debug(f"Decoding completed for session {session_id}")
                    for f in session_results[session_id]:
                        try:
                            _ = f.result()
                        except Exception:
                            logger.exception(f"{session_id} | Failed:")
                    logger.info(f"{session_id} | Completed")

    else:  # single-process mode
        for row in tqdm.tqdm(
            combinations_df.iter_rows(named=True),
            total=len(combinations_df),
            unit="row",
            desc="decoding",
        ):
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


def wrap_decoder_helper(
    params: Params,
    session_id: str,
    model_label: str,
    lock=None,
) -> None:
    logger.debug(f"Getting trials for {session_id}")
    feature_config = feature_config_map[model_label]
    results = []
    trials = (
        utils.get_df("trials", lazy=True)
        .filter(
            pl.col("session_id") == session_id,
            params.trials_filter,
            # obs_intervals may affect number of trials available
        )
        .sort("trial_index")
        .collect()
    )
    if (
        trials["block_index"].n_unique() == 1
        and not (
            utils.get_df("session").filter(
                pl.col("session_id") == session_id,
                pl.col("keywords").list.contains("templeton"),
            )
        ).is_empty()
    ):
        logger.info(f"Adding dummy context labels for Templeton session {session_id}")
        trials = (
            trials.with_columns(
                pl.col("start_time")
                .sub(pl.col("start_time").min().over("session_id"))
                .truediv(10 * 60)
                .floor()
                .clip(0, 5)
                .alias("block_index")
                # short 7th block will sometimes be present: merge into 6th with clip
            )
            .with_columns(
                pl.when(pl.col("block_index").mod(2).eq(random.choice([0, 1])))
                .then(pl.lit("vis"))
                .otherwise(pl.lit("aud"))
                .alias("rewarded_modality")
            )
            .sort("trial_index")
        )
    if trials.n_unique("block_index") != 6:
        raise NotEnoughBlocksError(
            f'Expecting 6 blocks: {session_id} has {trials.n_unique("block_index")} blocks of observed ephys data'
        )
    logger.debug(f"Got {len(trials)} trials")
    
    nwb_path = next((p for p in utils.get_nwb_paths() if p.stem == session_id), None)
    if nwb_path is None:
        raise FileNotFoundError(
            f"NWB file for session {session_id} not found in the datacube directory"
        )
        
    if feature_config.is_table:
        columns = []
        for col in feature_config.features:
            columns.extend(
                [
                    f"{col}_x",
                    f"{col}_y",
                    f"{col}_temporal_norm",
                    f"{col}_likelihood",
                ]
            )
    else:
        columns = ["data"]
    
    try:
        df = lazynwb.scan_nwb(
            nwb_path,
            table_path=feature_config.table_path,
        ).select(*columns, "timestamps")
    except lazynwb.exceptions.InternalPathError: 
        raise lazynwb.exceptions.InternalPathError(f"{session_id} | {feature_config.table_path} not found in {nwb_path}") from None

    timestamps = df.select('timestamps').collect()['timestamps']
    if (timestamps[-1] < trials["stop_time"][-1]) | (timestamps[1] > trials["start_time"][0]):
        raise IndexError(f"For session_id {session_id}, video recording does not cover the entire task (timestamps: [{timestamps[0]}: {timestamps[-1]}], trials: [{trials['start_time'][0]}, {trials['stop_time'][-1]}]). Aborting.")
        
    def process_LP_column(df: pl.DataFrame | pl.LazyFrame, column_name: str):
        likelihood = pl.col(f"{column_name}_likelihood")
        temporal_norm = pl.col(f"{column_name}_temporal_norm")
        x = pl.col(f"{column_name}_x")
        y = pl.col(f"{column_name}_y")
        return (
            df.with_columns(
                (np.sqrt(x**2 + y**2)).alias("_xy"),
            )
            .with_columns(
                pl.when(
                    (likelihood >= 0.98)
                    & (
                        temporal_norm
                        <= temporal_norm.mean() + 3 * temporal_norm.std()
                    )
                )
                .then(pl.col("_xy"))
                .otherwise(None)
            )
            .with_columns(
                pl.col("_xy")
                .fill_nan(None)
                .interpolate()
                .backward_fill()
                .forward_fill()
                .alias(f"{column_name}_xy"),
            )
        )
    if feature_config.is_table:
        for col in feature_config.features:
            assert isinstance(
                col, str
            ), f"Expected string column name in feature config: {feature_config!r}"
            df = df.pipe(
                process_LP_column,
                column_name=col,
            )
        col_names = [f"{col}_xy" for col in feature_config.features]
        feature_timeseries = df.select(col_names).collect()[col_names].to_numpy()
    else:
        feature_timeseries = df.select("data").collect()['data'].to_numpy()[:, feature_config.features]

    for interval_config in params.feature_interval_configs:
        for start, stop in interval_config.intervals:

            event_times = trials.select(
                start=pl.col(interval_config.event_column_name) + start,
                stop=pl.col(interval_config.event_column_name) + stop,
            )
            
            binned_features = []
            for a, b in zip(event_times["start"], event_times["stop"]):
                start_index = np.searchsorted(timestamps, a, side="left")
                stop_index = np.searchsorted(timestamps, b, side="right") 
                binned_features.append(
                    np.nanmedian(feature_timeseries[start_index:stop_index, :], axis=0)
                )
            feature_array = np.array(binned_features) # shape (n_trials, n_features)
            assert feature_array.shape == (len(trials), len(feature_config.features)), f"{feature_array.shape=} != {len(trials)=}, {len(feature_config.features)=}"
            assert ~np.any(np.isnan(feature_array)), f"{session_id} | NaN values in {feature_config.model_label} feature array for {interval_config.event_column_name}"
            logger.debug(f"Got feature array: {feature_array.shape}")

            context_labels = trials["rewarded_modality"].to_numpy().squeeze()

            max_neg_shift = math.ceil(
                len(trials.filter(pl.col("block_index") == 0)) / 2
            )
            max_pos_shift = math.floor(
                len(trials.filter(pl.col("block_index") == 5)) / 2
            )
            shifts = tuple(range(-max_neg_shift, max_pos_shift + 1))
            logger.debug(f"Using shifts from {shifts[0]} to {shifts[-1]}")

            for repeat_idx in tqdm.tqdm(
                range(params.n_repeats),
                total=params.n_repeats,
                unit="repeat",
                desc=f"repeating {model_label} | {session_id}",
            ):

                for shift in (
                    *shifts,
                    None,
                ):  # None will be a special case using all trials, with no shift

                    is_all_trials = shift is None
                    if is_all_trials:
                        labels = context_labels
                        data = feature_array
                    else:
                        labels = context_labels[max_neg_shift:-max_pos_shift]
                        first_trial_index = max_neg_shift + shift
                        last_trial_index = len(trials) - max_pos_shift + shift
                        logger.debug(
                            f"Shift {shift}: using trials {first_trial_index} to {last_trial_index} out of {len(trials)}"
                        )
                        assert first_trial_index >= 0, f"{first_trial_index=}"
                        assert (
                            last_trial_index > first_trial_index
                        ), f"{last_trial_index=}, {first_trial_index=}"
                        data = feature_array[first_trial_index:last_trial_index, :]

                    assert data.shape == (
                        len(labels),
                        len(feature_config.features),
                    ), f"{data.shape=} != {len(labels)=}, {len(feature_config.features)=}"
                    logger.debug(
                        f"Shift {shift}: using data shape {data.shape} with {len(labels)} context labels"
                    )

                    # hyperparameter tuning 
                    reg_values = np.logspace(-4, 4, 9)
                    rnd_ind = np.random.choice(len(data), int(0.3 * len(data)), replace=False)
                    data_validation = data[rnd_ind] # randomly select 30% of the trials
                    labels_validation = labels[rnd_ind]
                    decoder_validation_accuracy = np.zeros((len(reg_values))) 
                    for rv, reg_value in enumerate(reg_values): 

                        _result_validation = decoder_helper(
                            data_validation,
                            labels_validation,
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
                        decoder_validation_accuracy[rv] = _result_validation["balanced_accuracy_test"].item()
                    params.regularization = reg_values[np.nanargmax(decoder_validation_accuracy)]
                    
                    # run test and train
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
                    result["balanced_accuracy_test"] = _result[
                        "balanced_accuracy_test"
                    ].item()
                    result["balanced_accuracy_train"] = _result[
                        "balanced_accuracy_train"
                    ].item()
                    result["time_aligned_to"] = interval_config.event_column_name
                    result["bin_size"] = interval_config.bin_size
                    result["bin_center"] = (start + stop) / 2
                    result["shift_idx"] = shift
                    result["repeat_idx"] = repeat_idx

                    if shift in (0, None):
                        result["predict_proba"] = _result["predict_proba"][
                            :, np.where(_result["label_names"] == "vis")[0][0]
                        ].tolist()
                    else:
                        # don't save probabilities from shifts which we won't use
                        result["predict_proba"] = None

                    if is_all_trials:
                        result["trial_indices"] = trials["trial_index"].to_list()
                    elif shift in (0, None):
                        result["trial_indices"] = trials["trial_index"].to_list()[
                            first_trial_index:last_trial_index
                        ]
                    else:
                        # don't save trial indices for all shifts
                        result["trial_indices"] = None

                    result["coefs"] = _result["coefs"][0].tolist()
                    result["is_all_trials"] = is_all_trials
                    results.append(result)
                    if params.test:
                        break
                if params.test:
                    break
            if params.test:
                logger.info(
                    f"Test mode: exiting after first bin in relative to {interval_config.event_column_name}"
                )
                break
        if params.test:
            logger.info("Test mode: exiting after first event intervals config")
            break

    with lock or contextlib.nullcontext():
        logger.info("Writing data")
        (
            pl.DataFrame(results)
            .with_columns(
                pl.lit(session_id).alias("session_id"),
                pl.lit(feature_config.model_label).alias("model_label"),
            )
            .cast(
                {
                    "shift_idx": pl.Int16,
                    "repeat_idx": pl.UInt16,
                    "time_aligned_to": pl.Enum(
                        [c.event_column_name for c in params.feature_interval_configs]
                    ),
                    "trial_indices": pl.List(pl.UInt16),
                    "predict_proba": pl.List(pl.Float64),
                    "coefs": pl.List(pl.Float64),
                }
            )
            .write_parquet(
                (params.data_path / f"{uuid.uuid4()}.parquet").as_posix(),
                compression_level=18,
                statistics="full",
            )
            # .write_delta(params.data_path.as_posix(), mode='append')
        )
    logger.info(
        f"Completed decoding for session {session_id}, {feature_config.model_label}"
    )
    # return results
