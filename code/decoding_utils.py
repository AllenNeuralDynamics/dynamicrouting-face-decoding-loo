import datetime
import os
import pathlib
import random

os.environ["RUST_BACKTRACE"] = "1"
# os.environ['POLARS_MAX_THREADS'] = '1'
os.environ["TOKIO_WORKER_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["RAYON_NUM_THREADS"] = "1"

import contextlib
import logging
import uuid
from typing import Annotated, Iterable, Literal

import lazynwb
import numpy as np
import polars as pl
import polars_ds as pds
import pydantic.functional_serializers
import pydantic_settings
import pydantic_settings.sources
import tqdm
import upath
import utils
import functools
from dynamic_routing_analysis.decoding_utils import decoder_helper

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


FEATURE_CONFIG_MAP: dict[str, FeatureConfig] = {
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
INTERVAL_CONFIG_PRESETS = {
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
    "quiescent": [
        BinnedRelativeIntervalConfig(
            event_column_name="quiescent_stop_time",
            start_time=-1.5,
            stop_time=0,
            bin_size=1.5,
        ),
    ]
}

class Params(pydantic_settings.BaseSettings):
    model_config  = pydantic.ConfigDict(protected_namespaces=()) # allow fields that start with `model_`

    # ----------------------------------------------------------------------------------
    # Required parameters
    result_prefix: str
    "An identifier for the decoding run, used to name the output files (can have duplicates with different run_id)"
    # ----------------------------------------------------------------------------------

    # Capsule-specific parameters -------------------------------------- #
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
    max_workers: int | None = pydantic.Field(int(os.environ['CO_CPUS']), exclude=True, repr=True)
    """For process pool"""

    # Decoding parameters ----------------------------------------------- #
    session_table_query: str = (
        "is_ephys & is_task & is_annotated & is_production & issues=='[]'"
    )
    input_data: list[str] = pydantic.Field(
        default_factory=lambda: [
            "facial_features",
            "facemap",
            # "ear",
            # "nose",
            # "jaw",
            # "whisker_pad",
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

    # ensure feature_intervals is a valid preset in `interval_config_presets`
    feature_intervals: Literal[tuple(INTERVAL_CONFIG_PRESETS.keys())] = "quiescent" # type: ignore[valid-type]

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
        return INTERVAL_CONFIG_PRESETS[self.feature_intervals]

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
def process_lp_feature(df: pl.DataFrame | pl.LazyFrame, column_name: str, use_pca: bool = False):
    """Filters on temporal_norm and likelihood and appends a new <column_name> column (with None
    for non-qualifying rows) to the DataFrame, with xy being the Euclidean distance from pixel (0,0)
    if use_pca=False, or the projection of xy onto the first principle component if use_pca=True."""
    likelihood = pl.col(f"{column_name}_likelihood")
    temporal_norm = pl.col(f"{column_name}_temporal_norm")
    x = pl.col(f"{column_name}_x")
    y = pl.col(f"{column_name}_y")
    if use_pca:
        new_col = pds.principal_components(x, y, k=1, center=True).struct.field('pc1')
    else:
        new_col = (x**2 + y**2).sqrt()
    return (
        df
        # filter x and y based on likelihood and temporal_norm:
        # (do this before PCA so we don't feed in junk)
        .with_columns(
            [
                pl.when(
                    (likelihood >= 0.98)
                    & (
                        temporal_norm
                        <= temporal_norm.mean() + 3 * temporal_norm.std()
                    )
                )
                .then(x_or_y)
                .otherwise(None)
                
                for x_or_y in [x, y]
            ]
        )
        .with_columns(
            new_col.alias(column_name),
        )
        .with_columns(
            pl.col(column_name)
            .fill_nan(None)
            .interpolate()
            .backward_fill()
            .forward_fill()
            .alias(column_name),
        )
    )
    
@functools.cache
def get_lp_df(nwb_paths: tuple[str, ...]) -> pl.DataFrame:
    """Comes with all features in FEATURE_CONFIG_MAP['facial_features']
    - do (
        lf
        .with_columns(pl.concat_list(features).alias('data'))
        .select('data', "_nwb_path")
    )
    """
    lf = lazynwb.scan_nwb(nwb_paths, "processing/behavior/lp_side_camera")
    feature_cols = FEATURE_CONFIG_MAP["facial_features"].features
    for feature_col in feature_cols:
        
        lf = lf.pipe(
            process_lp_feature,
            column_name=feature_col,
        )
    return lf.select(feature_cols).collect()

@functools.cache
def get_facemap_df(nwb_paths: tuple[str, ...]) -> pl.DataFrame:
    a = []
    for p in nwb_paths:
        try:
            ts = lazynwb.get_timeseries(
                p,
                "processing/behavior/facemap_side_camera",
                exact_path=True,
                match_all=False,
            )
        except KeyError:
            continue
            # NOTE number of sessions with facemap and lp may be different
        a.append(
            {
                "data": ts.data[:, :10], # keep top 10 SVDs
                "timestamps": ts.timestamps[:],
                "_nwb_path": str(p), # single value will be broadcast when creating pl.DataFrame from the dict
            }
        )
    return pl.concat((pl.DataFrame(x) for x in a), rechunk=True)
 
# Do we want to do all combinations? LP and Facemap only, no need for individual features.  
# Wont it be less memory intense if we grab data for each session, make the input matrix and then concatenate? For now, lets get everything 


def decode_context(
    params: Params,
) -> None:
    
    # get nwbs to use 
    session_table = (
        utils.get_session_table()
        .filter(
            'is_production', 'is_video', 'is_task', 
            ~pl.col('is_opto_perturbation', 'is_injection_perturbation', 'is_opto_control', 'is_injection_control'),
        )
    )
    available_nwb_paths = [p for p in utils.get_nwb_paths() if p.stem in session_table['session_id']]
    session_table = session_table.join(
        pl.DataFrame({'session_id': [p.stem for p in available_nwb_paths], 'nwb_path': available_nwb_paths}),
        on='session_id',
    )
    assert not session_table.is_empty(), "No sessions found for the given criteria & available NWBs"
    
    templeton_sessions = session_table.filter('is_templeton', 'is_production')
    dr_sessions = session_table.filter('is_engaged', 'is_good_behavior')
    # TODO select individual blocks with good behavior, since we no longer need all 6 to do linear shift
    assert templeton_sessions.join(dr_sessions, on='session_id').is_empty(), "Templeton and DR session IDs should not overlap"
    
    if params.skip_existing and params.data_path.exists():
        logger.warning("Skipping existing is not implemented!")

    for name, sessions in (('Templeton', templeton_sessions), ('DR', dr_sessions)):
        assert len(sessions) > 0, f"No {name} sessions to use - check filtering criteria"
        for model_label in params.input_data:
            logger.info(f"Processing {model_label} for {len(sessions)} {name} sessions")
            try:
                wrap_decoder_helper(
                    params=params,
                    sessions=sessions,
                    model_label=model_label,
                    is_templeton=(name.lower() == 'templeton'),
                )
            except Exception:
                logger.exception(f'{name} session processing failed:')
            
            if params.test:
                logger.info(f"Test mode: exiting after {name} sessions")
                break


def wrap_decoder_helper(
    params: Params,
    sessions: pl.DataFrame,
    model_label: str,
    is_templeton: bool,
    lock=None,
) -> None:
    logger.debug("Getting trials for all sessions")
    feature_config = FEATURE_CONFIG_MAP[model_label]

    trials = (
        utils.get_df("trials", lazy=True)
        .filter(
            params.trials_filter,
            pl.col("session_id").is_in(sessions["session_id"]),
        )
    )
    # TODO get all data, take one subject out at a time

    if is_templeton:
        logger.info("Adding dummy context labels for Templeton sessions")
        trials = (
            trials
            .with_columns(
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
        )
        
    if feature_config.is_table:
        data_df = (
            get_lp_df()
            .with_columns(pl.concat_list(feature_config.features).alias('data'))
            .select('data', "_nwb_path")
        )
    else:
        data_df = get_facemap_df()    
    
    # drop sessions without total video coverage:      
    video_start_stop_times = (
        data_df
        .group_by("_nwb_path")
        .agg(
            pl.col('timestamps').min().alias('video_start_time'),
            pl.col('timestamps').max().alias('video_stop_time'),
        )
    )
    trials: pl.DataFrame = (
        trials.collect()
        .join(video_start_stop_times, on="_nwb_path", how="left")
        .filter(
            pl.col("video_start_time") <= pl.col("start_time").min().over('_nwb_path'),
            pl.col("video_stop_time") >= pl.col("stop_time").max().over('_nwb_path'),
        )
    )
    # everything from here on must ensure this alphabetical order of nwb_path and trial_index is preserved
    trials = trials.sort("_nwb_path", "trial_index")
    logger.debug(f"Got {len(trials)} trials")
        
    data_df = data_df.filter(pl.col("_nwb_path").is_in(trials["_nwb_path"].implode()))


    results = []
    for interval_config in params.feature_interval_configs:
        for interval_start, interval_stop in interval_config.intervals: # relative times from abs time in event column
            
            binned_features = [] # will be list (len = all trials) of np.ndarrays, each with shape (n_features,) 
            for nwb_path in trials["_nwb_path"].unique(maintain_order=True):
               
                event_times = (
                    trials
                    .filter(pl.col('_nwb_path') == nwb_path)
                    .select(
                        start=pl.col(interval_config.event_column_name) + interval_start,
                        stop=pl.col(interval_config.event_column_name) + interval_stop,
                    )
                )
        
                for a, b in zip(event_times["start"], event_times["stop"]):
                    data = (
                        data_df
                        .filter(
                            pl.col("timestamps").is_between(a, b, closed="left")
                        )['data'].to_list()
                    )
                    binned_features.append(np.nanmedian(data, axis=0)) # shape (n_features,)
                    
            feature_array = np.array(binned_features).T
            assert feature_array.shape == (len(trials), len(feature_config.features)), f"{feature_array.shape=} != {len(trials)=}, {len(feature_config.features)=}"
            assert ~np.any(np.isnan(feature_array)), f"feature array contains Nans"
            logger.debug(f"Got feature array: {feature_array.shape}")

            context_labels = trials["rewarded_modality"].to_numpy().squeeze()

            shifts = []
            repeat_idx = 0
            for excluded_subject_id in tqdm.tqdm(
                trials['subject_id'].unique(maintain_order=True),
                total=trials['subject_id'].unique().height,
                unit="subject",
                desc=f"{model_label} for subjects",
            ):

                for shift in (
                    *shifts,
                    None,
                ):  # None will be a special case using all trials, with no shift
                    
                    assert shift is None, 'linear shift not implemented yet'

                    is_all_trials = shift is None
                    train_row_mask = trials.with_row_index().filter(pl.col('subject_id') != excluded_subject_id).select('index').to_list()
                    test_row_mask = trials.with_row_index().filter(pl.col('subject_id') == excluded_subject_id).select('index').to_list()
                    
                    # e.g. 
                    labels = context_labels[train_row_mask, :]
                    data = feature_array[train_row_mask, :]
                    
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
                    result["bin_center"] = (interval_start + interval_stop) / 2
                    result["shift_idx"] = shift
                    result["repeat_idx"] = repeat_idx
                    result["excluded_subject_id"] = excluded_subject_id
                    
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
                pl.lit(is_templeton).alias("is_templeton"),
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
        f"Completed decoding for {feature_config.model_label} ({'templeton' if is_templeton else 'DR'}) with {len(results)} results"
    )
    # return results
