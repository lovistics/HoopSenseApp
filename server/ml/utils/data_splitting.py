"""
Utilities for splitting data into training, validation, and test sets.
"""
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

from app.core.logger import logger


def train_val_test_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        data: DataFrame containing features and target
        test_size: Proportion of data to use for testing
        val_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
        stratify_col: Column to use for stratified sampling
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if len(data) < 10:
        logger.warning("Dataset too small for reliable splitting")
        
    # Calculate the actual validation size relative to the remaining data after test split
    remaining_fraction = 1 - test_size
    val_fraction = val_size / remaining_fraction
    
    # Get stratification values if needed
    stratify = data[stratify_col] if stratify_col and stratify_col in data.columns else None
    
    # First split: separate out test set
    train_val, test = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )
    
    # Update stratification for the second split
    if stratify is not None:
        stratify = train_val[stratify_col]
    
    # Second split: separate training and validation sets
    train, val = train_test_split(
        train_val, 
        test_size=val_fraction,
        random_state=random_state,
        stratify=stratify
    )
    
    logger.info(f"Data split into train ({len(train)} rows), validation ({len(val)} rows), "
                f"and test ({len(test)} rows) sets")
    
    return train, val, test


def time_based_split(
    data: pd.DataFrame,
    date_column: str,
    test_cutoff_date: Union[str, datetime],
    val_cutoff_date: Optional[Union[str, datetime]] = None,
    val_size: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data based on time, which is more realistic for predictions.
    
    Args:
        data: DataFrame containing features and target
        date_column: Column containing dates
        test_cutoff_date: Date before which data goes to train/val, after which it goes to test
        val_cutoff_date: Optional date before which data goes to train, after to validation
        val_size: If val_cutoff_date is not provided, proportion of training data to use for validation
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column])
    
    # Convert cutoff dates to datetime if they're strings
    if isinstance(test_cutoff_date, str):
        test_cutoff_date = pd.to_datetime(test_cutoff_date)
    
    if val_cutoff_date and isinstance(val_cutoff_date, str):
        val_cutoff_date = pd.to_datetime(val_cutoff_date)
    
    # Split test data
    train_val = data[data[date_column] < test_cutoff_date].copy()
    test = data[data[date_column] >= test_cutoff_date].copy()
    
    # Split validation data
    if val_cutoff_date:
        train = train_val[train_val[date_column] < val_cutoff_date].copy()
        val = train_val[train_val[date_column] >= val_cutoff_date].copy()
    elif val_size:
        # If no validation cutoff date, use random split with the specified size
        train, val = train_test_split(train_val, test_size=val_size, random_state=42)
    else:
        # Default: use last 15% of train_val period for validation
        train_val_sorted = train_val.sort_values(by=date_column)
        val_count = int(len(train_val) * 0.15)
        train = train_val_sorted.iloc[:-val_count].copy()
        val = train_val_sorted.iloc[-val_count:].copy()
    
    logger.info(f"Time-based split: train period {train[date_column].min()} to {train[date_column].max()}, "
                f"val period {val[date_column].min()} to {val[date_column].max()}, "
                f"test period {test[date_column].min()} to {test[date_column].max()}")
    
    logger.info(f"Data split into train ({len(train)} rows), validation ({len(val)} rows), "
                f"and test ({len(test)} rows) sets")
    
    return train, val, test


def cross_validation_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    stratify_col: Optional[str] = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate cross-validation splits for k-fold cross-validation.
    
    Args:
        data: DataFrame containing features and target
        n_splits: Number of folds
        shuffle: Whether to shuffle the data before splitting
        random_state: Random seed for reproducibility
        stratify_col: Column to use for stratified sampling
    
    Returns:
        List of (train_fold, val_fold) tuples
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    
    if stratify_col and stratify_col in data.columns:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        split_indices = splitter.split(data, data[stratify_col])
    else:
        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        split_indices = splitter.split(data)
    
    # Generate the folds
    cv_splits = []
    for train_idx, val_idx in split_indices:
        train_fold = data.iloc[train_idx].copy()
        val_fold = data.iloc[val_idx].copy()
        cv_splits.append((train_fold, val_fold))
    
    logger.info(f"Created {n_splits} cross-validation folds")
    return cv_splits


def temporal_block_split(
    data: pd.DataFrame,
    date_column: str,
    n_splits: int = 3,
    test_size: int = 30,  # days
    gap: int = 0  # days
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create temporal blocks for time series data, ensuring each test set
    is forward in time from its training set, with optional gap.
    
    Args:
        data: DataFrame containing features and target
        date_column: Column containing dates
        n_splits: Number of train-test splits to create
        test_size: Size of each test set in days
        gap: Number of days to skip between train and test sets
    
    Returns:
        List of (train_fold, test_fold) tuples
    """
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column])
    
    # Sort by date
    data_sorted = data.sort_values(by=date_column).copy()
    
    # Get unique dates
    dates = data_sorted[date_column].dt.date.unique()
    
    # Create splits
    splits = []
    for i in range(n_splits):
        # Calculate end index for this split's test set
        if i == 0:
            test_end_idx = len(dates) - 1
        else:
            test_end_idx = len(dates) - 1 - i * (test_size + gap)
        
        # Ensure we have enough data for this split
        if test_end_idx - test_size < 0:
            logger.warning(f"Not enough data for split {i+1}, skipping")
            continue
        
        # Define test set date range
        test_start_date = dates[test_end_idx - test_size + 1]
        test_end_date = dates[test_end_idx]
        
        # Define train set date range (everything before gap)
        if gap > 0:
            train_end_date = dates[test_end_idx - test_size - gap]
        else:
            train_end_date = dates[test_end_idx - test_size]
        
        # Create masks and subsets
        test_mask = (data_sorted[date_column].dt.date >= test_start_date) & \
                    (data_sorted[date_column].dt.date <= test_end_date)
        train_mask = (data_sorted[date_column].dt.date <= train_end_date)
        
        test_set = data_sorted[test_mask].copy()
        train_set = data_sorted[train_mask].copy()
        
        splits.append((train_set, test_set))
        
        logger.info(f"Split {i+1}: Train data from {train_set[date_column].min().date()} to "
                    f"{train_set[date_column].max().date()}, Test data from "
                    f"{test_set[date_column].min().date()} to {test_set[date_column].max().date()}")
    
    return splits