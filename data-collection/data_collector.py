"""
Data Collection System for Cyber-Physical Systems
==============================================

This module provides utilities for collecting, processing, and storing
time series data from various sensors and sources.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects and processes time series data from various sources.
    
    Features:
    - Data collection from multiple sensors/sources
    - Automatic data validation and cleaning
    - Standardized data storage format
    - Real-time data processing capabilities
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the data collector.
        
        Args:
            storage_path: Path to store collected data. Defaults to 'data' folder
                        in the current directory.
        """
        self.storage_path = Path(storage_path) if storage_path else Path(__file__).parent / "data"
        self.storage_path.mkdir(exist_ok=True)
        
        self.current_data: Dict[str, pd.DataFrame] = {}
        self._setup_storage()
        
    def _setup_storage(self):
        """Set up data storage directory structure."""
        (self.storage_path / "raw").mkdir(exist_ok=True)
        (self.storage_path / "processed").mkdir(exist_ok=True)
        (self.storage_path / "metadata").mkdir(exist_ok=True)
        
    def collect_data(
        self,
        source_id: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Collect data from a source and store it.
        
        Args:
            source_id: Unique identifier for the data source
            data: Dictionary containing the data points
            timestamp: Optional timestamp for the data. Defaults to current time.
            
        Returns:
            bool: True if data was successfully collected and stored
        """
        try:
            timestamp = timestamp or datetime.now()
            
            # Create or update source DataFrame
            if source_id not in self.current_data:
                self.current_data[source_id] = pd.DataFrame()
            
            # Add new data point
            new_data = {**data, "timestamp": timestamp}
            self.current_data[source_id] = pd.concat([
                self.current_data[source_id],
                pd.DataFrame([new_data])
            ]).reset_index(drop=True)
            
            # Save data periodically
            self._save_data(source_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting data from {source_id}: {str(e)}")
            return False
            
    def _save_data(self, source_id: str):
        """Save collected data to storage."""
        if source_id in self.current_data:
            df = self.current_data[source_id]
            
            # Save raw data
            raw_path = self.storage_path / "raw" / f"{source_id}.csv"
            df.to_csv(raw_path, index=False)
            
            # Save metadata
            metadata = {
                "last_update": datetime.now().isoformat(),
                "num_records": len(df),
                "columns": list(df.columns)
            }
            metadata_path = self.storage_path / "metadata" / f"{source_id}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
    def process_data(
        self,
        source_id: str,
        operations: List[Dict[str, Any]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Process collected data with specified operations.
        
        Args:
            source_id: ID of the data source to process
            operations: List of operations to apply. Each operation is a dict with:
                       - 'type': operation type (e.g., 'filter', 'aggregate')
                       - 'params': operation parameters
                       
        Returns:
            Processed DataFrame or None if processing fails
        """
        try:
            if source_id not in self.current_data:
                logger.error(f"No data found for source {source_id}")
                return None
                
            df = self.current_data[source_id].copy()
            
            if operations:
                for op in operations:
                    df = self._apply_operation(df, op)
                    
            # Save processed data
            processed_path = self.storage_path / "processed" / f"{source_id}_processed.csv"
            df.to_csv(processed_path, index=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing data for {source_id}: {str(e)}")
            return None
            
    def _apply_operation(
        self,
        df: pd.DataFrame,
        operation: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply a single processing operation to the data."""
        op_type = operation.get('type', '')
        params = operation.get('params', {})
        
        if op_type == 'filter':
            column = params.get('column')
            condition = params.get('condition')
            value = params.get('value')
            
            if all([column, condition, value]):
                if condition == 'greater_than':
                    df = df[df[column] > value]
                elif condition == 'less_than':
                    df = df[df[column] < value]
                elif condition == 'equals':
                    df = df[df[column] == value]
                    
        elif op_type == 'aggregate':
            group_by = params.get('group_by')
            agg_func = params.get('function', 'mean')
            
            if group_by:
                df = df.groupby(group_by).agg(agg_func).reset_index()
                
        return df