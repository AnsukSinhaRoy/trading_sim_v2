import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("SplitAdjuster")

def adjust_corporate_actions(
    input_dir: str = "E:/data_1m", 
    output_dir: str = "E:/data_1m_adjusted",
    timestamp_col: str = "date",
    price_cols: list = ["open", "high", "low", "close"],
    vol_col: str = "volume",
    split_threshold: float = 1.4 # Trigger adjustment if Overnight Ratio > 1.4
):
    """
    Detects splits/bonus issues by looking for massive overnight price gaps
    and adjusts historical minute-level OHLCV data backwards.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(in_path.glob("*.csv"))
    log.info(f"Found {len(csv_files)} files to process in {in_path}")
    
    for file in tqdm(csv_files, desc="Adjusting Symbols"):
        try:
            # 1. Load the full 1-minute history into memory
            df = pd.read_csv(file)
            if timestamp_col not in df.columns:
                continue
                
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values(timestamp_col).reset_index(drop=True)
            
            # 2. Extract daily EOD Close and SOD Open
            # Create a temporary daily grouping
            df['day'] = df[timestamp_col].dt.date
            
            # Get the last close of each day and the first open of the next day
            daily_close = df.groupby('day')[price_cols[-1]].last()
            daily_open = df.groupby('day')[price_cols[0]].first()
            
            # Align T-1 Close with T Open
            daily_close_prev = daily_close.shift(1)
            
            # 3. Detect Splits (Calculate overnight ratio)
            # Example: 1000 prev close / 100 new open = 10.0 ratio (10-for-1 split)
            overnight_ratio = daily_close_prev / daily_open
            
            # Find days where the drop triggers our threshold
            split_days = overnight_ratio[overnight_ratio > split_threshold]
            
            if not split_days.empty:
                # We initialize an adjustment multiplier array of 1.0s
                df['split_adj'] = 1.0
                
                for split_date, raw_ratio in split_days.items():
                    # Round to the nearest common corporate action fraction 
                    # (e.g. 1.5 for 3:2, 2.0 for 1:1, 5.0 for 5:1, 10.0 for 10:1)
                    # This removes market gap noise from the exact ratio
                    clean_ratio = round(raw_ratio * 2) / 2 
                    
                    if clean_ratio < 1.1:
                        continue
                        
                    log.info(f"[{file.stem}] Detected {clean_ratio}:1 split on {split_date}")
                    
                    # Apply multiplier to all rows PRIOR to the ex-date
                    df.loc[df['day'] < split_date, 'split_adj'] *= clean_ratio

                # 4. Apply the calculated multipliers
                for col in price_cols:
                    if col in df.columns:
                        df[col] = df[col] / df['split_adj']
                        
                if vol_col in df.columns:
                    df[vol_col] = df[vol_col] * df['split_adj']
                    
                # Clean up temp columns
                df = df.drop(columns=['day', 'split_adj'])

            # 5. Save the adjusted file
            out_file = out_path / file.name
            df.to_csv(out_file, index=False)
            
        except Exception as e:
            log.error(f"Failed processing {file.name}: {str(e)}")

if __name__ == "__main__":
    # Point this to your actual raw data folder and where you want the clean data
    adjust_corporate_actions(
        input_dir="E:/data_1m", 
        output_dir="E:/data_1m_adjusted",
        timestamp_col="date" # Ensure this matches your CSV headers!
    )