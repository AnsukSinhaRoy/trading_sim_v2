import pandas as pd
import sys
import os

def inspect_parquet(file_path):
    """
    Reads a parquet file and prints summary statistics and a data sample.
    """
    # 1. Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    try:
        # 2. Load the data
        # engine='pyarrow' is standard, but 'fastparquet' also works
        df = pd.read_parquet(file_path, engine='pyarrow')

        # 3. Print Summary Information
        print("-" * 40)
        print(f"FILE: {file_path}")
        print("-" * 40)
        
        # Dimensions
        print(f"\nSHAPE: {df.shape[0]} rows x {df.shape[1]} columns")
        
        # Column Names and Types
        print("\nCOLUMNS & DATA TYPES:")
        print(df.dtypes)
        
        # 4. Print a Sample
        print("\n" + "-" * 40)
        print("HEAD (First 5 rows):")
        print("-" * 40)
        # to_string() ensures the output doesn't get truncated in the terminal
        print(df.head().to_string()) 

        # Optional: Basic Statistics for numerical columns
        if not df.empty:
            print("\n" + "-" * 40)
            print("BASIC STATISTICS (Numerical):")
            print("-" * 40)
            print(df.describe().to_string())

    except Exception as e:
        print(f"An error occurred while reading the Parquet file:\n{e}")

if __name__ == "__main__":
    # Check if user provided a file path via command line
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # REPLACE THIS PATH if you don't want to use command line args
        path = "path/to/your/file.parquet" 
        
        # If the default path doesn't exist, prompt the user
        if path == "path/to/your/file.parquet" or not os.path.exists(path):
            path = input("Enter the full path to the parquet file: ").strip().strip('"').strip("'")

    inspect_parquet(path)