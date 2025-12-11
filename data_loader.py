"""
Data Loader Module for Texas Commercial Real Estate Data
Loads data from Google Drive with local caching
"""

import os
import pandas as pd
import gdown
from pathlib import Path

# Google Drive folder ID extracted from your URL
GDRIVE_FOLDER_ID = "1DRYCYv6scWVG9_XJ9i0smiJBNd6t1WKD"

# File names and their Google Drive file IDs
# You'll need to get the individual file IDs from the Google Drive folder
FILES = {
    '2018-2021 Non-Proprietary Land Sales_Appraisal Site.xlsx': None,
    '2018-2021 Non-Proprietary Land Sales_With Pins.xlsx': None,
    '2018-2021 Res Land Sales _10 Acres and Greater.xlsx': None,
    '2021 Non-Proprietary Land Sale Listing Notes.xlsx': None,
    '2022 Comm Land Sale Lisings - Non-proprietary.xlsx': None,
    '2022 Comm Land Sales - Non-proprietary.xlsx': None,
    '2023 Ag-Rural-or Large Acre Land Sales - Non-proprietary.xlsx': None,
    '2023 Comm Land Sale Listings - Non-proprietary.xlsx': None,
    '2023 Comm Land Sale and Listing Notes - Non-Proprietary.xlsx': None,
    '2023 Comm Land Sales - Non-proprietary.xlsx': None,
    '2024CommLandSales.xlsx': None,
    '2025 Comm Land Sales.xlsx': None,
}

DATA_DIR = Path(__file__).parent / 'data'

def download_file_from_gdrive(file_id, destination):
    """Download a file from Google Drive"""
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, str(destination), quiet=False)

def ensure_data_exists():
    """
    Ensure data directory exists and files are downloaded
    Returns True if data is ready, False otherwise
    """
    DATA_DIR.mkdir(exist_ok=True)

    # Check if any files exist locally
    existing_files = list(DATA_DIR.glob('*.xlsx'))

    if len(existing_files) == 0:
        print("⚠️  No data files found locally.")
        print("📁 Please add Excel files to the 'data' folder, or we can use sample data.")
        return False

    print(f"✅ Found {len(existing_files)} data files")
    return True

def load_texas_property_data(use_sample_if_missing=True):
    """
    Load all Texas commercial real estate data files

    Args:
        use_sample_if_missing: If True, returns sample data structure when files are missing

    Returns:
        dict: Dictionary of dataframes keyed by file name
    """
    datasets = {}

    # Check if data exists
    if not ensure_data_exists():
        if use_sample_if_missing:
            print("📊 Using sample data structure for demo purposes")
            return create_sample_datasets()
        else:
            raise FileNotFoundError("No data files found. Please add Excel files to the 'data' folder.")

    # Load all Excel files from data directory
    for file_path in DATA_DIR.glob('*.xlsx'):
        try:
            df = pd.read_excel(file_path)
            key = file_path.stem  # filename without extension
            datasets[key] = df
            print(f"✅ Loaded {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"❌ Error loading {file_path.name}: {e}")

    return datasets

def process_property_data(datasets):
    """
    Process multiple property datasets into a unified format

    Returns:
        pd.DataFrame: Combined and standardized property data
    """
    all_data = []

    for dataset_name, df in datasets.items():
        df_copy = df.copy()
        df_copy['data_source'] = dataset_name
        all_data.append(df_copy)

    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True, sort=False)

    # Clean and standardize column names
    combined_df.columns = [str(col).strip().lower().replace(' ', '_') for col in combined_df.columns]

    print(f"\n📊 Combined dataset: {combined_df.shape[0]} properties, {combined_df.shape[1]} columns")

    return combined_df

def create_property_knowledge_base(df, max_properties=200):
    """
    Convert property dataframe into formatted text chunks for RAG

    Args:
        df: DataFrame of property data
        max_properties: Maximum number of properties to include

    Returns:
        str: Formatted text representation of properties
    """
    try:
        # Filter to records with dates
        if 'document_date' in df.columns:
            df_filtered = df[df['document_date'].notna()].copy()
        else:
            df_filtered = df.copy()

        # Sample if too many
        if len(df_filtered) > max_properties:
            df_sample = df_filtered.sample(n=max_properties, random_state=42)
        else:
            df_sample = df_filtered

        property_texts = []
        available_cols = set(df_sample.columns)

        for idx, row in df_sample.iterrows():
            property_text = f"\n{'='*80}\nPROPERTY RECORD #{idx}\n"

            if 'data_source' in available_cols and pd.notna(row.get('data_source')):
                property_text += f"Data Source: {row['data_source']}\n"

            property_text += f"{'-'*80}\n"

            # Transaction Info
            property_text += "\nTRANSACTION INFORMATION:\n"
            if 'document_date' in available_cols and pd.notna(row.get('document_date')):
                property_text += f"  Sale Date: {row['document_date']}\n"

            # Property Details
            property_text += "\nPROPERTY DETAILS:\n"
            for col in ['site_class', 'site_name', 'address', 'neighborhood', 'property_type']:
                if col in available_cols and pd.notna(row.get(col)):
                    property_text += f"  {col.replace('_', ' ').title()}: {row[col]}\n"

            # Financial Info
            property_text += "\nFINANCIAL INFORMATION:\n"
            for col in available_cols:
                if 'price' in col.lower() or 'amount' in col.lower():
                    value = row.get(col)
                    if pd.notna(value) and isinstance(value, (int, float)):
                        property_text += f"  {col.replace('_', ' ').title()}: ${value:,.2f}\n"

            # Size Info
            property_text += "\nSIZE INFORMATION:\n"
            for col in available_cols:
                if any(x in col.lower() for x in ['acre', 'size', 'sqft', 'square']):
                    value = row.get(col)
                    if pd.notna(value):
                        property_text += f"  {col.replace('_', ' ').title()}: {value}\n"

            property_texts.append(property_text)

        combined_text = "\n".join(property_texts)

        print(f"✅ Created knowledge base with {len(property_texts)} properties")
        print(f"   Total text length: {len(combined_text):,} characters")

        return combined_text

    except Exception as e:
        print(f"❌ Error creating knowledge base: {e}")
        # Return a minimal knowledge base to prevent complete failure
        return "Texas commercial real estate market data (2018-2025)"

def create_sample_datasets():
    """
    Create sample datasets for demo purposes when real data is not available
    """
    sample_data = {
        'sample_texas_properties': pd.DataFrame({
            'document_date': pd.date_range('2018-01-01', periods=50, freq='M'),
            'site_class': ['Vacant Land -Commercial'] * 50,
            'neighborhood': ['Austin Metro Area'] * 50,
            'sale_price': [100000 + i * 10000 for i in range(50)],
            'land_acres': [0.5 + i * 0.1 for i in range(50)],
        })
    }
    print("⚠️  Using sample data - add real Excel files to 'data' folder for full functionality")
    return sample_data

if __name__ == "__main__":
    # Test the loader
    print("Testing data loader...\n")
    datasets = load_texas_property_data()
    if datasets:
        combined = process_property_data(datasets)
        print(f"\n✅ Successfully loaded {len(datasets)} datasets")
        print(f"✅ Combined into {len(combined)} total records")
