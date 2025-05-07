import pandas as pd
import matplotlib.pyplot as plt

from datetime import timedelta
from collections import Counter

from cache_simulator import (
    LRUCache, LFUCache, FIFOCache, MRUCache, ProtectedLRUCache,
    run_cache_comparison, plot_hit_rates, 
    plot_comparative_metrics, analyze_workload,
    analyze_protected_lru, get_protection_strategies
)

# Example: Loading your page request data
# Assuming you already have a pandas Series object with page numbers
# If you have data in another format, you'll need to convert it first

# Option 1: If you have data in a CSV file with a column of page numbers
# page_requests = pd.read_csv('your_data.csv')['page_column']

# Option 2: If you have data as a list
# page_requests = pd.Series([1, 2, 3, 1, 4, 3, 2, 1, 5, ...])

# For this example, let's generate synthetic data with some locality patterns
# You would replace this with your actual data
import numpy as np

def get_data_from_csv(file_path):
    """
    Load page request data from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing page requests.
        
    Returns:
        Pandas Series of page requests.
    """
    df = pd.read_csv(file_path)
    return df


def generate_synthetic_data(n_requests=10000, n_pages=1000, locality_factor=1.5):
    """
    Generate synthetic page request data with temporal and spatial locality
    
    Args:
        n_requests: Number of page requests
        n_pages: Number of unique pages
        locality_factor: Zipf distribution parameter (higher means more skewed)
        
    Returns:
        Pandas Series of page requests
    """
    np.random.seed(42)
    
    # Create a power-law distribution for page popularity (spatial locality)
    popularity = np.random.zipf(locality_factor, n_pages)
    pages = np.random.choice(n_pages, size=n_requests, p=popularity/popularity.sum())
    
    # Add temporal locality by creating bursts of the same page
    for i in range(1000, n_requests):
        if np.random.random() < 0.3:  # 30% chance to repeat a recent page
            recent_window = max(1, int(i * 0.1))  # Look back 10%
            pages[i] = pages[np.random.randint(i-recent_window, i)]
    
    return pd.Series(pages)

# Generate synthetic data (replace with your actual data)
#page_requests = generate_synthetic_data()

def get_most_common_pages(df, target_date, days_back, k):
    """
    Find the k most common page_number values in a date range.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data with datetime and page_number columns
    target_date : datetime.datetime or str
        The end date of the period to analyze
    days_back : int
        Number of days to look back from the target_date
    k : int
        Number of most common values to return
    
    Returns:
    --------
    list
        A list of the k most common page_number values, or empty list if conditions not met
    """
    try:
        # Ensure target_date is in datetime format
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        # Calculate the start date
        start_date = target_date - timedelta(days=days_back - 1) # -1 because we include the target date
        
        # Filter the DataFrame for the date range
        mask = (df['started_at'] >= start_date) & (df['started_at'] <= target_date)
        filtered_df = df.loc[mask]
        
        # Return empty list if no data in range
        if filtered_df.empty:
            # debug
            # print("No filtered rows")
            return []
        
        # Get the most common page_number values
        page_counts = Counter(filtered_df['page_number'])
        most_common = page_counts.most_common(k)
        
        # Extract just the page numbers
        most_common_pages = [page for page, count in most_common]
        
        # Pad with None if fewer than k values
        if len(most_common_pages) < k:
            most_common_pages.extend([None] * (k - len(most_common_pages)))
        
        return most_common_pages
        
    except:
        # Return empty list on any exception
        return []

def get_row_index_by_date(df, target_date, date_column='started_at'):
    """
    Find the index of a row that matches a specific date.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    target_date : datetime.datetime or str
        The date to search for
    date_column : str, default='date'
        The name of the column containing dates
        
    Returns:
    --------
    list
        List of indices that match the target date, empty list if no match found
    """
    try:
        # Ensure target_date is in datetime format
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
            
        # Ensure the date column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Find rows that match the target date
        # For exact datetime match:
        matching_indices = df[df[date_column] == target_date].index.tolist()
        
        # If you want to match just the date part (ignoring time):
        matching_indices = df[df[date_column].dt.date == target_date.date()].index.tolist()
        
        return matching_indices[0] if matching_indices else None
    
    except:
        # Return empty list on any exception
        return []

# Example usage:
# indices = get_row_index_by_date(df, '2023-01-15')

REQUESTS_DATA = 'data/sorted_page_numbers_two.zip'

# Load page request data for the whole month data from CSV
df = pd.read_csv(REQUESTS_DATA, compression='zip', parse_dates=['started_at'], usecols=['page_number', 'started_at'])

# Now, let's get the indicies for the date range we want to analyze
# Get the index for the start date
start_date = '2023-11-16'
start_index = get_row_index_by_date(df, start_date)

# Get the do_not_evict list of size k, for last i days
k = 20
days = 7
my_k_list = get_most_common_pages(
    df,
    pd.to_datetime(start_date),
    days_back=days,
    k=k,    
)

# Get the index for the end date
end_date = '2023-11-17'
end_index = get_row_index_by_date(df, end_date)
# Filter the page_requests with the appropriate indices for the date range
page_requests = df.iloc[start_index:end_index]['page_number']

#print(f"Filtered page requests from {start_date} to {end_date} with indices {start_index} to {end_index}")
#print(f"Most common pages in the last {days} days: {my_k_list}")


# Analyze the workload characteristics
print("Analyzing workload characteristics...")
workload_stats = analyze_workload(page_requests)

print(f"Total requests: {workload_stats['total_requests']}")
print(f"Unique pages: {workload_stats['unique_pages']}")
print(f"Reuse distance statistics:")
for k, v in workload_stats['temporal_locality'].items():
    if isinstance(v, float):
        print(f"  {k}: {v:.2f}")
    else:
        print(f"  {k}: {v}")

# Define cache sizes to test
# Choose sizes based on your workload's unique pages count
unique_pages = workload_stats['unique_pages']
cache_sizes = [
    max(10, int(unique_pages * 0.01)),  # 1% of unique pages
    max(20, int(unique_pages * 0.05)),  # 5% of unique pages
    max(50, int(unique_pages * 0.1)),   # 10% of unique pages
    max(100, int(unique_pages * 0.2)),  # 20% of unique pages
    #max(200, int(unique_pages * 0.5))   # 50% of unique pages
]
print(f"Testing cache sizes: {cache_sizes}")

# Run comparison of different cache algorithms
print("Running cache simulations...")
results = run_cache_comparison(page_requests, cache_sizes, protected_pages=my_k_list)

# Print summary of results
print("\nHit Rate Summary:")
print("Cache Size | LRU    | LFU    | FIFO   | MRU   | PrLRU ")
print("-" * 50)
for size in cache_sizes:
    lru_hit = results['LRU'][size]['hit_rate']
    lfu_hit = results['LFU'][size]['hit_rate']
    fifo_hit = results['FIFO'][size]['hit_rate']
    mru_hit = results['MRU'][size]['hit_rate']
    #arc_hit = results['ARC'][size]['hit_rate']
    plru_hit = results['ProtectedLRU'][size]['hit_rate']
    print(f"{size:9d} | {lru_hit:.4f} | {lfu_hit:.4f} | {fifo_hit:.4f} | {mru_hit:.4f} | {plru_hit:.4f} ")

# Generate plots
print("\nGenerating plots...")
hit_rate_plot = plot_hit_rates(results, cache_sizes)
hit_rate_plot.savefig('hit_rate_comparison.png')

metrics_plot = plot_comparative_metrics(results, cache_sizes)
metrics_plot.savefig('cache_metrics_comparison.png')

# If you want to examine detailed statistics for a specific algorithm and size
print("\nDetailed statistics for LRU with largest cache size:")
lru_stats = results['LRU'][cache_sizes[-1]]
for metric, value in lru_stats.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.4f}")
    else:
        print(f"  {metric}: {value}")

# Additional analysis: Find optimal cache size based on hit rate improvement
print("\nAnalyzing cache size efficiency...")
for algo in results:
    prev_hit_rate = 0
    print(f"\n{algo} cache size efficiency:")
    for size in cache_sizes:
        hit_rate = results[algo][size]['hit_rate']
        improvement = hit_rate - prev_hit_rate
        improvement_per_unit = improvement / (size if prev_hit_rate == 0 else (size - cache_sizes[cache_sizes.index(size)-1]))
        
        print(f"  Size {size}: Hit rate {hit_rate:.4f}, Improvement {improvement:.4f}, Efficiency {improvement_per_unit:.6f}")
        prev_hit_rate = hit_rate

print("\nSimulation complete!")

# Example of how to use this in your main script:
# Choose a representative cache size
typical_size = cache_sizes[len(cache_sizes) // 2]

# Generate protection strategies
protection_strategies = get_protection_strategies(page_requests, typical_size)

# Analyze the impact
protected_results = analyze_protected_lru(page_requests, typical_size, protection_strategies)
print("\nProtected LRU results:")
for size, result in protected_results.items():
    print(f"  Cache size {size}: Hit rate {result['hit_rate']:.4f}, Miss rate {result['miss_rate']:.4f}")