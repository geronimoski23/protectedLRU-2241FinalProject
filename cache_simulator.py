import pandas as pd
import time
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional

class CacheBase:
    """Base class for all cache implementations"""
    
    def __init__(self, capacity: int):
        """Initialize cache with given capacity"""
        self.capacity = capacity
        self.cache = {}  # Actual cache storage
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.lookup_times = []
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache"""
        start_time = time.perf_counter()
        if key in self.cache:
            self.hits += 1
            value = self._get_item(key)
        else:
            self.misses += 1
            value = None
        self.lookup_times.append(time.perf_counter() - start_time)
        return value
    
    def put(self, key: Any, value: Any) -> None:
        """Put item into cache"""
        if len(self.cache) >= self.capacity and key not in self.cache:
            self._evict()
            self.evictions += 1
        self._put_item(key, value)
    
    def _get_item(self, key: Any) -> Any:
        """Internal method to get item and update metadata"""
        return self.cache[key]
    
    def _put_item(self, key: Any, value: Any) -> None:
        """Internal method to put item and update metadata"""
        self.cache[key] = value
    
    def _evict(self) -> None:
        """Internal method to evict an item based on policy"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Return cache performance statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        miss_rate = self.misses / total_requests if total_requests > 0 else 0
        avg_lookup_time = sum(self.lookup_times) / len(self.lookup_times) if self.lookup_times else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'miss_rate': miss_rate,
            'evictions': self.evictions,
            'avg_lookup_time': avg_lookup_time
        }
    
    def reset_stats(self) -> None:
        """Reset cache statistics"""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.lookup_times = []


class LRUCache(CacheBase):
    """Least Recently Used Cache Implementation"""
    
    def __init__(self, capacity: int):
        super().__init__(capacity)
        # Using OrderedDict for LRU implementation
        self.cache = OrderedDict()
    
    def _get_item(self, key: Any) -> Any:
        """Get item and move to end (most recently used)"""
        value = self.cache.pop(key)
        self.cache[key] = value  # Move to end (most recently used)
        return value
    
    def _put_item(self, key: Any, value: Any) -> None:
        """Put item and update order"""
        if key in self.cache:
            self.cache.pop(key)
        self.cache[key] = value
    
    def _evict(self) -> None:
        """Evict least recently used item"""
        if self.cache:
            self.cache.popitem(last=False)  # Remove first item (least recently used)


class LFUCache(CacheBase):
    """Least Frequently Used Cache Implementation"""
    
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.frequency = defaultdict(int)  # Tracks access frequency
        self.item_time = {}  # Tracks insertion time for tie-breaking
        self.time = 0  # Internal timestamp
    
    def _get_item(self, key: Any) -> Any:
        """Get item and update frequency"""
        self.frequency[key] += 1
        return self.cache[key]
    
    def _put_item(self, key: Any, value: Any) -> None:
        """Put item and update frequency metadata"""
        self.cache[key] = value
        self.frequency[key] = self.frequency.get(key, 0) + 1
        self.time += 1
        self.item_time[key] = self.time
    
    def _evict(self) -> None:
        """Evict least frequently used item"""
        if not self.cache:
            return
            
        # Find minimum frequency
        min_freq = min(self.frequency.values())
        
        # Find all items with minimum frequency
        min_freq_items = [k for k, v in self.frequency.items() if v == min_freq and k in self.cache]
        
        # If multiple items have same frequency, remove oldest
        if min_freq_items:
            oldest_item = min(min_freq_items, key=lambda k: self.item_time[k])
            del self.cache[oldest_item]
            del self.frequency[oldest_item]
            del self.item_time[oldest_item]


class FIFOCache(CacheBase):
    """First In First Out Cache Implementation"""
    
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.order = []  # Tracks insertion order
    
    def _put_item(self, key: Any, value: Any) -> None:
        """Put item and update order"""
        if key not in self.cache:
            self.order.append(key)
        self.cache[key] = value
    
    def _evict(self) -> None:
        """Evict first inserted item"""
        if self.order:
            oldest_key = self.order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]


class MRUCache(CacheBase):
    """Most Recently Used Cache Implementation"""
    
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.access_order = []  # Tracks access order
    
    def _get_item(self, key: Any) -> Any:
        """Get item and update access order"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        return self.cache[key]
    
    def _put_item(self, key: Any, value: Any) -> None:
        """Put item and update access order"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        self.cache[key] = value
    
    def _evict(self) -> None:
        """Evict most recently used item"""
        if self.access_order:
            newest_key = self.access_order.pop()
            if newest_key in self.cache:
                del self.cache[newest_key]

class ProtectedLRUCache(CacheBase):
    """LRU Cache Implementation with Protected Pages that cannot be evicted"""
    
    def __init__(self, capacity: int, do_not_evict=None):
        """
        Initialize protected LRU cache
        
        Args:
            capacity: Maximum number of items in cache
            do_not_evict: List of page numbers that should not be evicted if possible
        """
        super().__init__(capacity)
        # Using OrderedDict for LRU implementation
        self.cache = OrderedDict()
        # List of protected pages that should not be evicted if possible
        self.do_not_evict = set(do_not_evict or [])
    
    def _get_item(self, key: Any) -> Any:
        """Get item and move to end (most recently used)"""
        value = self.cache.pop(key)
        self.cache[key] = value  # Move to end (most recently used)
        return value
    
    def _put_item(self, key: Any, value: Any) -> None:
        """Put item and update order"""
        if key in self.cache:
            self.cache.pop(key)
        self.cache[key] = value
    
    def _evict(self) -> None:
        """
        Evict least recently used item that is not in the do_not_evict list.
        If all items are protected, evict the least recently used protected item.
        """
        if not self.cache:
            return
            
        # First try to find a non-protected page to evict
        for key in list(self.cache.keys()):  # Start from LRU end
            if key not in self.do_not_evict:
                self.cache.pop(key)
                return
                
        # If all pages are protected, evict the LRU one anyway
        # This should only happen if the do_not_evict list is larger than the cache capacity
        self.cache.popitem(last=False)  # Remove first item (least recently used)

def simulate_cache(page_requests, cache_type, cache_sizes):
    """
    Simulate cache performance for different cache sizes
    
    Args:
        page_requests: Series or list of page requests
        cache_type: Class of cache to simulate
        cache_sizes: List of cache sizes to test
        
    Returns:
        Dictionary of performance statistics for each cache size
    """
    results = {}
    
    for size in cache_sizes:
        cache = cache_type(size)
        
        for page in page_requests:
            if cache.get(page) is None:
                # Cache miss - add to cache
                cache.put(page, True)
        
        results[size] = cache.get_stats()
    
    return results


def run_cache_comparison(page_requests, cache_sizes, protected_pages=[]):
    """
    Compare different cache algorithms
    
    Args:
        page_requests: Series or list of page requests
        cache_sizes: List of cache sizes to test
        
    Returns:
        Dictionary of results for each algorithm
    """
    do_not_evict = protected_pages

    cache_algorithms = {
        'LRU': LRUCache,
        'LFU': LFUCache,
        'FIFO': FIFOCache,
        'MRU': MRUCache,
        'ProtectedLRU': lambda capacity: ProtectedLRUCache(capacity, do_not_evict=do_not_evict),  # Example protected pages
        #'ARC': ARCCache,  # Uncomment if ARC is selected
    }
    
    results = {}
    
    for name, algorithm in cache_algorithms.items():
        results[name] = simulate_cache(page_requests, algorithm, cache_sizes)
    
    return results


def plot_hit_rates(results, cache_sizes):
    """
    Plot hit rates for different cache algorithms and sizes
    
    Args:
        results: Dictionary of results from run_cache_comparison
        cache_sizes: List of cache sizes tested
    """
    plt.figure(figsize=(10, 6))
    
    for algo, sizes in results.items():
        hit_rates = [sizes[size]['hit_rate'] for size in cache_sizes]
        plt.plot(cache_sizes, hit_rates, marker='o', label=algo)
    
    plt.xlabel('Cache Size')
    plt.ylabel('Hit Rate')
    plt.title('Cache Hit Rate vs Size')
    plt.legend()
    plt.grid(True)
    
    return plt


def plot_comparative_metrics(results, cache_sizes):
    """
    Plot comparative metrics for different algorithms at each cache size
    
    Args:
        results: Dictionary of results from run_cache_comparison
        cache_sizes: List of cache sizes tested
    """
    metrics = ['hit_rate', 'miss_rate', 'evictions']
    fig, axes = plt.subplots(len(cache_sizes), len(metrics), figsize=(15, 3*len(cache_sizes)))
    
    algorithms = list(results.keys())
    
    for i, size in enumerate(cache_sizes):
        for j, metric in enumerate(metrics):
            metric_values = [results[algo][size][metric] for algo in algorithms]
            
            if len(cache_sizes) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
                
            ax.bar(algorithms, metric_values)
            ax.set_title(f'Cache Size {size} - {metric}')
            ax.set_ylim(0, max(1.0, max(metric_values) * 1.1))
            
            # Add values on top of bars
            for k, v in enumerate(metric_values):
                if metric in ['hit_rate', 'miss_rate']:
                    ax.text(k, v + 0.01, f'{v:.2f}', ha='center')
                else:
                    ax.text(k, v + 0.01, f'{v}', ha='center')
    
    plt.tight_layout()
    return plt


def analyze_workload(page_requests):
    """
    Analyze page request workload characteristics
    
    Args:
        page_requests: Series or list of page requests
        
    Returns:
        Dictionary of workload statistics
    """
    # Convert to pandas Series if not already
    if not isinstance(page_requests, pd.Series):
        page_requests = pd.Series(page_requests)
    
    unique_pages = page_requests.nunique()
    total_requests = len(page_requests)
    
    # Calculate frequency of each page
    freq = page_requests.value_counts()
    
    # Working set sizes (number of unique pages) for different windows
    windows = [10, 100, 1000]
    working_sets = {}
    
    for w in windows:
        if len(page_requests) >= w:
            # Calculate rolling window of unique values
            working_set_sizes = []
            for i in range(0, len(page_requests) - w + 1, w):
                working_set_sizes.append(page_requests.iloc[i:i+w].nunique())
            
            if working_set_sizes:
                working_sets[w] = {
                    'min': min(working_set_sizes),
                    'max': max(working_set_sizes),
                    'avg': sum(working_set_sizes) / len(working_set_sizes)
                }
    
    # Calculate temporal locality (how soon pages are reused)
    temporal_locality = {}
    last_seen = {}
    reuse_distances = []
    
    for i, page in enumerate(page_requests):
        if page in last_seen:
            distance = i - last_seen[page]
            reuse_distances.append(distance)
        last_seen[page] = i
    
    if reuse_distances:
        temporal_locality['min_distance'] = min(reuse_distances)
        temporal_locality['max_distance'] = max(reuse_distances)
        temporal_locality['avg_distance'] = sum(reuse_distances) / len(reuse_distances)
        
        # Calculate percentiles of reuse distance
        percentiles = [25, 50, 75, 90, 95]
        for p in percentiles:
            temporal_locality[f'p{p}'] = np.percentile(reuse_distances, p)
    
    return {
        'unique_pages': unique_pages,
        'total_requests': total_requests,
        'top_pages': freq.head(10).to_dict(),
        'working_sets': working_sets,
        'temporal_locality': temporal_locality
    }


def analyze_protected_lru(page_requests, cache_size, protect_strategies):
    """
    Analyze the impact of different protection strategies on LRU cache
    
    Args:
        page_requests: Series or list of page requests
        cache_size: Fixed cache size to test
        protect_strategies: Dictionary mapping strategy names to lists of protected pages
        
    Returns:
        Dictionary of performance stats for each strategy
    """
    results = {}
    
    # Standard LRU for baseline comparison
    standard_lru = LRUCache(cache_size)
    for page in page_requests:
        if standard_lru.get(page) is None:
            standard_lru.put(page, True)
    results['Standard LRU'] = standard_lru.get_stats()
    
    # Test each protection strategy
    for strategy_name, protected_pages in protect_strategies.items():
        protected_lru = ProtectedLRUCache(cache_size, do_not_evict=protected_pages)
        for page in page_requests:
            if protected_lru.get(page) is None:
                protected_lru.put(page, True)
        results[strategy_name] = protected_lru.get_stats()
    
    # Print comparison
    print(f"\nProtected LRU Analysis (Cache Size: {cache_size})")
    print("Strategy | Hit Rate | Improvement")
    print("-" * 50)
    baseline_hit_rate = results['Standard LRU']['hit_rate']
    for strategy, stats in results.items():
        improvement = stats['hit_rate'] - baseline_hit_rate
        print(f"{strategy:20s} | {stats['hit_rate']:.4f} | {improvement:+.4f}")
    
    return results

# Example usage:
# Find the most frequently accessed pages
def get_protection_strategies(page_requests, cache_size):
    """Generate protection strategies based on page access patterns"""
    if not isinstance(page_requests, pd.Series):
        page_requests = pd.Series(page_requests)
    
    # Get most frequent pages
    top_pages = page_requests.value_counts().head(min(10, cache_size // 2))
    frequent_pages = list(top_pages.index)
    
    # Early pages (pages that appear in the first 10% of requests)
    early_pages = list(page_requests.iloc[:len(page_requests)//10].unique())
    if len(early_pages) > cache_size // 2:
        early_pages = early_pages[:cache_size // 2]
    
    return {
        'Protect Most Frequent': frequent_pages,
        'Protect Early Pages': early_pages,
        'Hybrid Protection': list(set(frequent_pages[:5] + early_pages[:5]))
    }



# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    
    # Generate page request pattern with localities
    # 80% of requests come from 20% of pages (Pareto principle)
    n_requests = 10000
    n_pages = 1000
    
    # Create a power-law distribution for page popularity
    popularity = np.random.zipf(1.5, n_pages)
    pages = np.random.choice(n_pages, size=n_requests, p=popularity/popularity.sum())
    
    # Add some temporal locality by repeating recent pages
    for i in range(1000, n_requests):
        if np.random.random() < 0.3:  # 30% chance to repeat a recent page
            recent_window = max(1, int(i * 0.1))  # Look back 10%
            pages[i] = pages[np.random.randint(i-recent_window, i)]
    
    page_requests = pd.Series(pages)
    
    # Analyze workload
    workload_stats = analyze_workload(page_requests)
    print("Workload Analysis:")
    print(f"Total requests: {workload_stats['total_requests']}")
    print(f"Unique pages: {workload_stats['unique_pages']}")
    print(f"Average reuse distance: {workload_stats['temporal_locality'].get('avg_distance', 'N/A')}")
    
    # Run simulations with different cache sizes
    cache_sizes = [10, 50, 100, 200, 500]
    results = run_cache_comparison(page_requests, cache_sizes)
    
    # Plot results
    hit_rate_plot = plot_hit_rates(results, cache_sizes)
    metrics_plot = plot_comparative_metrics(results, cache_sizes)
    
    print("\nSimulation complete. Check the plots for results.")
    
    # Show plots
    hit_rate_plot.show()
    metrics_plot.show()