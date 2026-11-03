import time
import threading
import multiprocessing

def heavy_calculation(n=10_000_000):
    """A CPU-bound task that counts down to zero."""
    while n > 0:
        n -= 1

if __name__ == '__main__':
    # We will run the calculation twice
    
    # --- 1. MULTITHREADING VERSION ---
    t1 = threading.Thread(target=heavy_calculation)
    t2 = threading.Thread(target=heavy_calculation)
    
    start_time = time.time()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(f"⏱️ Multithreading Time: {time.time() - start_time:.4f} seconds (Choked by GIL)")

    # --- 2. MULTIPROCESSING VERSION ---
    p1 = multiprocessing.Process(target=heavy_calculation)
    p2 = multiprocessing.Process(target=heavy_calculation)
    
    start_time = time.time()
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print(f"🚀 Multiprocessing Time: {time.time() - start_time:.4f} seconds (True Parallelism)")
