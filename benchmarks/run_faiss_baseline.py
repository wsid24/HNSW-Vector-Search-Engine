import time
import numpy as np
import faiss
import os
import sys

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(root_dir, 'data', 'embeddings.bin')
    q_path = os.path.join(root_dir, 'data', 'queries.bin')
    gt_path = os.path.join(root_dir, 'data', 'ground_truth_full.txt')
    csv_path = os.path.join(root_dir, 'benchmarks', 'results_faiss.csv')
    
    print("Loading dataset...")
    db_floats = np.fromfile(db_path, dtype=np.float32).reshape(99000, 384)
    query_floats = np.fromfile(q_path, dtype=np.float32).reshape(1000, 384)
    
    print("Building FAISS HNSW index from scratch...")
    dim = 384
    M = 16
    efConstruction = 200
    
    # We must explicitly set OMP_NUM_THREADS=1 for a fair single-threaded benchmark
    # However, FAISS internal threads during add() might still be active. We time the build anyway.
    faiss.omp_set_num_threads(1)
    
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = efConstruction
    
    t0 = time.perf_counter()
    index.add(db_floats)
    t1 = time.perf_counter()
    build_time = t1 - t0
    print(f"Index built in {build_time:.3f} seconds.")
    
    print(f"Loading ground truth from {gt_path}...")
    ground_truth = []
    with open(gt_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            
            ids_part = parts[1]
            tokens = ids_part.split(',')
            
            gt_set = set()
            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                colon_idx = token.find(':')
                if colon_idx != -1:
                    vid = int(token[:colon_idx])
                    gt_set.add(vid)
            ground_truth.append(gt_set)
            
    if len(ground_truth) != 1000:
        print(f"Error: Expected 1000 ground truth queries, got {len(ground_truth)}")
        sys.exit(1)
        
    ef_values = [10, 20, 50, 100, 200]
    
    print("-" * 70)
    print(f"{'ef':<10}{'Recall@10':<15}{'QPS':<15}{'p50 (ms)':<10}{'p95 (ms)':<10}{'p99 (ms)':<10}")
    print("-" * 70)
    
    with open(csv_path, 'w') as csv_out:
        csv_out.write("ef,avg_recall@10,qps,p50_ms,p95_ms,p99_ms\n")
        
        for ef in ef_values:
            index.hnsw.efSearch = ef
            
            latencies = []
            total_recall = 0.0
            
            # Unbatched search
            for q_idx in range(1000):
                q_vec = query_floats[q_idx].reshape(1, -1)
                
                t_start = time.perf_counter()
                D, I = index.search(q_vec, 10)
                t_end = time.perf_counter()
                
                latencies.append((t_end - t_start) * 1000.0)  # ms
                
                # Check matches
                returned_ids = I[0]
                matches = sum(1 for ret_id in returned_ids if ret_id in ground_truth[q_idx])
                total_recall += matches / 10.0
                
            avg_recall = total_recall / 1000.0
            total_time_s = sum(latencies) / 1000.0
            qps = 1000.0 / total_time_s
            
            latencies.sort()
            p50 = latencies[int(1000 * 0.50)]
            p95 = latencies[int(1000 * 0.95)]
            p99 = latencies[int(1000 * 0.99)]
            
            print(f"{ef:<10}{avg_recall:<15.4f}{qps:<15.2f}{p50:<10.3f}{p95:<10.3f}{p99:<10.3f}")
            csv_out.write(f"{ef},{avg_recall:.4f},{qps:.2f},{p50:.3f},{p95:.3f},{p99:.3f}\n")
            
    print("-" * 70)

if __name__ == "__main__":
    main()
