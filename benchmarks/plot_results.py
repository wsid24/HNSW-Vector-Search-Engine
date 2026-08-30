import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hnsw_scalar_csv = os.path.join(root_dir, 'benchmarks', 'results_hnsw_scalar.csv')
    hnsw_final_csv = os.path.join(root_dir, 'benchmarks', 'results_hnsw_final.csv')
    faiss_csv = os.path.join(root_dir, 'benchmarks', 'results_faiss.csv')
    out_png = os.path.join(root_dir, 'benchmarks', 'recall_vs_qps.png')
    
    df_hnsw_scalar = pd.read_csv(hnsw_scalar_csv)
    df_hnsw_final = pd.read_csv(hnsw_final_csv)
    df_faiss = pd.read_csv(faiss_csv)
    
    plt.figure(figsize=(10, 6))
    
    # Plot our baseline engine
    plt.plot(df_hnsw_scalar['qps'], df_hnsw_scalar['avg_recall@10'], marker='o', linestyle=':', label='Our HNSW (Scalar Baseline)', color='blue')
    for i, row in df_hnsw_scalar.iterrows():
        plt.annotate(f"ef={int(row['ef'])}", (row['qps'], row['avg_recall@10']), 
                     textcoords="offset points", xytext=(0,10), ha='center', color='blue', alpha=0.6)
                     
    # Plot our optimized engine
    plt.plot(df_hnsw_final['qps'], df_hnsw_final['avg_recall@10'], marker='o', label='Our HNSW (Arena + NEON)', color='green')
    for i, row in df_hnsw_final.iterrows():
        plt.annotate(f"ef={int(row['ef'])}", (row['qps'], row['avg_recall@10']), 
                     textcoords="offset points", xytext=(0,-15), ha='center', color='green')
                     
    # Plot FAISS
    plt.plot(df_faiss['qps'], df_faiss['avg_recall@10'], marker='s', label='FAISS (SIMD)', color='orange')
    for i, row in df_faiss.iterrows():
        plt.annotate(f"ef={int(row['ef'])}", (row['qps'], row['avg_recall@10']), 
                     textcoords="offset points", xytext=(0,10), ha='center', color='orange')
                     
    plt.xscale('log')
    plt.xlabel('Queries Per Second (QPS) - Log Scale')
    plt.ylabel('Recall@10')
    plt.title('Recall vs QPS: Our HNSW Optimization vs FAISS Baseline')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(out_png, dpi=300)
    print(f"Plot saved to {out_png}")

if __name__ == "__main__":
    main()
