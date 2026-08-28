# HNSW Vector Search Engine From Scratch (C++)
### A 7-Day Build Plan — Systems + ML Depth Project for Google Fresher Shortlisting

---

## 0. What This Project Actually Is

You are building **your own version of the core algorithm that powers vector databases** like FAISS, ChromaDB, Pinecone, and Milvus — the thing you already used (as a black box) in DBQueryGPT's RAG pipeline.

The algorithm is called **HNSW — Hierarchical Navigable Small World graphs**. It solves one problem: given a database of millions of high-dimensional vectors (embeddings), find the *k* vectors closest to a query vector, **fast**, without comparing the query to every single vector (that would be too slow at scale — this is called brute-force / exact search, and it's O(N) per query).

HNSW gets you **approximate nearest neighbor (ANN) search** in roughly O(log N) time by building a multi-layer graph structure — think of it like a skip list, but for vectors in high-dimensional space instead of sorted numbers on a line.

### What you will end up with
1. A C++ library (`libhnsw`) implementing HNSW from scratch — insertion (with the paper's diversity heuristic, not naive pruning), search, no external ANN libraries.
2. A benchmark suite comparing your implementation against brute-force search and FAISS on recall@k, queries-per-second (QPS), and memory usage, on a real embedding dataset.
3. A profiling-driven optimization pass — a cache-friendly memory arena and hand-written SIMD (AVX2) distance functions — with a measured before/after table, not just claimed speedups.
4. A thin REST API wrapping the library so it's a runnable service, not just a script.
5. A README with your benchmark numbers, and a resume bullet you can defend in an interview line by line.

### Why this is the right project for you specifically
- You already understand *what* vector search does (from ChromaDB in DBQueryGPT) — now you'll understand *how* it works internally.
- It uses your CP strengths directly: priority queues, graph traversal, distance metrics, careful complexity analysis — this is not "build another CRUD app," it's algorithms + systems engineering.
- It's honest and defensible in an interview: you're not claiming to have beaten FAISS, you're claiming to understand and have implemented the algorithm FAISS is partly built on, with real numbers to back it up.

---

## 1. Tech Stack & Repo Structure

**Language:** C++17 (matches your existing C++ background from the Rubik's Cube solver)
**Build system:** CMake (you already know this)
**Dependencies (minimal, on purpose):**
- No ANN library dependencies for the core implementation (that's the whole point)
- `FAISS` — only as a benchmark baseline to compare against, not part of your engine
- A small HTTP library (`cpp-httplib`, header-only) or gRPC for the API layer
- `nlohmann/json` for request/response serialization
- Python (with `numpy`, `matplotlib`, `faiss-cpu`) — for the benchmarking scripts and plots, since Python is faster to prototype benchmarking/plotting in than C++

**Repo layout:**
```
hnsw-vector-engine/
├── CMakeLists.txt
├── include/hnsw/
│   ├── hnsw_index.hpp        # Main HNSWIndex class
│   ├── graph_node.hpp        # Node struct, layer connections
│   ├── distance.hpp          # L2, cosine, dot-product distance functions
│   ├── priority_queue_utils.hpp  # Bounded max-heap / min-heap helpers
│   └── serializer.hpp        # Save/load index to disk
├── src/
│   ├── hnsw_index.cpp
│   ├── distance.cpp
│   └── serializer.cpp
├── api/
│   ├── server.cpp            # REST/gRPC wrapper around HNSWIndex
│   └── server.hpp
├── benchmarks/
│   ├── generate_dataset.py   # Download/prepare SIFT1M or similar
│   ├── run_benchmark.cpp     # Your engine: build index, run queries, log recall+latency
│   ├── run_faiss_baseline.py # FAISS: same dataset, same queries, for comparison
│   └── plot_results.py       # Recall vs QPS curves
├── tests/
│   ├── test_insertion.cpp
│   ├── test_search_correctness.cpp
│   └── test_recall_small.cpp
├── data/                      # gitignored — datasets go here
├── README.md
└── RESULTS.md                 # Final benchmark numbers + graphs
```

---

## 2. The Algorithm, In Plain Terms (Read This Before Day 1)

**The core idea:** build a graph where each vector is a node, and nodes are connected to their approximate nearest neighbors. To search, start at a random entry point and greedily walk toward the query — always moving to whichever neighbor is closer to the query than your current position — until you can't improve anymore.

**The "hierarchical" part:** a single-layer graph like that is slow to navigate from a random start point (you need many hops). HNSW stacks multiple layers on top, like a skip list:
- The **top layer** has very few nodes, with long-range connections — good for jumping across the whole space quickly.
- Each layer below has progressively more nodes and shorter-range, denser connections.
- The **bottom layer (layer 0)** contains *every* node in the dataset.

**Insertion, at a glance:**
1. Randomly assign the new node a "top layer" — most nodes only exist at layer 0; a few lucky nodes (chosen via an exponentially decaying random distribution) also exist at layer 1, fewer at layer 2, etc. This is what gives you the skip-list-like structure.
2. Starting from the graph's global entry point, greedily search layer by layer from the top down to find the closest existing nodes to your new node.
3. At each layer the new node belongs to, connect it to its `M` nearest neighbors found during that search (M is a tunable parameter, typically 12–48).
4. Optionally prune existing nodes' connections if they now have too many neighbors, to keep the graph well-connected without any node having too many edges (a heuristic, not shown in most tutorials — this matters for search quality).

**Search (k-NN query), at a glance:**
1. Start at the entry point at the top layer.
2. Greedily descend: at each layer, walk to the closest neighbor to the query until no neighbor improves the distance; then drop one layer and repeat, using the best point found as the starting point for the layer below.
3. At layer 0, instead of stopping at the single closest point, keep a candidate list of size `ef` (search-time parameter — larger `ef` = higher recall, slower search) and do a wider best-first search to collect the top-k closest nodes.

This is the entire algorithm. Everything you build over the next 7 days is just carefully implementing these two procedures (insert, search) plus the supporting data structures (distance functions, bounded priority queues, layer bookkeeping) and then measuring how well it works.

**Read before Day 2:** the original paper — Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (arXiv 1603.09320). You don't need to read it front to back; read Sections 3–4 for the algorithm pseudocode (Algorithms 1–5 in the paper). This is the primary source you should implement against, not a blog post's paraphrase of it.

---

## 3. Day-by-Day Plan

> **Time budget assumption:** ~4-6 focused hours/day. If you only have 5 days instead of 7, merge Day 1+2 and Day 6+7 as noted below.

---

### **Day 1 — Foundations: Distance Functions, Data Structures, Repo Skeleton**

**Goal:** By end of day, you understand the math cold, and your repo compiles with working distance functions and basic node structures — no graph logic yet.

**Tasks:**
1. Set up the repo structure above with CMake. Get a "hello world" build working with the folder skeleton.
2. Implement `distance.hpp/.cpp`:
   - L2 (Euclidean) squared distance between two `float` vectors
   - Cosine distance
   - Dot-product distance
   - Write these as tight, cache-friendly loops (SIMD-friendly if you want to go further — not required for v1)
3. Implement `GraphNode`:
   - Vector data (or a pointer/index into a shared vector store — decide now: do you copy vectors into each node, or store one big contiguous array and reference by index? **Recommendation: contiguous float array + index-based references**, it's more cache-friendly and closer to how real implementations do it)
   - Per-layer neighbor lists (`std::vector<std::vector<uint32_t>>` — outer index = layer, inner = neighbor IDs)
   - The node's max layer
4. Write a small `main.cpp` that loads 1,000 random vectors, computes brute-force nearest neighbors for 10 queries, and prints them. This is your **ground truth generator** for correctness testing later — build it now while it's fresh.

**Key concepts to nail today:** why L2 vs cosine matters (embeddings are often normalized, in which case cosine ≈ dot product ≈ a monotonic function of L2 — know why), and why index-based references beat storing full vectors per node at scale.

**Using an AI coding agent (Antigravity) efficiently:** hand off the distance function implementations and CMake boilerplate — these are mechanical. Do NOT hand off "implement HNSW" as one big prompt; you'll get code you don't understand and can't defend in an interview. Use it task-by-task, and read every line it gives you before moving on.

**Checkpoint:** `./build/main` runs, prints brute-force nearest neighbors for a toy dataset, distance functions have unit tests passing.

---

### **Day 2 — Insertion Algorithm (the hard part, part 1)**

**Goal:** Implement `HNSWIndex::insert(vector)`, without worrying about search yet — you can verify insertion is working by inspecting graph structure directly.

**Tasks:**
1. Implement the **layer assignment** function: `assign_random_layer()` using the exponential decay distribution from the paper (`level = floor(-ln(uniform_random()) * mL)`, where `mL = 1 / ln(M)`). Test this in isolation — plot a histogram of layers across 10,000 calls and confirm it looks right (most nodes at layer 0, exponentially fewer at each layer up).
2. Implement `SearchLayer(query, entry_points, ef, layer)` — the core greedy best-first search primitive used both by insertion and by the final k-NN search. This returns the `ef` closest candidates found at a given layer, using:
   - A **candidate set** (min-heap, ordered by distance to query — "things to explore")
   - A **result set** (max-heap of size `ef`, ordered by distance — "best found so far")
   - Standard best-first search: pop closest candidate, expand its neighbors, add unvisited neighbors to candidate set and result set if they improve on the current worst result.
3. Implement the full `insert()` using `SearchLayer`:
   - Assign the new node's max layer
   - Descend from the graph's entry point through layers above the new node's max layer, greedily finding the single closest point at each layer (ef=1 search)
   - From the new node's max layer down to layer 0, run `SearchLayer` with `ef = efConstruction` (typically 100–200) to find candidate neighbors, then connect the new node to its `M` nearest of those candidates at each layer
   - Update the entry point if the new node's layer is higher than the current max layer in the graph
4. Implement neighbor selection using the **paper's diversity heuristic** (Algorithm 4 in Malkov & Yashunin), not the naive "just keep the M closest" rule. The naive rule tends to cluster a node's connections into one region of the candidate set, leaving the graph poorly connected in other directions. The heuristic instead builds the neighbor list one candidate at a time, closest-first, and only accepts a candidate if it is closer to the query than it is to every neighbor already accepted — if a candidate is closer to an already-picked neighbor than to the query itself, it's redundant (that region is already covered) and gets skipped in favor of a more distant but more diverse candidate. This is what actually keeps the graph "navigable" rather than clumpy, and it's the detail most tutorial implementations skip.
5. Implement neighbor-list pruning (the "shrink connections" step): after connecting a new node, check if any of its new neighbors now exceed `M` connections at that layer; if so, re-run the same diversity heuristic on that neighbor's existing connection list plus the new candidate to decide what to keep, rather than just dropping the single farthest one.

**Key concepts to nail today:** why insertion needs its own search (`SearchLayer` with `efConstruction`) that's separate from query-time search (`efSearch`) — they're the same function, different parameter, but the *reason* they're allowed to be different values is worth understanding. Also understand *why* the diversity heuristic matters, concretely: without it, recall at low `ef` (fast queries) degrades noticeably as the dataset grows, because the graph has redundant, clustered edges instead of edges that actually span the space well. This is a real, measurable effect you'll be able to point to later — build both versions if you have time on Day 3 (naive-pruning vs. heuristic-pruning) and compare recall@10 at ef=10 between them; that comparison is a genuinely interesting thing to show in an interview.

**Checkpoint:** after inserting 1,000 vectors one at a time, print the graph — check that layer 0 has all 1,000 nodes, higher layers have exponentially fewer, and no node's neighbor list exceeds `M` (or `M0` at layer 0, which is usually `2*M`).

---

### **Day 3 — Search / k-NN Query**

**Goal:** Implement `HNSWIndex::search(query, k, ef)` returning the top-k approximate nearest neighbors, and verify correctness against your Day 1 brute-force ground truth.

**Tasks:**
1. Implement `search()`:
   - Start at the entry point, descend from the top layer to layer 1 using `SearchLayer` with `ef=1` (single greedy path down, same as during insertion)
   - At layer 0, run `SearchLayer` with the user-provided `ef` (must be ≥ k) to get a wider candidate pool
   - Extract and return the top-k from that pool, sorted by distance
2. Write `test_search_correctness.cpp`: for your 1,000-vector toy dataset, run search for 50 random queries at a high `ef` (e.g., ef=200) and compare against brute-force ground truth. **At high ef, recall should approach ~100%** — if it doesn't, your bug is in insertion or search logic, not in an inherent approximation limit. This is your critical debugging checkpoint — don't move to Day 4 until this passes.
3. Add a `recall@k` calculation utility: `|approx_results ∩ true_results| / k`, averaged across queries. You'll reuse this constantly in benchmarking.
4. Now deliberately lower `ef` (e.g., ef=10, ef=20, ef=50) and observe recall drop and latency improve — plot this relationship informally (even just printing numbers) so you understand the recall/speed tradeoff you're about to formally benchmark.

**Key concepts to nail today:** the recall/latency/`ef` tradeoff is the single most important practical concept in ANN search — every real vector database exposes this knob, and being able to explain it clearly in an interview (with your own numbers) is exactly the kind of "I understand systems tradeoffs" signal that matters.

**Checkpoint:** recall@10 ≥ 95% at ef=200 on your toy dataset, verified against brute-force ground truth. If this fails, do not proceed — debug here.

---

### **Day 4 — Scale Up + Persistence**

**Goal:** Move from a 1,000-vector toy dataset to a real dataset (~100K–1M vectors), and add save/load so you don't have to rebuild the index every run.

**Tasks:**
1. Pick a real dataset. Recommended: **SIFT1M** (1 million 128-dim SIFT descriptors, the standard ANN benchmark dataset) or, more relevant to your background, **generate your own embedding dataset** using a sentence-transformer model on a text corpus (e.g., embed Wikipedia paragraphs or your own DBQueryGPT schema docs) — this makes the project more clearly "I connected this to real RAG/ML use" for your resume story. Either is defensible; SIFT1M is more standard for benchmarking, your own embeddings are more narratively tied to your DBQueryGPT project.
2. Implement `serializer.hpp/.cpp`: save the graph (all nodes, their vectors or vector-index references, per-layer neighbor lists, entry point, max layer) to a binary file, and load it back. Test round-trip: save, load, run the same queries, confirm identical results.
3. Build the index on your full dataset (100K–1M vectors) and measure **build time**. This alone is a useful benchmark number — note it down.
4. Profile insertion — if 1M vectors takes unreasonably long (hours), this is expected for a naive implementation; note where time is going (likely: distance computation in tight loops, or vector copies you can avoid) but don't over-optimize yet — get correctness and a working benchmark first, optimize on Day 7 if time allows.

**Key concepts to nail today:** why persistence matters for a "real" system (you don't want to rebuild a million-vector index every time you restart a service — this is an obvious question an interviewer might ask, "what happens on restart?").

**Checkpoint:** index built on a 100K+ vector real dataset, saved to disk, reloaded, and produces identical search results to the pre-save index.

---

### **Day 5 — Formal Benchmarking Against Brute-Force and FAISS**

**Goal:** Produce the actual numbers that go on your resume and in your README.

**Tasks:**
1. Write `run_benchmark.cpp`:
   - Load your saved index
   - Run a fixed query set (e.g., 1,000 queries) at several `ef` values (10, 20, 50, 100, 200)
   - For each `ef`, measure: **recall@10** (against precomputed ground truth), **QPS** (queries per second), **p50/p95/p99 latency**
   - Output as CSV
2. Write `run_faiss_baseline.py`: build a FAISS `IndexHNSWFlat` (FAISS's own HNSW implementation) or `IndexFlatL2` (exact brute-force) on the *same* dataset and queries, and measure the same metrics. This is your comparison baseline — you are not trying to "beat" FAISS (a mature, SIMD-optimized C++ library will likely be faster), you're showing you understand *why* the numbers differ and can reason about the gap.
3. Write `plot_results.py`: plot **recall@10 vs QPS** curves for your engine vs. brute-force vs. FAISS on the same axes. This single chart is the centerpiece of your `RESULTS.md` and your GitHub README.
4. Write up `RESULTS.md`: dataset used, vector count, dimensionality, hardware specs, the recall/QPS chart, and 3-4 sentences interpreting the results honestly (e.g., "at ef=100, my implementation achieves X% recall at Y QPS, compared to FAISS's Z QPS at the same recall — the gap is largely explained by [SIMD distance computation / better memory layout / graph pruning heuristics] which FAISS implements and mine does not").

**Key concepts to nail today:** benchmarking methodology matters as much as the numbers — same dataset, same queries, same hardware, multiple `ef` values, percentile latencies not just averages. This is literally how real systems teams evaluate infra changes, and demonstrating you know *how* to benchmark (not just that you ran one) is a strong signal.

**Checkpoint:** `RESULTS.md` exists with a real recall-vs-QPS chart comparing your engine, brute-force, and FAISS.

---

### **Day 6 — Profiling-Driven Optimization: Memory Arena + SIMD**

**Goal:** Take the naive-but-correct engine you benchmarked on Day 5 and make it fast, using measurements to justify every change — not optimizing blind.

**Why this order matters:** you now have a working, correctness-verified engine with a Day-5 baseline number in hand. Everything below is a *measured* improvement over that baseline, which is a far stronger story than having started with SIMD/arena code on Day 1 with nothing to compare against.

**Tasks:**
1. **Profile first.** Run `perf` (Linux) or a simple manual timer breakdown around your search loop, and confirm where time actually goes. In almost every naive HNSW implementation, distance computation dominates (often 80-90%+ of search time) — but *verify this on your own binary* rather than assuming it, and write down the number.
2. **Memory arena refactor:** replace your `std::vector<std::vector<uint32_t>>` neighbor lists and scattered node allocations with a single contiguous block — e.g., one big `std::vector<float>` for all vector data (indexed by `node_id * dim`), and one big flat array for neighbor lists per layer with fixed-width slots (pad unused slots with a sentinel value, e.g., `UINT32_MAX`). This removes the pointer-chasing that causes cache misses during graph traversal. Re-run your Day 5 benchmark after this change alone and record the delta — this is your first "before → after" number.
3. **SIMD distance functions:** rewrite your L2 and cosine distance functions using AVX2 intrinsics (`_mm256_*` functions on 8 floats at a time). Handle the tail case explicitly (when dimensionality isn't a multiple of 8 — most embedding models use 384, 768, or 1536 dims, so check whether your dimension divides evenly and write the remainder loop carefully). **Validate correctness first**: write a unit test comparing SIMD distance output against your original scalar implementation on 1,000 random vector pairs — they must match within floating-point tolerance (e.g., 1e-5) before you trust any speed number. Then re-run the benchmark and record this second "before → after" delta separately from the arena change, so you know which optimization contributed what.
4. Update `RESULTS.md` with a small table: baseline (Day 5) → +memory arena → +SIMD, showing QPS and p50/p95 latency at each stage. This progression table is more convincing to an interviewer than a single final number, because it shows you understand *why* each change helped, not just that the final number is good.

**Key concepts to nail today:** the discipline of "profile → change one thing → measure → record" instead of stacking multiple optimizations blindly and hoping the total is better. This is literally how performance work is done on real infra teams, and being able to walk through this table in an interview is a much stronger signal than the raw final QPS number.

**Checkpoint:** `RESULTS.md` has a 3-row before/after table (baseline, +arena, +SIMD) with matching correctness tests passing at every stage — recall@10 must be unchanged from Day 5's numbers (these are pure speed optimizations; if recall changes, you introduced a correctness bug, not a speedup).

---

### **Day 7 — API Layer, Polish, Documentation, Resume Bullet, Push**

**Goal:** Wrap the optimized engine in a runnable service, then make the repo something you're proud to link on your resume.

**Tasks:**
1. **Minimal API layer** (keep this scoped — you optimized yesterday, don't let this day balloon): add `cpp-httplib` (header-only, fastest to integrate) with three endpoints — `POST /insert`, `POST /search` (query vector, k, ef → top-k IDs + distances), `GET /stats` (node count, layer distribution). Skip gRPC and Docker unless you finish everything else with time to spare — a working REST API is enough to demonstrate "this is a service, not a script."
2. Write the main `README.md`:
   - One-paragraph summary (what it is, why you built it, what it does)
   - Architecture diagram (even simple ASCII) showing the memory arena layout and layer structure
   - How to build and run it (someone should be able to clone and run in <5 minutes)
   - Link to `RESULTS.md` with your baseline→arena→SIMD progression table and the recall-vs-QPS chart front and center
   - Explicitly credit the HNSW paper (Malkov & Yashunin) and note that you implemented the diversity heuristic (Algorithm 4), not just naive pruning — this signals you read the actual paper, not a blog summary
3. Clean up code: consistent naming, remove dead code/debug prints, comment the non-obvious parts (layer assignment formula, diversity heuristic, SIMD tail handling).
4. Write 2-3 unit tests you're missing (edge cases: empty index search, k > number of nodes, duplicate vectors).
5. Draft your resume bullet(s). Template:
   > *Implemented a Hierarchical Navigable Small World (HNSW) approximate nearest-neighbor search engine from scratch in C++ with SIMD-accelerated distance computation and a cache-optimized memory layout, achieving [X]% recall@10 at [Y] QPS ([Z]x speedup over the naive baseline) on a [N]-vector, [D]-dimensional embedding dataset.*

   Fill in real numbers from your `RESULTS.md` — don't round up or guess. If asked in an interview, you should be able to explain *every number* in that sentence, including why the SIMD/arena changes helped by exactly that much.
6. Push to GitHub with a clean commit history (a few meaningful commits per day, at minimum, shows real incremental work if anyone checks your commit graph).
7. **If you have leftover time:** pick one stretch goal from Section 5 below (Docker, gRPC, or a stretch item) instead of further polish — an implemented stretch goal is worth more than extra README polish.

**Checkpoint:** repo is public, `curl` against `/insert` and `/search` works end-to-end, README is clear enough that a stranger could understand and run the project in under 5 minutes, resume bullet is written with real numbers you can defend.

---

## 4. If You Only Have 5 Days (Compressed Plan)

- **Day 1** = original Day 1 + Day 2 (foundations + insertion, including the diversity heuristic) — this is the heaviest day, budget more hours
- **Day 2** = original Day 3 (search + correctness verification) — do not skip the recall verification step, it's your only correctness guarantee
- **Day 3** = original Day 4 (scale up + persistence)
- **Day 4** = original Day 5 (benchmarking) — this is non-negotiable, it's where your resume numbers come from
- **Day 5** = a trimmed version of original Day 6 + Day 7: pick **one** optimization only (memory arena is simpler and lower-risk than SIMD — do that one, skip SIMD if pressed for time), re-benchmark to get one before/after number, add a minimal `/search` + `/insert` API (skip Docker), write the README, and lock in the resume bullet.

**Priority order if you must cut something:** correctness verification (Day 2) and benchmarking (Day 4) are non-negotiable — never cut those. After that, cut in this order: SIMD first (it's the riskiest to get right under time pressure and the memory arena alone still gives you a real before/after story), then the API layer, then Docker/stretch goals. A correct, well-benchmarked library with one honest optimization number is more defensible in an interview than an unverified one wrapped in a nice API with unverified SIMD code.

---

## 5. Stretch Goals (Only If Days 1–7 Finish Early)

- **Docker + gRPC** — containerize the service and/or swap the REST API for gRPC (genuinely used heavily inside Google, so it's a nice detail to mention if you get to it, but not worth cutting correctness or benchmarking time for)
- **Product Quantization (PQ)** for memory compression — this is what real vector DBs use to fit billions of vectors in RAM; implementing even a basic version is a significant additional signal
- **Multi-threaded index construction** — parallelize insertion across threads with proper locking on shared graph structures (this is a hard, real concurrency problem with race conditions that are genuinely difficult to test — don't attempt unless Days 1-7 are solid first and you have real time left over)
- **Filtered search** (search with metadata constraints, e.g., "nearest neighbors where category = X") — a feature real production vector DBs need and toy implementations usually skip
- **AVX-512 or ARM NEON variant** of your Day 6 SIMD code, with a runtime CPU-feature check that falls back to your scalar version — this is the kind of portability detail that separates "wrote some intrinsics" from "understands how production code actually ships across different hardware"

---

## 6. Common Pitfalls (Read Before You Start)

- **Confusing `efConstruction` and `efSearch`** — they're the same underlying parameter used in two different contexts (build-time vs. query-time); mixing them up is the most common source of "my recall is terrible" bugs.
- **Forgetting to prune neighbor lists** — without pruning, the graph degrades into something closer to random connections at scale, and recall silently drops as you insert more vectors. If recall gets worse as your dataset grows, check pruning first.
- **Not fixing a random seed during benchmarking** — you need reproducible comparisons across runs (yours vs. FAISS's); always seed your RNG and log the seed.
- **Testing correctness only on tiny datasets** — bugs in layer-skipping logic often only show up at scale (10K+ vectors); don't declare victory after only testing on 100 vectors.
- **Comparing your single-threaded benchmark to FAISS's multi-threaded default** — make sure you're comparing apples to apples (thread counts, batch size) or explicitly call out the difference in `RESULTS.md`.

---

## 7. Primary Reference

Malkov, Y. A., & Yashunin, D. A. (2016). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* arXiv:1603.09320. Read Sections 3 and 4 for the algorithm definitions (Algorithms 1–5) before Day 2 — implement against this, not a secondary blog summary, so you can speak to the actual algorithm in an interview rather than someone else's simplified retelling of it.
