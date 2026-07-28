# USM-Core: Universal Streaming Mechanism

**Stop regularizing your data. Regularize your kernel.**

USM is a lightweight, header-only CUDA C++ template engine designed to eliminate the "Regularity Tax" in GPU programming. It replaces complex, multi-stage kernel pipelines and expensive pre-processing with a single Traversal

USM is engineered specifically for **irregular, ragged, and nested workloads** where standard libraries (CUB, Thrust) force you to pad, sort, or launch thousands of tiny kernels.

---

## ⚙️ How It Works (Implementation)

Standard approaches try to make the **data** regular so the GPU is happy. USM makes the **kernel** logic robust so the data can stay irregular.

1.  **Grid-Stride Traversal:** The kernel launches enough blocks to saturate the GPU. Threads iterate over the input array in a coalesced pattern (index `tid`, `tid + stride`, etc.).
2.  **Amortized Stream Resolution:** Instead of searching for the Stream ID for every element, USM exploits the fact that data is contiguous. Threads only check for stream boundaries when necessary, keeping the hot-path extremely fast.
3.  **Local Aggregation:** Partial results are accumulated in registers and shared memory.
4.  **Sparse Global Atomics:** Only the final aggregated chunks are written to global memory, preventing the "atomic bottleneck" typical of naive kernels.

---

## 🛠️ Usage

USM is header-only. Just include `usm_core.cuh`.

### 1. Basic Ragged Reduction
If you have a flat array of values and a list of offsets defining the streams:

```cpp
#include "usm_core.cuh"

// Input: 10M values, 100k streams defined by offsets
// offsets[i] is start, offsets[i+1] is end
usm::flat_reduce(
    d_values,       // Device pointer to data
    d_offsets,      // Device pointer to stream boundaries
    num_streams,    // Number of streams
    d_output,       // Output buffer
    usm::op::sum()  // Operation (Sum, Max, Min, L2, etc.)
);
```

### 2. Nested / Complex Logic (The "Functor" API)
For complex pipelines (e.g., RecSys events -> items -> users), use the `launch` API with custom navigators.

```cpp
// 1. Define Navigation (The "Map")
// Tells the kernel how to traverse the hierarchy from Event -> User
struct HierarchyNav {
    int* event_to_item;
    int* item_to_user;

    __device__ auto get_context(int event_idx) { 
        int item_id = event_to_item[event_idx];
        int user_id = item_to_user[item_id];
        return thrust::make_pair(item_id, user_id); 
    }
};

// 2. Define Operation (The "Action")
// Tells the kernel what to do at each step
struct DualAggregator {
    float* item_outputs;
    float* user_outputs;
    const float* event_values;

    __device__ void apply(int idx, auto context) {
        float val = event_values[idx];
        
        // Update BOTH layers simultaneously in one pass
        atomicAdd(&item_outputs[context.first], val);
        atomicAdd(&user_outputs[context.second], val);
    }
};

// 3. Launch
usm::launch(num_events, HierarchyNav{...}, DualAggregator{...});
```

---

## 🎯 Ideal Use Cases

USM is SOTA (State-of-the-Art) specifically for **fragmented, large-scale workloads** where `N_total` is large but `N_per_stream` varies wildly.

* **Recommender Systems:** User history aggregation (variable length sessions).
* **Graph Analytics:** Aggregating neighbor messages (power-law degree distribution).
* **Physical Simulations:** Unstructured meshes and particle systems.
* **NLP / Sequence Models:** Batching sequences of different lengths without padding.
* **Signal Processing:** Real-time classification of irregular sensor streams (using the $\gamma$ metric).

## 📦 Integration

1.  Clone the repo.
2.  Add `include/` to your include path.
3.  Compile with `nvcc -std=c++17`.

```bash
git clone https://github.com/OSelymesi/USM-Core.git
```

## 📜 License

MIT License. See `LICENSE` for details.
