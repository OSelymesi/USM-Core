# USM: Universal Streaming Mechanism

**Stop regularizing your data. Regularize your kernel.**

USM is a lightweight, header-only CUDA C++ template engine designed to eliminate the "Regularity Tax" in GPU programming. It replaces complex, multi-stage kernel pipelines and expensive pre-processing with a single **Holistic Traversal**.

USM is engineered specifically for **irregular, ragged, and nested workloads** where standard libraries (CUB, Thrust) force you to pad, sort, or launch thousands of tiny kernels.

---

## ⚡ The Paradigm Shift

### The Old Way: "Thread Ownership"
Classic GPU reduction assumes every thread owns a specific stream or segment.
* **The Flaw:** If Stream A has 1 element and Stream B has 10,000, you get massive warp divergence and load imbalance.
* **The Cost:** You must pad data to fixed lengths (wasting memory) or run multiple kernels for different sizes (wasting launch latency).

### The USM Way: "Holistic Traversal"
USM inverts the model. Threads do not "own" streams. Instead, a swarm of threads traverses the entire data field in a unified **Grid-Stride Loop**.
* **Navigator:** A lightweight functor resolves context on-the-fly (e.g., *"Which user does this event belong to?"*).
* **Operator:** Executes atomic business logic in that context.
* **The Result:** Perfect load balancing without pre-processing. Long and short streams are processed by the same wave of threads.

---

## 🚀 Benchmarks & Performance

USM is not just about raw speed; it's about architectural efficiency. By collapsing multi-stage pipelines into a single kernel, it reduces VRAM traffic and CPU overhead.

**Measured on Desktop NVIDIA GPU vs. Optimized Multi-Kernel Baselines:**

| Scenario | Problem Type | Baseline Time | USM Time | Speedup | Why USM Wins |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ragged Reduction** | 10M elements $\to$ 100k variable length streams | 5.51 ms | **2.25 ms** | **2.45x** | Eliminates binary search overhead per thread & warp divergence. |
| **Nested Analytics** | 5M events $\to$ Items $\to$ Users (Hierarchical) | 0.65 ms | **0.47 ms** | **1.38x** | **Zero-Copy / Single-Pass.** Updates multiple levels simultaneously without intermediate VRAM buffers. |
| **Mixed Batch** | Heterogeneous Ops (Sum, Max, L2) | *N/A* | 1.83 ms | *Feature* | **Capabilities Demo.** Fuses different operations in one batch (impossible in standard libs). |

> **Note:** The "Nested Analytics" speedup of ~1.4x is purely computational. In a real-world end-to-end pipeline, the savings are significantly higher because USM eliminates the need to allocate and write intermediate buffers to global memory.

---

## 🛠️ Usage

USM is header-only. Just include `usm_core.cuh`.

### Example: Nested RecSys Pipeline
Compute aggregates for `Items` and `Users` from a stream of `Events` in a **single pass**.

```cpp
#include "usm_core.cuh"

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
        
        // Update BOTH layers simultaneously
        atomicAdd(&item_outputs[context.first], val);
        atomicAdd(&user_outputs[context.second], val);
    }
};

// 3. Launch
// No loops, no streams, no complexity. Just launch.
usm::launch(num_events, HierarchyNav{...}, DualAggregator{...});
