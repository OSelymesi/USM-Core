# Vertical Difference Profiles and the Derivative-Wave Growth Exponent

## 1. Introduction and Motivation

In high-performance signal processing and irregular workload analysis, one usually studies sequences horizontally: how a term $a_n$ behaves as $n \to \infty$. This document introduces a **vertical viewpoint**.

Consider the forward difference operator:
$$\Delta a_n := a_{n+1} - a_n$$

By repeatedly applying $\Delta$, we build a difference table similar to Pascal's triangle. [cite_start]Instead of analyzing the sequence along time $n$, we analyze the behavior as we move **upwards** through the order of differences $k$ [cite: 461-465].

[cite_start]For many sequences, this vertical growth pattern acts as a stable **"derivative-wave fingerprint"**[cite: 467]. [cite_start]Unlike standard spectral analysis (FFT), this metric is robust against polynomial trends and provides a distinct signature for separating signal from noise[cite: 466, 541].

## 2. Definitions

### 2.1 The Vertical Difference Profile
Fix a sequence $a = (a_n)_{n \ge 0}$ and a window length $N$. The $k$-th forward difference is denoted by $a^{(k)}$.
We define the **Vertical Difference Profile** $W_a(k; N)$ as the average absolute magnitude of the $k$-th differences:

$$W_a(k; N) := \frac{1}{N - k} \sum_{n=0}^{N-k-1} |a^{(k)}_n|$$

[cite_start]This quantity captures how strongly the sequence reacts to repeated differencing [cite: 476-477].

### 2.2 The Derivative-Wave Growth Exponent
The core metric is the asymptotic exponential growth rate of this profile as the order of differentiation $k$ increases. We define the **Derivative-Wave Growth Exponent** $\gamma(a)$:

$$\gamma(a) := \limsup_{k \to \infty} \left( \sup_{N \ge ck} \frac{1}{k} \log W_a(k; N) \right)$$

[cite_start]Intuitively, $\gamma(a)$ is the growth rate per level in the difference table [cite: 480-481].

## 3. Universality Classes

[cite_start]The power of $\gamma(a)$ lies in its ability to classify sequences into three distinct structural categories[cite: 517].

### 3.1 Class I: Trend-Dominated ($\gamma = -\infty$)
If the sequence is dominated by a polynomial trend of any finite degree $d$, the differences eventually vanish.
* **Condition:** $a_n = P(n)$ where $P$ is a polynomial.
* **Result:** For $k > d$, $a^{(k)} \equiv 0$.
* **Exponent:** $\gamma(a) = -\infty$.

[cite_start]**Implication:** This metric is blind to polynomial drifts, making it an excellent filter for detecting underlying signals in drifting data [cite: 491-493].

### 3.2 Class II: Pure Discrete Waves
For a complex exponential wave $a_n = e^{i \omega n}$, differencing acts as a scaling operation. The signal is never destroyed, only amplified or attenuated based on frequency.
* **Result:** $W_a(k; N) = (2|\sin(\omega/2)|)^k$.
* **Exponent:** $\gamma(a) = \log(2|\sin(\omega/2)|)$.

**Implication:** The exponent encodes the frequency $\omega$. [cite_start]As $\omega \to \pi$ (high frequency), $\gamma \to \log 2$ [cite: 494-498, 501].

### 3.3 Class III: Noise and Entropy ($\gamma \approx \log 2$)
For sequences behaving like independent random variables (e.g., Bernoulli $\pm 1$ noise), the difference operator amplifies the variance combinatorially.
* **Heuristic:** The magnitude grows as $\approx 2^k$.
* **Exponent:** $\gamma(a) = \log 2$.

**Implication:** $\log 2$ acts as a "natural barrier" for maximum entropy in the difference table. [cite_start]This class includes random noise, bit-parity sequences, and even prime number indicators [cite: 502-508, 512-515].

## 4. Summary of Classification

| Behavior Type | Growth Exponent $\gamma(a)$ | Physical Interpretation |
| :--- | :--- | :--- |
| **Polynomial Trend** | $-\infty$ | Deterministic drift, low entropy. |
| **Pure Wave** | $log(2)|sin(Ω/2)|)$ | Structured oscillation, frequency-dependent. |
| **Noise / Chaos** | $\approx \log 2$ | Maximum entropy, uncorrelated fluctuations. |

## 5. Applications in High-Performance Computing

This theoretical framework provides a lightweight, trend-invariant method for:
1.  **Signal Classification:** Distinguishing between structured waves and random noise without expensive Fourier transforms.
2.  **Anomaly Detection:** Monitoring $\gamma(a)$ on streaming data; a shift towards $\log 2$ indicates signal degradation into noise.
3.  **Preprocessing:** Validating input data quality for irregular workloads (like those processed by USM) before expensive compute kernels are launched.

---
*Based on the internal research note: "Vertical Difference Profiles and the Derivative-Wave Growth Exponent" [v2].*
