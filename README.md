# Simple PINN-Based Heston Option Pricing

This project demonstrates a simple Physics-Informed Neural Network (PINN) approach to price European call options under the Heston stochastic volatility model.

---

## Data Preparation

The dataset includes:

- **Collocation points:** Random samples of $$(t, S, v)$$ where $$t \in (0.01, 1.0)$$, $$S \in [0,200]$$, $$v \in [0.01, 0.5]$$
- **Initial (payoff) points:** Points at $$t = 0$$ for various $$S$$ and $$v$$
- **Boundary points:** Along the domain edges, at e.g. $$S = 0$$, $$S = 200$$, $$v = 0.01$$, $$v = 0.5$$, with random times

All Heston model parameters are fixed: $$\kappa = 2.0$$, $$\theta = 0.04$$, $$\sigma = 0.3$$, $$\rho = -0.7$$, $$r = 0.03$$, $$K = 100$$

---

## The Heston Framework

The Heston model captures option prices as functions $$V(t,S,v)$$, where $$S$$ is the spot, $$v$$ is variance.

The price function solves the following PDE:

$$\frac{\partial V}{\partial t} + r S \frac{\partial V}{\partial S} + \kappa (\theta-v) \frac{\partial V}{\partial v} + \frac{1}{2} v S^2 \frac{\partial^2 V}{\partial S^2} + \rho \sigma v S \frac{\partial^2 V}{\partial S \partial v} + \frac{1}{2}\sigma^2 v \frac{\partial^2 V}{\partial v^2} - r V = 0$$

with:

- **Initial condition (European call payoff):** $$V(0, S, v) = \max(S-K, 0)$$
- **Boundary example:** $$V(t, 0, v) = 0$$ and as $$S \to \infty$$, $$V(t, S, v) \approx S - K e^{-rt}$$

---

## Loss Function

The PINN is trained by minimizing these terms:

- **PDE loss:**  
  $$L_{PDE} = \frac{1}{N_c} \sum_{i=1}^{N_c} \left[ \text{PDE}(t_i, S_i, v_i, V) \right]^2$$

- **Initial loss:**  
  $$L_{payoff} = \frac{1}{N_p} \sum_{j=1}^{N_p} \left[V(0, S_j, v_j) - \max(S_j - K, 0)\right]^2$$

- **Boundary loss:** Mean squared error at sampled edge points.

---

## Implementation

- *Feedforward neural network*: Receives $$(t, S, v)$$, outputs an estimated price.
- *Optimizer*: Adam, run for thousands of steps or until loss stabilizes.
- *Comparison*: PINN solution is checked at 1000 random test points against Monte Carlo prices for the same $(t, S, v)$.

Final metrics include MAE, RMSE, and $$R^2$$, with visual comparison showing model fit.

---

## Areas for Improvement

- **Increase deep OTM sampling:** Oversampling zones where $$S \ll K$$ makes the network better at predicting when a call should be nearly worthless.
- **Stronger boundary and initial loss:** Penalize mismatches at $$S=0$$, $$t=0$$, and $$S\to\infty$$ more strongly.
- **Smoother output activation:** Prefer $$\mathrm{Softplus}$$ over ReLU for improved gradient flow and differentiability.
- **Parametric PINN as extension:** For more general models, the network can be trained to accept $$(\kappa, \theta, \sigma, \rho, r, K)$$ as additional inputs, allowing flexible repricing for many Heston scenarios.
- **Better treatment of low or zero price regions:** Explicitly add loss or penalties when PINN predicts nonzero prices in the deep OTM regime.

---

This notebook keeps things basic—no Greek calculation or parameteric solver features—but the structure can be extended for more advanced financial deep learning research.
