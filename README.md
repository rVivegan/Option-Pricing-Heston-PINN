# Option-Pricing-Heston-PINN


This project offers a simple demonstration of using Physics-Informed Neural Networks (PINNs) to price European call options under the Heston stochastic volatility model.

---

## Data Preparation

The training data is crafted by simulating three types of points:

- **Collocation points:** Random samples of $$ (t, S, v) $$ inside the domain, where $$ t \in (0.01, 1.0) $$, $$ S \in [0, 200] $$, and $$ v \in [0.01, 0.5] $$.
- **Initial (payoff) points:** Points at $$ t = 0 $$ with different $$ S $$ and $$ v $$, enforcing early exercise payoff.
- **Boundary points:** Points along the domain edges, specifically at $$ S = 0 $$, $$ S = 200 $$, $$ v = 0.01 $$, and $$ v = 0.5 $$, with random times.

Fixed parameters used: $$ \kappa = 2.0, \theta = 0.04, \sigma = 0.3, \rho = -0.7, r = 0.03, K = 100 $$.

---

## The Heston Model

The Heston model frames the price of a European call as a function $$ V(t, S, v) $$ that solves a partial differential equation capturing both risky asset dynamics and stochastic volatility.

---

## Heston PDE

The governing PDE is:

$$
\frac{\partial V}{\partial t} 
+ r S \frac{\partial V}{\partial S} 
+ \kappa (\theta-v) \frac{\partial V}{\partial v}
+ \frac{1}{2} v S^2 \frac{\partial^2 V}{\partial S^2}
+ \rho \sigma v S \frac{\partial^2 V}{\partial S \partial v}
+ \frac{1}{2}\sigma^2 v \frac{\partial^2 V}{\partial v^2}
- r V = 0
$$

The model is trained to respect boundary and initial conditions:

- **Initial condition (payoff):** $$ V(0, S, v) = \max(S-K, 0) $$
- **Boundary conditions:** For example, $$ V(t, 0, v) = 0 $$ and $$ V(t, S \to \infty, v) \approx S - K e^{-rt} $$.

---

## Loss Function

The model minimizes a composite loss:

- **PDE loss:** The mean squared value of the residual across collocation points:

  $$
  L_{\text{PDE}} = \frac{1}{N_c} \sum_{i=1}^{N_c} \left[ \text{PDE}(t_i, S_i, v_i, V) \right]^2
  $$

- **Initial loss:** The mean squared difference between predicted and theoretical payoff:

  $$
  L_{\text{payoff}} = \frac{1}{N_p} \sum_{j=1}^{N_p} (V(0, S_j, v_j) - \max(S_j-K, 0))^2
  $$

- **Boundary loss:** The mean squared error at all domain edges.

---

## Implementation Summary

- A feedforward neural network receives $$ (t, S, v) $$ as input and outputs the estimated option price.
- The neural net is trained over tens of thousands of epochs using the Adam optimizer.
- After training, out-of-sample PINN predictions are compared to Monte Carlo-generated prices at random points within the training domain.
- Common error metrics like RMSE, MAE, and $$ R^2 $$ are tabulated and plotted to visualize overall fit.

---

## Areas for Improvement

- **Oversampling Deep OTM Points:** Future data preparation could further emphasize regions where $$ S \ll K $$, providing more "zero-price" training examples. This would help the network more accurately learn that the option should be worthless when far out-of-the-money.
- **Tighter Boundary Enforcement:** Increasing the penalty term or sample density at boundaries (especially for $$ S \approx 0 $$) could anchor predictions to zero in those regions.
- **Smoother Output Activation:** Using a smooth activation function like Softplus, rather than ReLU, at the output would make the prediction surface more suitable for financial derivatives and Greek calculations.
- **Parametric PINN Extension:** While this example fixes the Heston model parameters, a more advanced approach would input $$ \kappa, \theta, \sigma, \rho, r, K $$ alongside $$ t, S, v $$. This "parametric solver" setup enables the model to provide prices across a range of market scenarios, not just a single calibration.
- **Better OTM Generalization:** Adding explicit loss penalties for nonzero predictions where $$ S \ll K $$, or increasing training data in low-price regions, will further improve consistency with financial intuition.

---

This notebook is intended only as an introduction to PINN-based Heston pricing and may be extended for greater generality, stability, and quantitative accuracy.
