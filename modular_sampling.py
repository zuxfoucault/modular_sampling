import numpy as np


def sample_correlated_noise(d, h, alpha):
    """
    Generates joint Gaussian noise vectors (W1, W2, W3) based on
    the Brownian motion simulation
    """
    # Covariance for (G1, H1)
    c_G1G1, c_H1H1 = 0.25 * (np.exp(4 * alpha * h) - 1), alpha * h
    c_G1H1 = 0.5 * (np.exp(2 * alpha * h) - 1)

    # Covariance for (G2, H2)
    c_G2G2, c_H2H2 = 0.25 * (np.exp(4 * h) - np.exp(4 * alpha * h)), (1 - alpha) * h
    c_G2H2 = 0.5 * (np.exp(2 * h) - np.exp(2 * alpha * h))

    def sample_gaussian_pair(c_gg, c_hh, c_gh):
        cov = [[c_gg, c_gh], [c_gh, c_hh]]
        return np.random.multivariate_normal([0, 0], cov, size=d)

    pair1, pair2 = (
        sample_gaussian_pair(c_G1G1, c_H1H1, c_G1H1),
        sample_gaussian_pair(c_G2G2, c_H2H2, c_G2H2),
    )
    G1, H1, G2, H2 = pair1[:, 0], pair1[:, 1], pair2[:, 0], pair2[:, 1]

    # Reconstruct W vectors
    W1 = H1 - np.exp(-2 * alpha * h) * G1
    W2 = (H1 + H2) - np.exp(-2 * h) * (G1 + G2)
    W3 = np.exp(-2 * h) * (G1 + G2)
    return W1, W2, W3


def randomized_midpoint_sampler(score_fn, x_init, num_steps, h=0.05, u=1.0):
    """
    Black-box SLC sampler using Algorithm 1 (Randomized Midpoint)
    """
    d = len(x_init)
    x_n, v_n = x_init, np.zeros(d)

    for _ in range(num_steps):
        alpha = np.random.uniform(0, 1)  #
        W1, W2, W3 = sample_correlated_noise(d, h, alpha)

        # Intermediate position x_{n+1/2}
        x_half = (
            x_n
            + 0.5 * (1 - np.exp(-2 * alpha * h)) * v_n
            + 0.5 * u * (alpha * h - 0.5 * (1 - np.exp(-2 * alpha * h))) * score_fn(x_n)
            + np.sqrt(u) * W1
        )

        # Final update x_{n+1} and v_{n+1}
        x_next = (
            x_n
            + 0.5 * (1 - np.exp(-2 * h)) * v_n
            + 0.5 * u * h * (1 - np.exp(-2 * (h - alpha * h))) * score_fn(x_half)
            + np.sqrt(u) * W2
        )
        v_next = (
            v_n * np.exp(-2 * h)
            + u * h * np.exp(-2 * (h - alpha * h)) * score_fn(x_half)
            + 2 * np.sqrt(u) * W3
        )
        x_n, v_n = x_next, v_next
    return x_n


def modular_sampling(target_score_fn, d, kappa, epsilon):
    """
    Modular reduction to SLC sub-problems (Theorem 1 & 2)[cite: 35, 114].
    """
    # Trajectory length K ≈ 1 + log2(kappa)
    K = int(np.ceil(1 + np.log2(max(2, kappa))))
    s_k = 1.0 / (K + 1)  # Error budget sequence

    # 1. Terminal Stage (Property T): Well-conditioned marginal pK
    y_current = np.random.normal(0, 1, size=(d,))
    M_k = kappa  # Initial condition number for the recursion

    # 2. Backward Path (Property B): Solving SLC sub-problems
    for k in reversed(range(K)):
        # Calculate adaptive stepsize a_k for this round
        # Chosen such that a_k^2 = Mk / (1 + Mk) ensures condition number <= 2
        a_k_sq = M_k / (1 + M_k)
        a_k = np.sqrt(a_k_sq)
        b_sq = 1 - a_k_sq

        def backward_conditional_score(u):
            # Tweedie-based conditional score
            return target_score_fn(u) - (a_k / b_sq) * (a_k * u - y_current)

        # Solving the sub-problem using the high-accuracy root-dimension sampler
        y_current = randomized_midpoint_sampler(
            backward_conditional_score,
            y_current,
            num_steps=int(np.sqrt(d) * np.log(1 / (s_k * epsilon)) ** 3),
            h=0.05,
            u=1.0,  # Parameters for O(sqrt(d)) complexity
        )

        # Update Mk for the next stage in the backward sequence
        M_k = 1 + 0.5 * (M_k - 1)

    return y_current


# Example Execution
final_sample = modular_sampling(lambda x: -x, d=10, kappa=10, epsilon=1e-3)
print(f"Final sample: {final_sample}")
