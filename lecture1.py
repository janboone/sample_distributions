import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import inspect

st.title("Sampling Distributions Demo")

tab1, tab2 = st.tabs(["Sample Mean", "OLS Slope (Bootstrapping)"])

with tab1:
    st.markdown(
        """
        This tab demonstrates the **Central Limit Theorem** by simulating the distribution of sample means.
        You can choose the distribution below, set its parameters, and click 'Calculate now' to see the distribution of the sample means.
        """
    )

    # Distribution choice
    distribution = st.selectbox(
        "Choose a distribution",
        ("Normal", "Uniform", "Exponential"),
        key="dist_select"
    )

    # Parameters per distribution
    if distribution == "Normal":
        mu = st.slider("Mean (mu)", -10.0, 10.0, 0.0, key="mu")
        sigma = st.slider("Standard deviation (sigma)", 0.1, 10.0, 1.0, key="sigma")
    elif distribution == "Uniform":
        low = st.slider("Minimum", -10.0, 10.0, 0.0, key="low")
        high = st.slider("Maximum", -10.0, 10.0, 1.0, key="high")
        if high <= low:
            st.warning("Maximum must be greater than minimum!")
    elif distribution == "Exponential":
        scale = st.slider("Scale (1/lambda)", 0.1, 10.0, 1.0, key="scale")

    num_experiments = st.slider('Number of experiments (simulations)', 100, 10000, 1000, key="num_exp")
    sample_size = st.slider('Sample size (n)', 1, 100, 9, key="sample_size")

    calculate = st.button("Calculate now", key="calc_mean")

    if calculate:
        st.markdown(
            """
            For each experiment, a sample is drawn and its mean is calculated.
            The histogram below shows the distribution of these sample means.
            """
        )
        # Simulate samples
        if distribution == "Normal":
            samples = np.random.normal(loc=mu, scale=sigma, size=(num_experiments, sample_size))
        elif distribution == "Uniform":
            samples = np.random.uniform(low=low, high=high, size=(num_experiments, sample_size))
        elif distribution == "Exponential":
            samples = np.random.exponential(scale=scale, size=(num_experiments, sample_size))
        else:
            st.error("Unknown distribution!")
            st.stop()

        sample_means = samples.mean(axis=1)

        # Theoretical parameters for the normal approximation
        if distribution == "Normal":
            mean_of_means = mu
            std_of_means = sigma / np.sqrt(sample_size)
        elif distribution == "Uniform":
            mean_of_means = (low + high) / 2
            std_of_means = (high - low) / (np.sqrt(12 * sample_size))
        elif distribution == "Exponential":
            mean_of_means = scale
            std_of_means = scale / np.sqrt(sample_size)

        x = np.linspace(sample_means.min(), sample_means.max(), 100)
        pdf = stats.norm.pdf(x, loc=mean_of_means, scale=std_of_means)

        # Plot
        fig, ax = plt.subplots()
        ax.hist(sample_means, bins=30, density=True, alpha=0.6, color='g', edgecolor='black', label='Sample Means')
        ax.plot(x, pdf, 'r-', lw=2, label='Normal approximation')
        ax.set_title(f'Histogram of sample means ({distribution})')
        ax.set_xlabel('Sample mean')
        ax.set_ylabel('Density')
        ax.legend()
        st.pyplot(fig)

        st.markdown(
            """
            As the sample size increases, the distribution of the sample means becomes more and more normally distributed,
            regardless of the original distribution. This is the Central Limit Theorem!
            """
        )

with tab2:
    st.markdown(
        """
        This tab demonstrates the **sampling distribution of the OLS regression slope** using bootstrapping.
        You can set the sample size and the true slope value for the linear model $y = a + b x + \\epsilon$.
        Bootstrapping means we repeatedly sample (with replacement) from the data and recalculate the slope each time.
        """
    )

    # Sliders for regression
    n_reg = st.slider("Sample size", 10, 200, 50, key="reg_n")
    true_slope = st.slider("True slope (b)", -5.0, 5.0, 2.0, key="reg_slope")
    true_intercept = st.slider("Intercept (a)", -10.0, 10.0, 0.0, key="reg_intercept")
    noise_std = st.slider("Noise std. deviation", 0.1, 10.0, 1.0, key="reg_noise")
    n_boot = st.slider("Number of bootstrap samples", 100, 5000, 1000, key="reg_boot")

    calc_reg = st.button("Bootstrap now", key="reg_boot_btn")

    if calc_reg:
        # Simuleer data
        rng = np.random.default_rng()
        x = np.linspace(0, 10, n_reg)
        y = true_intercept + true_slope * x + rng.normal(0, noise_std, n_reg)

        # Bootstrapping
        slopes = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_reg, n_reg)  # sample with replacement
            x_boot = x[idx]
            y_boot = y[idx]
            # OLS slope calculation
            b_boot = np.cov(x_boot, y_boot, bias=True)[0, 1] / np.var(x_boot)
            slopes.append(b_boot)
        slopes = np.array(slopes)

        # Plot
        fig2, ax2 = plt.subplots()
        ax2.hist(slopes, bins=30, density=True, alpha=0.7, color='b', edgecolor='black')
        ax2.axvline(true_slope, color='r', linestyle='--', label='True slope')
        ax2.set_title("Bootstrap distribution of OLS slope")
        ax2.set_xlabel("Slope")
        ax2.set_ylabel("Density")
        ax2.legend()
        st.pyplot(fig2)

        st.markdown(
            """
            **Bootstrapping** is a resampling method: each new sample is drawn *with replacement* from the original data.
            This allows us to estimate the sampling distribution of the OLS slope, even if we don't know the true population distribution.
            """
        )
