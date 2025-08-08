import numpy as np
import jax.numpy as jnp
import scipy
import time

def main():
    matrix_size = 100
    num_trials = 100

    np_times = []
    scipy_times = []
    jnp_times = []

    for _ in range(num_trials):
        # Generate a random Hermitian matrix
        A = np.random.rand(matrix_size, matrix_size)
        A = A + A.T  # Make it Hermitian

        # Time numpy.linalg.eigh
        start_time = time.time()
        np.linalg.eigh(A)
        np_times.append(time.time() - start_time)

        start_time = time.time()
        jnp.linalg.eigh(A)
        jnp_times.append(time.time() - start_time)

        # Time scipy.linalg.eigh
        start_time = time.time()
        scipy.linalg.eigh(A)
        scipy_times.append(time.time() - start_time)

    avg_np_time = np.mean(np_times)
    avg_jnp_time = np.mean(jnp_times)
    avg_scipy_time = np.mean(scipy_times)

    print(f"Average time for np.linalg.eigh: {avg_np_time:.6f} seconds")
    print(f"Average time for jnp.linalg.eigh: {avg_jnp_time:.6f} seconds")
    print(f"Average time for scipy.linalg.eigh: {avg_scipy_time:.6f} seconds")


if __name__ == "__main__":
    main()
