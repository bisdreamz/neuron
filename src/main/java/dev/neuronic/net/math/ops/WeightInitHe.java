package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;
import java.util.Random;

/**
 * He weight initialization - the modern standard for ReLU networks.
 * 
 * <p><strong>What it does:</strong>
 * Initializes weights by sampling from a Gaussian distribution with mean 0 and 
 * standard deviation sqrt(2 / fanIn). The factor of 2 compensates for ReLU zeroing
 * out negative values, which cuts the variance in half.
 * 
 * <p><strong>Why it works:</strong>
 * ReLU activation sets all negative values to zero, effectively killing half the neurons
 * on average. This halves the variance of the output. The factor of 2 in He initialization
 * compensates for this, maintaining proper signal flow through deep networks.
 * 
 * <p><strong>When to use:</strong>
 * <ul>
 *   <li>Networks with ReLU activation (always use this)</li>
 *   <li>Leaky ReLU, PReLU, or other ReLU variants</li>
 *   <li>Most modern deep learning architectures</li>
 *   <li>Default choice unless you specifically need sigmoid/tanh</li>
 * </ul>
 * 
 * <p><strong>Performance notes:</strong>
 * - Fully vectorized using Java Vector API for SIMD acceleration
 * - Uses FastRandom's Xoroshiro128++ for Gaussian sampling
 * - Zero allocation after JIT warm-up
 */
public final class WeightInitHe {
    
    public interface Impl {
        void compute(float[][] weights, int fanIn);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[][] weights, int fanIn) {
            if (fanIn <= 0)
                throw new IllegalArgumentException("fanIn must be positive, got: " + fanIn);
            
            float scale = (float) Math.sqrt(2.0 / fanIn);
            Random rnd = new Random();
            
            for (float[] row : weights) {
                for (int i = 0; i < row.length; i++) {
                    row[i] = (float)(rnd.nextGaussian() * scale);
                }
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.WeightInitHeVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    private WeightInitHe() {}
    
    /**
     * Initialize weights using He initialization.
     * 
     * @param weights 2D weight matrix to initialize
     * @param fanIn number of inputs (must be > 0)
     * @throws IllegalArgumentException if fanIn <= 0
     */
    public static void compute(float[][] weights, int fanIn) {
        IMPL.compute(weights, fanIn);
    }
    
    static void computeVectorized(float[][] weights, float scale) {
        IMPL.compute(weights, (int)(2.0f / (scale * scale)));
    }
    
    static void computeScalar(float[][] weights, float scale) {
        new ScalarImpl().compute(weights, (int)(2.0f / (scale * scale)));
    }
}