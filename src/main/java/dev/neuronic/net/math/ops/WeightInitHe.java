package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.math.Vectorization;

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
        void compute(float[][] weights, int fanIn, FastRandom random);
        void compute(float[][] weights, int fanIn, float noiseLevel, FastRandom random);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[][] weights, int fanIn, FastRandom random) {
            compute(weights, fanIn, 0.0f, random);
        }

        @Override
        public void compute(float[][] weights, int fanIn, float noiseLevel, FastRandom random) {
            if (fanIn <= 0)
                throw new IllegalArgumentException("fanIn must be positive, got: " + fanIn);
            
            float scale = (float) Math.sqrt(2.0 / fanIn);
            
            for (float[] row : weights) {
                for (int i = 0; i < row.length; i++) {
                    row[i] = (float)(random.nextGaussian() * scale) + (random.nextFloat() - 0.5f) * noiseLevel;
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
     * @param random random number generator
     * @throws IllegalArgumentException if fanIn <= 0
     */
    public static void compute(float[][] weights, int fanIn, FastRandom random) {
        IMPL.compute(weights, fanIn, random);
    }

    /**
     * Initialize weights using He initialization with added uniform noise.
     *
     * @param weights 2D weight matrix to initialize
     * @param fanIn number of inputs (must be > 0)
     * @param noiseLevel the level of uniform noise to add
     * @param random random number generator
     * @throws IllegalArgumentException if fanIn <= 0
     */
    public static void compute(float[][] weights, int fanIn, float noiseLevel, FastRandom random) {
        IMPL.compute(weights, fanIn, noiseLevel, random);
    }
}