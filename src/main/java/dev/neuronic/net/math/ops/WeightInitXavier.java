package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.math.Vectorization;

/**
 * Xavier/Glorot uniform weight initialization - the gold standard for networks with sigmoid/tanh activations.
 * 
 * <p><strong>What it does:</strong>
 * Initializes weights uniformly in the range [-limit, +limit] where limit = sqrt(6 / (fanIn + fanOut)).
 * This specific formula with factor 6 (not 2) is for uniform distribution and is the most common
 * implementation used by major frameworks like PyTorch and TensorFlow.
 * 
 * <p><strong>Why it works:</strong>
 * Maintains equal variance of activations and gradients across all layers by considering both
 * input and output connections. This prevents the vanishing/exploding gradient problem that
 * plagued early deep networks.
 * 
 * <p><strong>When to use:</strong>
 * <ul>
 *   <li>Networks with sigmoid or tanh activations (ideal choice)</li>
 *   <li>GRU, LSTM layers (they use sigmoid/tanh gates internally)</li>
 *   <li>Any layer where ReLU causes training instability</li>
 *   <li>As the default for older network architectures</li>
 * </ul>
 * 
 * <p><strong>Performance notes:</strong>
 * - Fully vectorized using Java Vector API for SIMD acceleration
 * - Uses FastRandom's Xoroshiro128++ for high-quality, fast random numbers
 * - Zero allocation after JIT warm-up
 */
public final class WeightInitXavier {
    
    public interface Impl {
        void compute(float[][] weights, int fanIn, int fanOut, FastRandom random);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[][] weights, int fanIn, int fanOut, FastRandom random) {
            if (fanIn <= 0 || fanOut <= 0)
                throw new IllegalArgumentException("fanIn and fanOut must be positive, got: " + fanIn + ", " + fanOut);
                
            float limit = (float)Math.sqrt(6.0f / (fanIn + fanOut));
            float twoLimit = 2f * limit;
            
            for (float[] row : weights) {
                for (int i = 0; i < row.length; i++) {
                    row[i] = (random.nextFloat() - 0.5f) * twoLimit;
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
                        "dev.neuronic.net.math.ops.vector.WeightInitXavierVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    private WeightInitXavier() {}

    public static void compute(float[][] weights, int fanIn, int fanOut, FastRandom random) {
        IMPL.compute(weights, fanIn, fanOut, random);
    }
}