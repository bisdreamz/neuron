package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Weight decay operation: weights[i] = weights[i] * (1 - decay_rate)
 * 
 * <p><b>What is weight decay?</b> Weight decay is a regularization technique that prevents
 * neural networks from overfitting by gradually shrinking weights toward zero. This encourages
 * the network to use simpler patterns and generalize better to new data.
 * 
 * <p><b>AdamW vs L2 regularization:</b> AdamW uses "decoupled weight decay" which applies
 * decay directly to weights, while traditional L2 adds penalty to gradients. AdamW is
 * more effective and predictable than L2 regularization with adaptive optimizers.
 * 
 * <p><b>Common decay rates:</b>
 * <ul>
 *   <li><b>0.01</b> - Light regularization for large models</li>
 *   <li><b>0.1</b> - Moderate regularization for medium models</li>
 *   <li><b>0.3</b> - Strong regularization for small models or overfitting problems</li>
 * </ul>
 */
public final class WeightDecay {
    
    public interface Impl {
        void compute(float[] weights, float keepRate);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] weights, float keepRate) {
            for (int i = 0; i < weights.length; i++) {
                weights[i] *= keepRate;
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.WeightDecayVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Apply weight decay to parameter array.
     * 
     * <p><b>Formula:</b> weights[i] = weights[i] * (1 - decay_rate)
     * 
     * <p><b>Why this works:</b> By multiplying weights by a factor slightly less than 1,
     * we gradually shrink them toward zero. This prevents any single weight from becoming
     * too large and dominating the network's decisions.
     * 
     * @param weights weight array to decay in-place
     * @param decayRate decay rate (typically 0.01-0.3, higher = more decay)
     * @throws IllegalArgumentException if decay rate is negative or >= 1
     */
    public static void compute(float[] weights, float decayRate) {
        if (decayRate < 0 || decayRate >= 1)
            throw new IllegalArgumentException("Decay rate must be in [0, 1): " + decayRate);
        
        if (decayRate == 0) return; // No decay needed
        
        float keepRate = 1.0f - decayRate;
        
        IMPL.compute(weights, keepRate);
    }
    
    /**
     * Apply weight decay to 2D weight matrix.
     * Commonly used for layer weights that are stored as [input][neuron] arrays.
     * 
     * @param weights 2D weight matrix to decay in-place
     * @param decayRate decay rate (typically 0.01-0.3)
     */
    public static void compute(float[][] weights, float decayRate) {
        if (decayRate < 0 || decayRate >= 1)
            throw new IllegalArgumentException("Decay rate must be in [0, 1): " + decayRate);
        
        if (decayRate == 0) return; // No decay needed
        
        for (float[] row : weights) {
            compute(row, decayRate);
        }
    }
    
    static void computeVectorized(float[] weights, float keepRate) {
        IMPL.compute(weights, keepRate);
    }
    
    static void computeScalar(float[] weights, float keepRate) {
        new ScalarImpl().compute(weights, keepRate);
    }
    
    private WeightDecay() {}
}