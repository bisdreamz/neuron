package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Fused exponential moving average update with gradient squaring.
 * Combines: state = β * state + (1 - β) * gradient²
 * 
 * This is a common pattern in optimizers like RMSprop, Adam, and AdaGrad
 * that track the second moment (variance) of gradients.
 * 
 * Fusing the squaring operation with the EMA update saves a memory pass.
 */
public final class FusedEMASquared {
    
    public interface Impl {
        void compute(float[] state, float[] gradients, float beta);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] state, float[] gradients, float beta) {
            float oneMinusBeta = 1.0f - beta;
            
            for (int i = 0; i < state.length; i++) {
                float gradSquared = gradients[i] * gradients[i];
                state[i] = beta * state[i] + oneMinusBeta * gradSquared;
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.FusedEMASquaredVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    private FusedEMASquared() {}
    
    /**
     * Update state array with exponential moving average of squared gradients.
     * state[i] = β * state[i] + (1 - β) * gradient[i]²
     * 
     * @param state state array to update (modified in-place)
     * @param gradients gradient array
     * @param beta decay rate (0 < beta < 1)
     * @throws IllegalArgumentException if arrays have different lengths or beta is invalid
     */
    public static void compute(float[] state, float[] gradients, float beta) {
        if (state.length != gradients.length) {
            throw new IllegalArgumentException("Arrays must have same length: " +
                                             "state.length=" + state.length + 
                                             ", gradients.length=" + gradients.length);
        }
        if (beta < 0 || beta >= 1) {
            throw new IllegalArgumentException("Beta must be in [0, 1): " + beta);
        }
        
        IMPL.compute(state, gradients, beta);
    }
    
    public static void computeVectorized(float[] state, float[] gradients, float beta) {
        IMPL.compute(state, gradients, beta);
    }
    
    public static void computeScalar(float[] state, float[] gradients, float beta) {
        new ScalarImpl().compute(state, gradients, beta);
    }
}