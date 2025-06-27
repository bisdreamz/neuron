package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Fused multiply-divide-subtract operation: params[i] -= scale * (numerator[i] / denominator[i])
 * 
 * This is the core Adam parameter update operation:
 * params[i] -= learningRate * biascorrectedMomentum[i] / sqrtVelocity[i]
 * 
 * Combines three operations into one vectorized call for maximum performance.
 */
public final class FusedMultiplyDivideSubtract {
    
    public interface Impl {
        void compute(float[] params, float[] numerator, float[] denominator, float scale);
        void computeAdd(float[] params, float[] numerator, float[] denominator, float scale);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] params, float[] numerator, float[] denominator, float scale) {
            for (int i = 0; i < params.length; i++) {
                params[i] -= scale * (numerator[i] / denominator[i]);
            }
        }
        
        @Override
        public void computeAdd(float[] params, float[] numerator, float[] denominator, float scale) {
            for (int i = 0; i < params.length; i++) {
                params[i] += scale * (numerator[i] / denominator[i]);
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.FusedMultiplyDivideSubtractVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute fused multiply-divide-subtract in-place.
     * Updates params array with: params[i] -= scale * (numerator[i] / denominator[i])
     * 
     * @param params parameter array to update (modified in-place)
     * @param numerator numerator for division
     * @param denominator denominator for division
     * @param scale scaling factor (typically learning rate)
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public static void compute(float[] params, float[] numerator, float[] denominator, float scale) {
        if (params.length != numerator.length || params.length != denominator.length)
            throw new IllegalArgumentException("All arrays must have same length: " +
                                             "params.length=" + params.length + 
                                             ", numerator.length=" + numerator.length +
                                             ", denominator.length=" + denominator.length);
        
        IMPL.compute(params, numerator, denominator, scale);
    }
    
    /**
     * Compute fused multiply-divide-add in-place (for operations that add instead of subtract).
     * Updates params array with: params[i] += scale * (numerator[i] / denominator[i])
     */
    public static void computeAdd(float[] params, float[] numerator, float[] denominator, float scale) {
        if (params.length != numerator.length || params.length != denominator.length)
            throw new IllegalArgumentException("All arrays must have same length");
        
        IMPL.computeAdd(params, numerator, denominator, scale);
    }
    
    static void computeVectorized(float[] params, float[] numerator, float[] denominator, float scale) {
        IMPL.compute(params, numerator, denominator, scale);
    }
    
    static void computeVectorizedAdd(float[] params, float[] numerator, float[] denominator, float scale) {
        IMPL.computeAdd(params, numerator, denominator, scale);
    }
    
    static void computeScalar(float[] params, float[] numerator, float[] denominator, float scale) {
        new ScalarImpl().compute(params, numerator, denominator, scale);
    }
    
    static void computeScalarAdd(float[] params, float[] numerator, float[] denominator, float scale) {
        new ScalarImpl().computeAdd(params, numerator, denominator, scale);
    }
    
    private FusedMultiplyDivideSubtract() {}
}