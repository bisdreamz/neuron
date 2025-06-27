package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Exponential moving average: output[i] = decay * current[i] + (1 - decay) * newValue[i]
 * Core operation for Adam optimizer momentum and velocity updates.
 * 
 * This is the fundamental EMA formula used in:
 * - Adam momentum: m_t = β₁ * m_{t-1} + (1 - β₁) * gradient
 * - Adam velocity: v_t = β₂ * v_{t-1} + (1 - β₂) * gradient²
 */
public final class ExponentialMovingAverage {
    
    public interface Impl {
        void computeInPlace(float[] current, float[] newValues, float decay);
        void compute(float[] current, float[] newValues, float decay, float[] output);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void computeInPlace(float[] current, float[] newValues, float decay) {
            float oneMinusDecay = 1.0f - decay;
            for (int i = 0; i < current.length; i++) {
                current[i] = decay * current[i] + oneMinusDecay * newValues[i];
            }
        }
        
        @Override
        public void compute(float[] current, float[] newValues, float decay, float[] output) {
            float oneMinusDecay = 1.0f - decay;
            for (int i = 0; i < current.length; i++) {
                output[i] = decay * current[i] + oneMinusDecay * newValues[i];
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.ExponentialMovingAverageVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute exponential moving average in-place.
     * Updates current array with EMA of current and new values.
     * 
     * @param current array to update (modified in-place)
     * @param newValues new values to incorporate
     * @param decay decay factor (0.9 for momentum, 0.999 for velocity)
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public static void computeInPlace(float[] current, float[] newValues, float decay) {
        if (current.length != newValues.length)
            throw new IllegalArgumentException("Arrays must have same length: " +
                                             "current.length=" + current.length + ", newValues.length=" + newValues.length);
        
        IMPL.computeInPlace(current, newValues, decay);
    }
    
    /**
     * Compute exponential moving average to output array.
     * 
     * @param current current values
     * @param newValues new values to incorporate  
     * @param decay decay factor
     * @param output output array (must be same length as inputs)
     */
    public static void compute(float[] current, float[] newValues, float decay, float[] output) {
        if (current.length != newValues.length || current.length != output.length)
            throw new IllegalArgumentException("All arrays must have same length");
        
        IMPL.compute(current, newValues, decay, output);
    }
    
    static void computeVectorizedInPlace(float[] current, float[] newValues, float decay) {
        IMPL.computeInPlace(current, newValues, decay);
    }
    
    static void computeVectorized(float[] current, float[] newValues, float decay, float[] output) {
        IMPL.compute(current, newValues, decay, output);
    }
    
    static void computeScalarInPlace(float[] current, float[] newValues, float decay) {
        new ScalarImpl().computeInPlace(current, newValues, decay);
    }
    
    static void computeScalar(float[] current, float[] newValues, float decay, float[] output) {
        new ScalarImpl().compute(current, newValues, decay, output);
    }
    
    private ExponentialMovingAverage() {}
}