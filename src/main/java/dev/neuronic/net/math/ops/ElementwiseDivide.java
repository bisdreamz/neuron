package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Element-wise division: output[i] = numerator[i] / denominator[i]
 * Used in Adam optimizer for dividing bias-corrected momentum by sqrt(velocity).
 */
public final class ElementwiseDivide {
    
    public interface Impl {
        void compute(float[] numerator, float[] denominator, float[] output);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] numerator, float[] denominator, float[] output) {
            for (int i = 0; i < numerator.length; i++) {
                output[i] = numerator[i] / denominator[i];
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.ElementwiseDivideVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute element-wise division.
     * 
     * @param numerator numerator array
     * @param denominator denominator array  
     * @param output output array (must be same length as inputs)
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public static void compute(float[] numerator, float[] denominator, float[] output) {
        if (numerator.length != denominator.length || numerator.length != output.length)
            throw new IllegalArgumentException("All arrays must have same length: " +
                                             "numerator.length=" + numerator.length + 
                                             ", denominator.length=" + denominator.length +
                                             ", output.length=" + output.length);
        
        IMPL.compute(numerator, denominator, output);
    }
    
    static void computeVectorized(float[] numerator, float[] denominator, float[] output) {
        IMPL.compute(numerator, denominator, output);
    }
    
    static void computeScalar(float[] numerator, float[] denominator, float[] output) {
        new ScalarImpl().compute(numerator, denominator, output);
    }
    
    private ElementwiseDivide() {}
}