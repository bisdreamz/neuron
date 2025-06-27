package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Element-wise subtraction: output[i] = a[i] - b[i]
 */
public final class ElementwiseSubtract {
    
    public interface Impl {
        void compute(float[] a, float[] b, float[] output);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] a, float[] b, float[] output) {
            for (int i = 0; i < a.length; i++) {
                output[i] = a[i] - b[i];
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.ElementwiseSubtractVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute element-wise subtraction of two arrays.
     * 
     * @param a first input array (minuend)
     * @param b second input array (subtrahend)
     * @param output pre-allocated output buffer
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public static void compute(float[] a, float[] b, float[] output) {
        if (a.length != b.length || a.length != output.length)
            throw new IllegalArgumentException("All arrays must have same length: a=" + a.length + 
                                             ", b=" + b.length + ", output=" + output.length);
        
        IMPL.compute(a, b, output);
    }
    
    static void computeVectorized(float[] a, float[] b, float[] output) {
        IMPL.compute(a, b, output);
    }
    
    static void computeScalar(float[] a, float[] b, float[] output) {
        new ScalarImpl().compute(a, b, output);
    }
    
    private ElementwiseSubtract() {}
}