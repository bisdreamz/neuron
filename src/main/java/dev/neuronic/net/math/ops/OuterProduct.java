package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Outer product: output[i][j] = a[i] * b[j]
 */
public final class OuterProduct {
    
    public interface Impl {
        void compute(float[] a, float[] b, float[][] output);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] a, float[] b, float[][] output) {
            for (int i = 0; i < a.length; i++) {
                float ai = a[i];
                float[] outputRow = output[i];
                
                for (int j = 0; j < b.length; j++) {
                    outputRow[j] = ai * b[j];
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
                        "dev.neuronic.net.math.ops.vector.OuterProductVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute outer product of two vectors.
     * 
     * @param a first vector (length m)
     * @param b second vector (length n)
     * @param output pre-allocated m x n matrix
     * @throws IllegalArgumentException if output dimensions don't match input lengths
     */
    public static void compute(float[] a, float[] b, float[][] output) {
        if (output.length != a.length)
            throw new IllegalArgumentException("Output rows must match first vector length: " +
                                             "output.length=" + output.length + ", a.length=" + a.length);
        
        if (output.length > 0 && output[0].length != b.length)
            throw new IllegalArgumentException("Output columns must match second vector length: " +
                                             "output[0].length=" + output[0].length + ", b.length=" + b.length);
        
        IMPL.compute(a, b, output);
    }
    
    static void computeVectorized(float[] a, float[] b, float[][] output) {
        IMPL.compute(a, b, output);
    }
    
    static void computeScalar(float[] a, float[] b, float[][] output) {
        new ScalarImpl().compute(a, b, output);
    }
    
    private OuterProduct() {}
}