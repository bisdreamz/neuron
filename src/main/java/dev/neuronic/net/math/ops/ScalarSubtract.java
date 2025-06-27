package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Scalar subtraction: output[i] = scalar - array[i]
 * Useful for operations like (1.0 - sigmoid_output) in neural networks.
 */
public final class ScalarSubtract {
    
    public interface Impl {
        void compute(float scalar, float[] array, float[] output);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float scalar, float[] array, float[] output) {
            for (int i = 0; i < array.length; i++) {
                output[i] = scalar - array[i];
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.ScalarSubtractVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute scalar minus array: output[i] = scalar - array[i]
     * 
     * @param scalar the scalar value to subtract from
     * @param array the array to subtract
     * @param output pre-allocated output buffer
     * @throws IllegalArgumentException if array and output have different lengths
     */
    public static void compute(float scalar, float[] array, float[] output) {
        if (array.length != output.length)
            throw new IllegalArgumentException("Array and output must have same length: array=" + array.length + 
                                             ", output=" + output.length);
        
        IMPL.compute(scalar, array, output);
    }
    
    static void computeVectorized(float scalar, float[] array, float[] output) {
        IMPL.compute(scalar, array, output);
    }
    
    static void computeScalar(float scalar, float[] array, float[] output) {
        new ScalarImpl().compute(scalar, array, output);
    }
    
    private ScalarSubtract() {}
}