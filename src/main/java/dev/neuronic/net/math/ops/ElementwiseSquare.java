package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Element-wise square operation: output[i] = input[i] * input[i]
 * Essential for Adam optimizer gradient squared computation.
 */
public final class ElementwiseSquare {
    
    public interface Impl {
        void compute(float[] input, float[] output);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] input, float[] output) {
            for (int i = 0; i < input.length; i++) {
                output[i] = input[i] * input[i];
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.ElementwiseSquareVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute element-wise square of input array.
     * 
     * @param input input array
     * @param output output array (must be same length as input)
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public static void compute(float[] input, float[] output) {
        if (input.length != output.length)
            throw new IllegalArgumentException("Input and output arrays must have same length: " +
                                             "input.length=" + input.length + ", output.length=" + output.length);
        
        IMPL.compute(input, output);
    }
    
    static void computeVectorized(float[] input, float[] output) {
        IMPL.compute(input, output);
    }
    
    static void computeScalar(float[] input, float[] output) {
        new ScalarImpl().compute(input, output);
    }
    
    private ElementwiseSquare() {}
}