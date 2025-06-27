package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Element-wise scaling operation: output[i] = scale * input[i]
 * Used in Adam optimizer for bias correction and learning rate scaling.
 */
public final class ElementwiseScale {
    
    public interface Impl {
        void compute(float[] input, float scale, float[] output);
        void computeInPlace(float[] array, float scale);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] input, float scale, float[] output) {
            for (int i = 0; i < input.length; i++) {
                output[i] = scale * input[i];
            }
        }
        
        @Override
        public void computeInPlace(float[] array, float scale) {
            for (int i = 0; i < array.length; i++) {
                array[i] = scale * array[i];
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.ElementwiseScaleVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Scale input array by a scalar value.
     * 
     * @param input input array
     * @param scale scalar multiplier
     * @param output output array (must be same length as input)
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public static void compute(float[] input, float scale, float[] output) {
        if (input.length != output.length)
            throw new IllegalArgumentException("Input and output arrays must have same length: " +
                                             "input.length=" + input.length + ", output.length=" + output.length);
        
        IMPL.compute(input, scale, output);
    }
    
    /**
     * Scale array in-place: array[i] = scale * array[i]
     * More memory efficient when input and output can be the same array.
     * 
     * @param array array to scale in-place
     * @param scale scalar multiplier
     */
    public static void computeInPlace(float[] array, float scale) {
        IMPL.computeInPlace(array, scale);
    }
    
    static void computeVectorized(float[] input, float scale, float[] output) {
        IMPL.compute(input, scale, output);
    }
    
    static void computeVectorizedInPlace(float[] array, float scale) {
        IMPL.computeInPlace(array, scale);
    }
    
    static void computeScalar(float[] input, float scale, float[] output) {
        new ScalarImpl().compute(input, scale, output);
    }
    
    static void computeScalarInPlace(float[] array, float scale) {
        new ScalarImpl().computeInPlace(array, scale);
    }
    
    private ElementwiseScale() {}
}