package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Fused bias correction with scaling operation.
 * Combines: output[i] = scale * (input[i] / correction)
 * 
 * This is used in bias-corrected optimizers like Adam to combine
 * the bias correction division with scaling operations.
 */
public final class FusedBiasCorrectScale {
    
    public interface Impl {
        void compute(float[] input, float scale, float correction, float[] output);
        void computeInPlace(float[] array, float scale, float correction);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] input, float scale, float correction, float[] output) {
            float scaleDivCorrection = scale / correction;
            
            for (int i = 0; i < input.length; i++) {
                output[i] = input[i] * scaleDivCorrection;
            }
        }
        
        @Override
        public void computeInPlace(float[] array, float scale, float correction) {
            float scaleDivCorrection = scale / correction;
            
            for (int i = 0; i < array.length; i++) {
                array[i] *= scaleDivCorrection;
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.FusedBiasCorrectScaleVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    private FusedBiasCorrectScale() {}
    
    /**
     * Compute bias-corrected scaled values.
     * output[i] = scale * (input[i] / correction)
     * 
     * @param input input array
     * @param scale scaling factor
     * @param correction bias correction factor
     * @param output output array (can be same as input for in-place operation)
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public static void compute(float[] input, float scale, float correction, float[] output) {
        if (input.length != output.length) {
            throw new IllegalArgumentException("Arrays must have same length: " +
                                             "input.length=" + input.length + 
                                             ", output.length=" + output.length);
        }
        
        IMPL.compute(input, scale, correction, output);
    }
    
    /**
     * Compute bias-corrected scaled values in-place.
     * array[i] = scale * (array[i] / correction)
     * 
     * @param array array to update in-place
     * @param scale scaling factor
     * @param correction bias correction factor
     */
    public static void computeInPlace(float[] array, float scale, float correction) {
        IMPL.computeInPlace(array, scale, correction);
    }
    
    public static void computeVectorized(float[] input, float scale, float correction, float[] output) {
        IMPL.compute(input, scale, correction, output);
    }
    
    public static void computeScalar(float[] input, float scale, float correction, float[] output) {
        new ScalarImpl().compute(input, scale, correction, output);
    }
}