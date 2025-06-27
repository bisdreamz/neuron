package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Element-wise square root operation: output[i] = sqrt(input[i])
 * Used in Adam optimizer for computing sqrt(v_t) + epsilon in the denominator.
 */
public final class ElementwiseSqrt {
    
    public interface Impl {
        void compute(float[] input, float[] output);
        void computeWithEpsilon(float[] input, float epsilon, float[] output);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] input, float[] output) {
            for (int i = 0; i < input.length; i++) {
                output[i] = (float) Math.sqrt(input[i]);
            }
        }
        
        @Override
        public void computeWithEpsilon(float[] input, float epsilon, float[] output) {
            for (int i = 0; i < input.length; i++) {
                output[i] = (float) Math.sqrt(input[i] + epsilon);
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.ElementwiseSqrtVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute element-wise square root of input array.
     * 
     * @param input input array (all values must be non-negative)
     * @param output output array (must be same length as input)
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public static void compute(float[] input, float[] output) {
        if (input.length != output.length)
            throw new IllegalArgumentException("Input and output arrays must have same length: " +
                                             "input.length=" + input.length + ", output.length=" + output.length);
        
        IMPL.compute(input, output);
    }
    
    /**
     * Compute element-wise square root with epsilon: output[i] = sqrt(input[i] + epsilon)
     * This is the common pattern in Adam optimizer to avoid division by zero.
     * 
     * @param input input array
     * @param epsilon small constant to add before sqrt (typically 1e-8)
     * @param output output array (must be same length as input)
     */
    public static void computeWithEpsilon(float[] input, float epsilon, float[] output) {
        if (input.length != output.length)
            throw new IllegalArgumentException("Input and output arrays must have same length");
        
        IMPL.computeWithEpsilon(input, epsilon, output);
    }
    
    static void computeVectorized(float[] input, float[] output) {
        IMPL.compute(input, output);
    }
    
    static void computeVectorizedWithEpsilon(float[] input, float epsilon, float[] output) {
        IMPL.computeWithEpsilon(input, epsilon, output);
    }
    
    static void computeScalar(float[] input, float[] output) {
        new ScalarImpl().compute(input, output);
    }
    
    static void computeScalarWithEpsilon(float[] input, float epsilon, float[] output) {
        new ScalarImpl().computeWithEpsilon(input, epsilon, output);
    }
    
    private ElementwiseSqrt() {}
}