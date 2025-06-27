package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Computes L2 norm of gradients efficiently with vectorization support.
 * Used for gradient clipping and monitoring gradient magnitudes.
 */
public final class GradientNorm {
    
    public interface Impl {
        float computeNormSquared(float[] array);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public float computeNormSquared(float[] array) {
            float sum = 0.0f;
            for (float val : array) {
                sum += val * val;
            }
            return sum;
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.GradientNormVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute L2 norm squared of a single array.
     */
    public static float computeNormSquared(float[] array) {
        return IMPL.computeNormSquared(array);
    }
    
    /**
     * Compute L2 norm squared of a 2D array (matrix).
     */
    public static float computeNormSquared(float[][] matrix) {
        float normSquared = 0.0f;
        for (float[] row : matrix) {
            normSquared += computeNormSquared(row);
        }
        return normSquared;
    }
    
    /**
     * Compute L2 norm squared across multiple weight matrices and bias vectors.
     * 
     * @param weights array of weight matrices (can contain nulls)
     * @param biases array of bias vectors (can contain nulls)
     * @return total L2 norm squared
     */
    public static float computeNormSquared(float[][][] weights, float[][] biases) {
        float normSquared = 0.0f;
        
        // Sum weight norms
        if (weights != null) {
            for (float[][] weightMatrix : weights) {
                if (weightMatrix != null) {
                    normSquared += computeNormSquared(weightMatrix);
                }
            }
        }
        
        // Sum bias norms
        if (biases != null) {
            for (float[] biasVector : biases) {
                if (biasVector != null) {
                    normSquared += computeNormSquared(biasVector);
                }
            }
        }
        
        return normSquared;
    }
    
    /**
     * Compute L2 norm (not squared) of a single array.
     */
    public static float computeNorm(float[] array) {
        return (float) Math.sqrt(computeNormSquared(array));
    }
    
    /**
     * Compute L2 norm (not squared) of a 2D array.
     */
    public static float computeNorm(float[][] matrix) {
        return (float) Math.sqrt(computeNormSquared(matrix));
    }
    
    /**
     * Compute L2 norm (not squared) across multiple weight matrices and bias vectors.
     */
    public static float computeNorm(float[][][] weights, float[][] biases) {
        return (float) Math.sqrt(computeNormSquared(weights, biases));
    }
}