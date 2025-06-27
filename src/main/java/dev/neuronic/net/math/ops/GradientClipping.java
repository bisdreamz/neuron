package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Gradient clipping operations for preventing gradient explosion during training.
 * Supports both norm-based and value-based clipping with vectorization.
 */
public final class GradientClipping {
    
    public interface Impl {
        void clipByValue(float[] array, float maxValue);
        void scaleInPlace(float[] array, float scale);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void clipByValue(float[] array, float maxValue) {
            for (int i = 0; i < array.length; i++) {
                array[i] = Math.max(-maxValue, Math.min(maxValue, array[i]));
            }
        }
        
        @Override
        public void scaleInPlace(float[] array, float scale) {
            for (int i = 0; i < array.length; i++) {
                array[i] *= scale;
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.GradientClippingVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Clip gradients by L2 norm across multiple weight matrices and bias vectors.
     * If the total norm exceeds maxNorm, all gradients are scaled down proportionally.
     * 
     * @param weights array of weight gradient matrices (can contain nulls)
     * @param biases array of bias gradient vectors (can contain nulls)
     * @param maxNorm maximum allowed L2 norm
     * @return the scale factor applied (1.0 if no clipping, < 1.0 if clipped)
     */
    public static float clipByNorm(float[][][] weights, float[][] biases, float maxNorm) {
        if (maxNorm <= 0) {
            return 1.0f; // No clipping
        }
        
        // Compute current norm
        float norm = GradientNorm.computeNorm(weights, biases);
        
        if (norm <= maxNorm) {
            return 1.0f; // No clipping needed
        }
        
        // Compute scale factor
        float scale = maxNorm / norm;
        
        // Apply scaling to all gradients
        applyScale(weights, biases, scale);
        
        return scale;
    }
    
    /**
     * Clip gradient values to be within [-maxValue, maxValue].
     * 
     * @param array gradient array to clip in place
     * @param maxValue maximum absolute value allowed
     */
    public static void clipByValue(float[] array, float maxValue) {
        if (maxValue <= 0 || array == null) {
            return;
        }
        
        IMPL.clipByValue(array, maxValue);
    }
    
    /**
     * Clip gradient values in a 2D array to be within [-maxValue, maxValue].
     */
    public static void clipByValue(float[][] matrix, float maxValue) {
        if (matrix == null) return;
        
        for (float[] row : matrix) {
            clipByValue(row, maxValue);
        }
    }
    
    /**
     * Scale a single array by a factor.
     */
    public static void scaleInPlace(float[] array, float scale) {
        if (array == null || scale == 1.0f) return;
        
        IMPL.scaleInPlace(array, scale);
    }
    
    /**
     * Scale a 2D array (matrix) by a factor.
     */
    public static void scaleInPlace(float[][] matrix, float scale) {
        if (matrix == null || scale == 1.0f) return;
        
        for (float[] row : matrix) {
            scaleInPlace(row, scale);
        }
    }
    
    // Apply scale to all gradients
    private static void applyScale(float[][][] weights, float[][] biases, float scale) {
        // Scale weights
        if (weights != null) {
            for (float[][] weightMatrix : weights) {
                if (weightMatrix != null) {
                    scaleInPlace(weightMatrix, scale);
                }
            }
        }
        
        // Scale biases
        if (biases != null) {
            for (float[] biasVector : biases) {
                if (biasVector != null) {
                    scaleInPlace(biasVector, scale);
                }
            }
        }
    }
}