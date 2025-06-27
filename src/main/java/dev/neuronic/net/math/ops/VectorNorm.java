package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Vector norm calculations with vectorized implementations.
 * Used primarily for gradient clipping and monitoring.
 */
public final class VectorNorm {
    
    public interface Impl {
        float computeL2Squared(float[] vector);
        float computeL1(float[] vector);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public float computeL2Squared(float[] vector) {
            float sum = 0.0f;
            for (float value : vector) {
                sum += value * value;
            }
            return sum;
        }
        
        @Override
        public float computeL1(float[] vector) {
            float sum = 0.0f;
            for (float value : vector) {
                sum += Math.abs(value);
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
                        "dev.neuronic.net.math.ops.vector.VectorNormVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute L2 norm squared (sum of squares) of a vector.
     * This avoids the sqrt computation when only comparing norms.
     * 
     * @param vector input vector
     * @return sum of squares of all elements
     */
    public static float computeL2Squared(float[] vector) {
        if (vector.length == 0) return 0.0f;
        
        return IMPL.computeL2Squared(vector);
    }
    
    /**
     * Compute L2 norm of a vector.
     * 
     * @param vector input vector
     * @return L2 norm (sqrt of sum of squares)
     */
    public static float computeL2(float[] vector) {
        return (float) Math.sqrt(computeL2Squared(vector));
    }
    
    /**
     * Compute L1 norm (sum of absolute values) of a vector.
     * 
     * @param vector input vector
     * @return L1 norm
     */
    public static float computeL1(float[] vector) {
        if (vector.length == 0) return 0.0f;
        
        return IMPL.computeL1(vector);
    }
    
    /**
     * Compute infinity norm (maximum absolute value) of a vector.
     * 
     * @param vector input vector
     * @return infinity norm
     */
    public static float computeInfinity(float[] vector) {
        if (vector.length == 0) return 0.0f;
        
        float maxAbs = 0.0f;
        for (float value : vector) {
            float abs = Math.abs(value);
            if (abs > maxAbs) {
                maxAbs = abs;
            }
        }
        return maxAbs;
    }
    
    static float computeL2SquaredVectorized(float[] vector) {
        return IMPL.computeL2Squared(vector);
    }
    
    static float computeL2SquaredScalar(float[] vector) {
        return new ScalarImpl().computeL2Squared(vector);
    }
    
    static float computeL1Vectorized(float[] vector) {
        return IMPL.computeL1(vector);
    }
    
    static float computeL1Scalar(float[] vector) {
        return new ScalarImpl().computeL1(vector);
    }
    
    private VectorNorm() {}
}