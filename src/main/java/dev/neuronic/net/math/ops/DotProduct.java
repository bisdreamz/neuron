package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Dot product implementation with automatic SIMD/scalar selection.
 */
public final class DotProduct {
    
    public interface Impl {
        float compute(float[] a, float[] b);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public float compute(float[] a, float[] b) {
            float sum = 0;
            int i = 0;
            int unrollBound = a.length - 3;
            
            for (; i < unrollBound; i += 4) {
                sum += a[i] * b[i] + 
                       a[i+1] * b[i+1] + 
                       a[i+2] * b[i+2] + 
                       a[i+3] * b[i+3];
            }
            
            for (; i < a.length; i++) {
                sum += a[i] * b[i];
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
                        "dev.neuronic.net.math.ops.vector.DotProductVector");
                // Create wrapper that calls the static method
                impl = new Impl() {
                    private final java.lang.reflect.Method computeMethod;
                    {
                        computeMethod = vectorClass.getMethod("compute", float[].class, float[].class);
                    }
                    
                    @Override
                    public float compute(float[] a, float[] b) {
                        try {
                            return (float) computeMethod.invoke(null, a, b);
                        } catch (Exception e) {
                            // Fall back to scalar
                            return new ScalarImpl().compute(a, b);
                        }
                    }
                };
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute dot product of two float arrays.
     * Automatically selects vectorized or scalar implementation.
     * 
     * @param a first array
     * @param b second array
     * @return dot product result
     */
    public static float compute(float[] a, float[] b) {
        if (a.length != b.length)
            throw new IllegalArgumentException("Arrays must have same length");
            
        if (Vectorization.shouldVectorize(a.length))
            return computeVectorized(a, b);
        else
            return computeScalar(a, b);
    }
    
    /**
     * Vectorized implementation using Java Vector API.
     */
    public static float computeVectorized(float[] a, float[] b) {
        return IMPL.compute(a, b);
    }
    
    /**
     * Scalar implementation with loop unrolling for performance.
     */
    public static float computeScalar(float[] a, float[] b) {
        return new ScalarImpl().compute(a, b);
    }
    
    private DotProduct() {} // Prevent instantiation
}