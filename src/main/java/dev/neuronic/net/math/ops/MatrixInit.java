package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Initializes 2D matrices with a constant value.
 * Optimized with SIMD vectorization for maximum performance.
 */
public final class MatrixInit {
    
    public interface Impl {
        void compute(float[][] matrix, float value);
        void compute3D(float[][][] matrix, float value);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[][] matrix, float value) {
            for (int i = 0; i < matrix.length; i++) {
                float[] row = matrix[i];
                for (int j = 0; j < row.length; j++) {
                    row[j] = value;
                }
            }
        }
        
        @Override
        public void compute3D(float[][][] matrix, float value) {
            for (int i = 0; i < matrix.length; i++) {
                compute(matrix[i], value);
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.MatrixInitVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    public static void compute(float[][] matrix, float value) {
        IMPL.compute(matrix, value);
    }
    
    static void computeVectorized(float[][] matrix, float value) {
        IMPL.compute(matrix, value);
    }
    
    static void computeScalar(float[][] matrix, float value) {
        new ScalarImpl().compute(matrix, value);
    }
    
    /**
     * In-place initialization of a 3D matrix.
     * Useful for batch operations.
     */
    public static void compute3D(float[][][] matrix, float value) {
        IMPL.compute3D(matrix, value);
    }
}