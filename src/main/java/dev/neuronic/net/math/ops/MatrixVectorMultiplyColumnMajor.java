package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Optimized matrix-vector multiplication for column-major matrices.
 * Computes: result[i] = sum(matrix[i][j] * vector[j]) for all i
 * 
 * This is more efficient than multiple dot product calls because it reduces
 * the number of vector reduction operations.
 */
public final class MatrixVectorMultiplyColumnMajor {
    
    public interface Impl {
        void compute(float[][] matrix, float[] vector, float[] result);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[][] matrix, float[] vector, float[] result) {
            for (int i = 0; i < matrix.length; i++) {
                float sum = 0;
                for (int j = 0; j < vector.length; j++) {
                    sum += matrix[i][j] * vector[j];
                }
                result[i] = sum;
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.MatrixVectorMultiplyColumnMajorVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    public static void compute(float[][] matrix, float[] vector, float[] result) {
        IMPL.compute(matrix, vector, result);
    }
    
    private static void computeVectorized(float[][] matrix, float[] vector, float[] result) {
        IMPL.compute(matrix, vector, result);
    }
    
    private static void computeScalar(float[][] matrix, float[] vector, float[] result) {
        new ScalarImpl().compute(matrix, vector, result);
    }
    
    private MatrixVectorMultiplyColumnMajor() {}
}