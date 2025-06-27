package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.MatrixVectorMultiplyColumnMajor;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of MatrixVectorMultiplyColumnMajor.
 * This class is only loaded when Vector API is available.
 */
public final class MatrixVectorMultiplyColumnMajorVector implements MatrixVectorMultiplyColumnMajor.Impl {
    
    @Override
    public void compute(float[][] matrix, float[] vector, float[] result) {
        if (!Vectorization.shouldVectorize(vector.length)) {
            scalarCompute(matrix, vector, result);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int vectorBound = Vectorization.loopBound(vector.length);
        
        for (int i = 0; i < matrix.length; i++) {
            FloatVector sumVec = FloatVector.zero(species);
            int j = 0;
            
            for (; j < vectorBound; j += species.length()) {
                FloatVector matrixVec = FloatVector.fromArray(species, matrix[i], j);
                FloatVector vectorVec = FloatVector.fromArray(species, vector, j);
                sumVec = matrixVec.fma(vectorVec, sumVec);
            }
            
            float sum = sumVec.reduceLanes(VectorOperators.ADD);
            
            for (; j < vector.length; j++) {
                sum += matrix[i][j] * vector[j];
            }

            result[i] = sum;
        }
    }
    
    private void scalarCompute(float[][] matrix, float[] vector, float[] result) {
        for (int i = 0; i < matrix.length; i++) {
            float sum = 0;
            for (int j = 0; j < vector.length; j++) {
                sum += matrix[i][j] * vector[j];
            }
            result[i] = sum;
        }
    }
}