package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.MatrixInit;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of MatrixInit.
 * This class is only loaded when Vector API is available.
 */
public final class MatrixInitVector implements MatrixInit.Impl {
    
    @Override
    public void compute(float[][] matrix, float value) {
        if (!Vectorization.shouldVectorize(matrix.length * matrix[0].length)) {
            scalarCompute(matrix, value);
            return;
        }
        
        VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
        FloatVector valueVector = FloatVector.broadcast(SPECIES, value);
        
        for (int i = 0; i < matrix.length; i++) {
            float[] row = matrix[i];
            int bound = Vectorization.loopBound(row.length);
            
            int j = 0;
            for (; j < bound; j += SPECIES.length()) {
                valueVector.intoArray(row, j);
            }
            
            for (; j < row.length; j++) {
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
    
    private void scalarCompute(float[][] matrix, float value) {
        for (int i = 0; i < matrix.length; i++) {
            float[] row = matrix[i];
            for (int j = 0; j < row.length; j++) {
                row[j] = value;
            }
        }
    }
}