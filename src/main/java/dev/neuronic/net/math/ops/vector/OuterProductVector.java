package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.OuterProduct;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of OuterProduct.
 * This class is only loaded when Vector API is available.
 */
public final class OuterProductVector implements OuterProduct.Impl {
    
    @Override
    public void compute(float[] a, float[] b, float[][] output) {
        if (!Vectorization.shouldVectorize(b.length)) {
            scalarCompute(a, b, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int upperBound = Vectorization.loopBound(b.length);
        
        for (int i = 0; i < a.length; i++) {
            float ai = a[i];
            float[] outputRow = output[i];
            FloatVector aiVector = FloatVector.broadcast(species, ai);
            
            int j = 0;
            for (; j < upperBound; j += species.length()) {
                FloatVector bVector = FloatVector.fromArray(species, b, j);
                aiVector.mul(bVector).intoArray(outputRow, j);
            }
            
            for (; j < b.length; j++) {
                outputRow[j] = ai * b[j];
            }
        }
    }
    
    private void scalarCompute(float[] a, float[] b, float[][] output) {
        for (int i = 0; i < a.length; i++) {
            float ai = a[i];
            float[] outputRow = output[i];
            
            for (int j = 0; j < b.length; j++) {
                outputRow[j] = ai * b[j];
            }
        }
    }
}