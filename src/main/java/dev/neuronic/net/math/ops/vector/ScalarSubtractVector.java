package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.ScalarSubtract;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of ScalarSubtract.
 * This class is only loaded when Vector API is available.
 */
public final class ScalarSubtractVector implements ScalarSubtract.Impl {
    
    @Override
    public void compute(float scalar, float[] array, float[] output) {
        if (!Vectorization.shouldVectorizeLimited(array.length, 32)) {
            scalarCompute(scalar, array, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector scalarVector = FloatVector.broadcast(species, scalar);
        int i = 0;
        int upperBound = Vectorization.loopBound(array.length);
        
        for (; i < upperBound; i += species.length()) {
            FloatVector va = FloatVector.fromArray(species, array, i);
            scalarVector.sub(va).intoArray(output, i);
        }
        
        for (; i < array.length; i++) {
            output[i] = scalar - array[i];
        }
    }
    
    private void scalarCompute(float scalar, float[] array, float[] output) {
        for (int i = 0; i < array.length; i++) {
            output[i] = scalar - array[i];
        }
    }
}