package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.BiasInit;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of BiasInit.
 * This class is only loaded when Vector API is available.
 */
public final class BiasInitVector implements BiasInit.Impl {
    
    @Override
    public void compute(float[] biases, float value) {
        if (!Vectorization.shouldVectorize(biases.length)) {
            scalarCompute(biases, value);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int upperBound = Vectorization.loopBound(biases.length);
        
        FloatVector vValue = FloatVector.broadcast(species, value);
        
        for (int i = 0; i < upperBound; i += species.length()) {
            vValue.intoArray(biases, i);
        }
        
        for (int i = upperBound; i < biases.length; i++) {
            biases[i] = value;
        }
    }
    
    private void scalarCompute(float[] biases, float value) {
        for (int i = 0; i < biases.length; i++) {
            biases[i] = value;
        }
    }
}