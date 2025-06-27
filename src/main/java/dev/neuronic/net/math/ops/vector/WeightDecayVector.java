package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.WeightDecay;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of WeightDecay.
 * This class is only loaded when Vector API is available.
 */
public final class WeightDecayVector implements WeightDecay.Impl {
    
    @Override
    public void compute(float[] weights, float keepRate) {
        if (!Vectorization.shouldVectorize(weights.length)) {
            scalarCompute(weights, keepRate);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector keepVec = FloatVector.broadcast(species, keepRate);
        int upperBound = Vectorization.loopBound(weights.length);
        
        int i = 0;
        for (; i < upperBound; i += species.length()) {
            FloatVector v = FloatVector.fromArray(species, weights, i);
            v.mul(keepVec).intoArray(weights, i);
        }
        
        for (; i < weights.length; i++) {
            weights[i] *= keepRate;
        }
    }
    
    private void scalarCompute(float[] weights, float keepRate) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] *= keepRate;
        }
    }
}