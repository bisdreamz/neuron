package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.ParameterUpdate;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of ParameterUpdate.
 * This class is only loaded when Vector API is available.
 */
public final class ParameterUpdateVector implements ParameterUpdate.Impl {
    
    @Override
    public void compute(float[] parameters, float[] gradients, float learningRate) {
        if (!Vectorization.shouldVectorize(parameters.length)) {
            scalarCompute(parameters, gradients, learningRate);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector lr = FloatVector.broadcast(species, learningRate);
        int i = 0;
        int upperBound = Vectorization.loopBound(parameters.length);
        
        for (; i < upperBound; i += species.length()) {
            FloatVector params = FloatVector.fromArray(species, parameters, i);
            FloatVector grads = FloatVector.fromArray(species, gradients, i);
            grads.fma(lr.neg(), params).intoArray(parameters, i);
        }
        
        for (; i < parameters.length; i++) {
            parameters[i] -= learningRate * gradients[i];
        }
    }
    
    private void scalarCompute(float[] parameters, float[] gradients, float learningRate) {
        for (int i = 0; i < parameters.length; i++) {
            parameters[i] -= learningRate * gradients[i];
        }
    }
}