package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.WeightGradientsColumnMajor;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of WeightGradientsColumnMajor.
 * This class is only loaded when Vector API is available.
 */
public final class WeightGradientsColumnMajorVector implements WeightGradientsColumnMajor.Impl {
    
    @Override
    public void compute(float[] inputs, float[] neuronDeltas, float[][] weightGradients) {
        if (!Vectorization.shouldVectorize(neuronDeltas.length)) {
            scalarCompute(inputs, neuronDeltas, weightGradients);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int neuronsVectorBound = Vectorization.loopBound(neuronDeltas.length);
        
        for (int inputIdx = 0; inputIdx < inputs.length; inputIdx++) {
            float inputValue = inputs[inputIdx];
            FloatVector inputVec = FloatVector.broadcast(species, inputValue);
            
            int neuronIdx = 0;
            
            for (; neuronIdx < neuronsVectorBound; neuronIdx += species.length()) {
                FloatVector deltaVec = FloatVector.fromArray(species, neuronDeltas, neuronIdx);
                FloatVector gradientVec = inputVec.mul(deltaVec);
                gradientVec.intoArray(weightGradients[inputIdx], neuronIdx);
            }
            
            for (; neuronIdx < neuronDeltas.length; neuronIdx++) {
                weightGradients[inputIdx][neuronIdx] = inputValue * neuronDeltas[neuronIdx];
            }
        }
    }
    
    private void scalarCompute(float[] inputs, float[] neuronDeltas, float[][] weightGradients) {
        for (int inputIdx = 0; inputIdx < inputs.length; inputIdx++) {
            float inputValue = inputs[inputIdx];
            for (int neuronIdx = 0; neuronIdx < neuronDeltas.length; neuronIdx++) {
                weightGradients[inputIdx][neuronIdx] = inputValue * neuronDeltas[neuronIdx];
            }
        }
    }
}