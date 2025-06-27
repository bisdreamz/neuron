package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.ColumnMajorPreActivations;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of ColumnMajorPreActivations.
 * This class is only loaded when Vector API is available.
 */
public final class ColumnMajorPreActivationsVector implements ColumnMajorPreActivations.Impl {
    
    @Override
    public void compute(float[] inputs, float[][] weights, float[] biases, float[] output) {
        System.arraycopy(biases, 0, output, 0, biases.length);
        
        if (!Vectorization.shouldVectorize(output.length)) {
            scalarCompute(inputs, weights, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int upperBound = Vectorization.loopBound(output.length);
        
        for (int inputIdx = 0; inputIdx < inputs.length; inputIdx++) {
            float inputValue = inputs[inputIdx];
            float[] weightRow = weights[inputIdx];
            
            FloatVector vInputValue = FloatVector.broadcast(species, inputValue);
            
            for (int neuronIdx = 0; neuronIdx < upperBound; neuronIdx += species.length()) {
                FloatVector vWeights = FloatVector.fromArray(species, weightRow, neuronIdx);
                FloatVector vOutput = FloatVector.fromArray(species, output, neuronIdx);
                
                vInputValue.fma(vWeights, vOutput).intoArray(output, neuronIdx);
            }
            
            for (int neuronIdx = upperBound; neuronIdx < output.length; neuronIdx++) {
                output[neuronIdx] += inputValue * weightRow[neuronIdx];
            }
        }
    }
    
    private void scalarCompute(float[] inputs, float[][] weights, float[] output) {
        for (int inputIdx = 0; inputIdx < inputs.length; inputIdx++) {
            float inputValue = inputs[inputIdx];
            float[] weightRow = weights[inputIdx];
            
            for (int neuronIdx = 0; neuronIdx < output.length; neuronIdx++) {
                output[neuronIdx] += inputValue * weightRow[neuronIdx];
            }
        }
    }
}