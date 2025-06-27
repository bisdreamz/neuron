package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.BatchGradientAccumulation;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of BatchGradientAccumulation.
 * This class is only loaded when Vector API is available.
 */
public final class BatchGradientAccumulationVector implements BatchGradientAccumulation.Impl {
    
    private static final VectorSpecies<Float> SPECIES = Vectorization.getSpecies();
    
    @Override
    public void averageGradients(float[][] batchGradients, float[] output) {
        int batchSize = batchGradients.length;
        if (batchSize == 0) return;
        
        int gradientSize = batchGradients[0].length;
        float scale = 1.0f / batchSize;
        
        java.util.Arrays.fill(output, 0.0f);
        
        for (int b = 0; b < batchSize; b++) {
            float[] gradient = batchGradients[b];
            
            int i = 0;
            for (; i < SPECIES.loopBound(gradientSize); i += SPECIES.length()) {
                FloatVector outputVec = FloatVector.fromArray(SPECIES, output, i);
                FloatVector gradVec = FloatVector.fromArray(SPECIES, gradient, i);
                outputVec = outputVec.add(gradVec);
                outputVec.intoArray(output, i);
            }
            
            for (; i < gradientSize; i++) {
                output[i] += gradient[i];
            }
        }
        
        scaleGradients(output, scale);
    }
    
    @Override
    public void averageWeightGradients(float[][][] batchWeightGradients, float[][] output) {
        int batchSize = batchWeightGradients.length;
        if (batchSize == 0) return;
        
        int inputSize = output.length;
        int neurons = output[0].length;
        float scale = 1.0f / batchSize;
        
        for (int i = 0; i < inputSize; i++) {
            java.util.Arrays.fill(output[i], 0.0f);
        }
        
        for (int b = 0; b < batchSize; b++) {
            float[][] sampleGradients = batchWeightGradients[b];
            
            for (int i = 0; i < inputSize; i++) {
                float[] gradientRow = sampleGradients[i];
                float[] outputRow = output[i];
                
                int n = 0;
                for (; n < SPECIES.loopBound(neurons); n += SPECIES.length()) {
                    FloatVector outputVec = FloatVector.fromArray(SPECIES, outputRow, n);
                    FloatVector gradVec = FloatVector.fromArray(SPECIES, gradientRow, n);
                    outputVec = outputVec.add(gradVec);
                    outputVec.intoArray(outputRow, n);
                }
                
                for (; n < neurons; n++) {
                    outputRow[n] += gradientRow[n];
                }
            }
        }
        
        for (int i = 0; i < inputSize; i++) {
            scaleGradients(output[i], scale);
        }
    }
    
    @Override
    public void scaleGradients(float[] gradients, float scale) {
        int size = gradients.length;
        
        int i = 0;
        FloatVector scaleVec = FloatVector.broadcast(SPECIES, scale);
        
        for (; i < SPECIES.loopBound(size); i += SPECIES.length()) {
            FloatVector gradVec = FloatVector.fromArray(SPECIES, gradients, i);
            gradVec = gradVec.mul(scaleVec);
            gradVec.intoArray(gradients, i);
        }
        
        for (; i < size; i++) {
            gradients[i] *= scale;
        }
    }
    
    @Override
    public void computeBatchWeightGradients(float[][] batchInputs, float[][] batchNeuronDeltas, 
                                                   float[][] output) {
        int batchSize = batchInputs.length;
        if (batchSize == 0) return;
        
        int inputSize = batchInputs[0].length;
        int neurons = batchNeuronDeltas[0].length;
        float scale = 1.0f / batchSize;
        
        for (int i = 0; i < inputSize; i++) {
            java.util.Arrays.fill(output[i], 0.0f);
        }
        
        for (int b = 0; b < batchSize; b++) {
            float[] input = batchInputs[b];
            float[] neuronDeltas = batchNeuronDeltas[b];
            
            for (int i = 0; i < inputSize; i++) {
                float inputValue = input[i];
                float[] outputRow = output[i];
                
                int n = 0;
                for (; n < SPECIES.loopBound(neurons); n += SPECIES.length()) {
                    FloatVector outputVec = FloatVector.fromArray(SPECIES, outputRow, n);
                    FloatVector deltaVec = FloatVector.fromArray(SPECIES, neuronDeltas, n);
                    outputVec = outputVec.add(deltaVec.mul(inputValue));
                    outputVec.intoArray(outputRow, n);
                }
                
                for (; n < neurons; n++) {
                    outputRow[n] += inputValue * neuronDeltas[n];
                }
            }
        }
        
        for (int i = 0; i < inputSize; i++) {
            scaleGradients(output[i], scale);
        }
    }
}