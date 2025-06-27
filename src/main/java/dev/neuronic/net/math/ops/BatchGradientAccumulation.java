package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Batch gradient accumulation operations for mini-batch training.
 * Provides efficient methods to accumulate and average gradients across a batch.
 */
public final class BatchGradientAccumulation {
    
    public interface Impl {
        void averageGradients(float[][] batchGradients, float[] output);
        void averageWeightGradients(float[][][] batchWeightGradients, float[][] output);
        void scaleGradients(float[] gradients, float scale);
        void computeBatchWeightGradients(float[][] batchInputs, float[][] batchNeuronDeltas, float[][] output);
    }
    
    private static final class ScalarImpl implements Impl {
    
        @Override
        public void averageGradients(float[][] batchGradients, float[] output) {
            int batchSize = batchGradients.length;
            if (batchSize == 0) return;
            
            int gradientSize = batchGradients[0].length;
            float scale = 1.0f / batchSize;
            
            java.util.Arrays.fill(output, 0.0f);
            
            for (int b = 0; b < batchSize; b++) {
                float[] gradient = batchGradients[b];
                for (int i = 0; i < gradientSize; i++) {
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
                    
                    for (int n = 0; n < neurons; n++) {
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
            for (int i = 0; i < gradients.length; i++) {
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
                    
                    for (int n = 0; n < neurons; n++) {
                        outputRow[n] += inputValue * neuronDeltas[n];
                    }
                }
            }
            
            for (int i = 0; i < inputSize; i++) {
                scaleGradients(output[i], scale);
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.BatchGradientAccumulationVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Accumulate gradients from a batch and compute the average.
     * Used for averaging gradients across mini-batch during backpropagation.
     * 
     * @param batchGradients gradients from each sample [batchSize][gradientSize]
     * @param output pre-allocated output array for averaged gradients [gradientSize]
     */
    public static void averageGradients(float[][] batchGradients, float[] output) {
        IMPL.averageGradients(batchGradients, output);
    }
    
    /**
     * Accumulate weight gradients from a batch for a 2D weight matrix.
     * Computes the average gradient for each weight across the batch.
     * 
     * @param batchWeightGradients gradients from each sample [batchSize][inputSize][neurons]
     * @param output pre-allocated output array [inputSize][neurons]
     */
    public static void averageWeightGradients(float[][][] batchWeightGradients, float[][] output) {
        IMPL.averageWeightGradients(batchWeightGradients, output);
    }
    
    /**
     * Scale gradients by a factor (e.g., 1/batchSize for averaging).
     * In-place operation for efficiency.
     * 
     * @param gradients gradient array to scale
     * @param scale scaling factor
     */
    public static void scaleGradients(float[] gradients, float scale) {
        IMPL.scaleGradients(gradients, scale);
    }
    
    /**
     * Compute batch weight gradients using outer product accumulation.
     * For each sample: weightGrad[i][j] += input[i] * neuronDelta[j]
     * Then average across the batch.
     * 
     * @param batchInputs inputs for each sample [batchSize][inputSize]
     * @param batchNeuronDeltas neuron deltas for each sample [batchSize][neurons]
     * @param output pre-allocated output [inputSize][neurons]
     */
    public static void computeBatchWeightGradients(float[][] batchInputs, float[][] batchNeuronDeltas, 
                                                   float[][] output) {
        IMPL.computeBatchWeightGradients(batchInputs, batchNeuronDeltas, output);
    }
    
    private BatchGradientAccumulation() {}
}