package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Parallelization;
import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.BatchMatrixMultiply;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import java.util.concurrent.ExecutorService;

/**
 * Vector implementation of BatchMatrixMultiply.
 * This class is only loaded when Vector API is available.
 */
public final class BatchMatrixMultiplyVector implements BatchMatrixMultiply.Impl {
    
    private static final VectorSpecies<Float> SPECIES = Vectorization.getSpecies();
    
    @Override
    public void compute(float[][] inputs, float[][] weights, float[] biases, float[][] outputs) {
        int batchSize = inputs.length;
        int inputSize = weights.length;
        int neurons = weights[0].length;
        
        for (int b = 0; b < batchSize; b++) {
            float[] input = inputs[b];
            float[] output = outputs[b];
            
            System.arraycopy(biases, 0, output, 0, neurons);
            
            for (int i = 0; i < inputSize; i++) {
                float inputValue = input[i];
                float[] weightRow = weights[i];
                
                int n = 0;
                for (; n < SPECIES.loopBound(neurons); n += SPECIES.length()) {
                    FloatVector outputVec = FloatVector.fromArray(SPECIES, output, n);
                    FloatVector weightVec = FloatVector.fromArray(SPECIES, weightRow, n);
                    outputVec = outputVec.add(weightVec.mul(inputValue));
                    outputVec.intoArray(output, n);
                }
                
                for (; n < neurons; n++) {
                    output[n] += weightRow[n] * inputValue;
                }
            }
        }
    }
    
    @Override
    public void computeParallel(float[][] inputs, float[][] weights, float[] biases, 
                                      float[][] outputs, ExecutorService executor) {
        int batchSize = inputs.length;
        
        if (!Parallelization.shouldParallelize(batchSize, executor)) {
            compute(inputs, weights, biases, outputs);
            return;
        }
        
        int numThreads = Parallelization.calculateOptimalThreads(batchSize, executor);
        Parallelization.WorkRange[] ranges =
            Parallelization.splitWork(batchSize, numThreads);
        
        java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(numThreads);
        
        for (Parallelization.WorkRange range : ranges) {
            executor.submit(() -> {
                try {
                    for (int b = range.start; b < range.end; b++) {
                        computeSingleSample(inputs[b], weights, biases, outputs[b]);
                    }
                } finally {
                    latch.countDown();
                }
            });
        }
        
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Batch matrix multiplication interrupted", e);
        }
    }
    
    private void computeSingleSample(float[] input, float[][] weights, float[] biases, float[] output) {
        int inputSize = weights.length;
        int neurons = weights[0].length;
        
        System.arraycopy(biases, 0, output, 0, neurons);
        
        for (int i = 0; i < inputSize; i++) {
            float inputValue = input[i];
            float[] weightRow = weights[i];
            
            int n = 0;
            for (; n < SPECIES.loopBound(neurons); n += SPECIES.length()) {
                FloatVector outputVec = FloatVector.fromArray(SPECIES, output, n);
                FloatVector weightVec = FloatVector.fromArray(SPECIES, weightRow, n);
                outputVec = outputVec.add(weightVec.mul(inputValue));
                outputVec.intoArray(output, n);
            }
            
            for (; n < neurons; n++) {
                output[n] += weightRow[n] * inputValue;
            }
        }
    }
}