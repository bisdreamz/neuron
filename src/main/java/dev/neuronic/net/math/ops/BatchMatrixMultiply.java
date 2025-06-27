package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Parallelization;
import dev.neuronic.net.math.Vectorization;
import java.util.concurrent.ExecutorService;

/**
 * Batch matrix multiplication for processing multiple samples simultaneously.
 * Optimized for neural network forward passes with mini-batches.
 * 
 * Computes: output[batch][neuron] = sum(inputs[batch][input] * weights[input][neuron]) + biases[neuron]
 * 
 * This is equivalent to computing preactivations for all samples in a batch at once,
 * enabling significant performance improvements through better cache usage and vectorization.
 */
public final class BatchMatrixMultiply {
    
    public interface Impl {
        void compute(float[][] inputs, float[][] weights, float[] biases, float[][] outputs);
        void computeParallel(float[][] inputs, float[][] weights, float[] biases, float[][] outputs, ExecutorService executor);
    }
    
    private static final class ScalarImpl implements Impl {
    
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
                    
                    for (int n = 0; n < neurons; n++) {
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
                
                for (int n = 0; n < neurons; n++) {
                    output[n] += weightRow[n] * inputValue;
                }
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.BatchMatrixMultiplyVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute batch matrix multiplication for forward pass.
     * 
     * @param inputs batch inputs [batchSize][inputSize]
     * @param weights weight matrix [inputSize][neurons] (column-major)
     * @param biases bias vector [neurons]
     * @param outputs pre-allocated output [batchSize][neurons]
     */
    public static void compute(float[][] inputs, float[][] weights, float[] biases, float[][] outputs) {
        IMPL.compute(inputs, weights, biases, outputs);
    }
    
    /**
     * Compute batch matrix multiplication with parallelization across samples.
     * Useful for large batch sizes where parallelizing across samples is beneficial.
     * 
     * @param inputs batch inputs [batchSize][inputSize]
     * @param weights weight matrix [inputSize][neurons] (column-major)
     * @param biases bias vector [neurons]
     * @param outputs pre-allocated output [batchSize][neurons]
     * @param executor executor service for parallelization
     */
    public static void computeParallel(float[][] inputs, float[][] weights, float[] biases, 
                                      float[][] outputs, ExecutorService executor) {
        IMPL.computeParallel(inputs, weights, biases, outputs, executor);
    }
    
    private BatchMatrixMultiply() {}
}