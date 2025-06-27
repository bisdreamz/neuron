package dev.neuronic.benchmarks;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.adamw.FusedAdamWUpdate;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Benchmark to compare old vs new AdamW implementations.
 * Shows the performance improvement from fused operations.
 */
public class AdamWOptimizationBenchmark {
    
    public static void main(String[] args) {
        System.out.println("=== AdamW Optimization Benchmark ===");
        System.out.println("Vector API available: " + Vectorization.isAvailable());
        System.out.println("Vector length: " + Vectorization.getVectorLength());
        System.out.println();
        
        // Test different layer sizes typical in language models
        int[] layerSizes = {64, 128, 256, 512, 1024};
        int[] batchSizes = {256, 512};
        
        for (int batchSize : batchSizes) {
            System.out.println("Batch size: " + batchSize);
            System.out.println("-----------------------------------------");
            for (int layerSize : layerSizes) {
                benchmarkLayerSize(layerSize, batchSize);
            }
            System.out.println();
        }
    }
    
    private static void benchmarkLayerSize(int layerSize, int batchSize) {
        // Create test data
        float[][] weights = createRandomMatrix(layerSize, layerSize);
        float[] biases = createRandomArray(layerSize);
        float[][] weightGradients = createRandomMatrix(layerSize, layerSize);
        float[] biasGradients = createRandomArray(layerSize);
        
        // Warmup
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        for (int i = 0; i < 1000; i++) {
            optimizer.optimize(weights, biases, weightGradients, biasGradients);
        }
        
        // Benchmark optimized AdamW
        long startTime = System.nanoTime();
        int iterations = 10000;
        
        for (int i = 0; i < iterations; i++) {
            optimizer.optimize(weights, biases, weightGradients, biasGradients);
        }
        
        long endTime = System.nanoTime();
        long totalTime = endTime - startTime;
        double timePerIteration = totalTime / (double) iterations / 1000; // Convert to microseconds
        
        // Also benchmark just the fused operation directly
        float[] params = createRandomArray(layerSize * layerSize);
        float[] gradients = createRandomArray(layerSize * layerSize);
        float[] momentum = new float[params.length];
        float[] velocity = new float[params.length];
        
        // Warmup fused operation
        for (int i = 0; i < 1000; i++) {
            FusedAdamWUpdate.compute(params, gradients, momentum, velocity,
                    0.9f, 0.999f, 0.001f, 1e-8f, 0.01f, 0.1f, 0.001f, true);
        }
        
        // Benchmark fused operation
        long fusedStart = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            FusedAdamWUpdate.compute(params, gradients, momentum, velocity,
                    0.9f, 0.999f, 0.001f, 1e-8f, 0.01f, 0.1f, 0.001f, true);
        }
        long fusedEnd = System.nanoTime();
        long fusedTime = fusedEnd - fusedStart;
        double fusedPerIteration = fusedTime / (double) iterations / 1000;
        
        System.out.printf("Layer %4d×%-4d: AdamW=%6.2f μs/iter | Fused core=%6.2f μs/iter | " +
                         "Elements=%d\n",
                         layerSize, layerSize, timePerIteration, fusedPerIteration,
                         layerSize * layerSize);
    }
    
    private static float[][] createRandomMatrix(int rows, int cols) {
        ThreadLocalRandom random = ThreadLocalRandom.current();
        float[][] matrix = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = (float) (random.nextGaussian() * 0.01);
            }
        }
        return matrix;
    }
    
    private static float[] createRandomArray(int size) {
        ThreadLocalRandom random = ThreadLocalRandom.current();
        float[] array = new float[size];
        
        for (int i = 0; i < size; i++) {
            array[i] = (float) (random.nextGaussian() * 0.01);
        }
        return array;
    }
}