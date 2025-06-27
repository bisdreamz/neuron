package dev.neuronic;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

public class GradientClippingPerformanceTest {
    
    @Test
    public void comparePerformanceWithAndWithoutClipping() {
        int inputSize = 100;
        int hiddenSize = 256;
        int outputSize = 50;
        int iterations = 1000;
        
        // Create network WITHOUT gradient clipping
        NeuralNet netWithoutClipping = NeuralNet.newBuilder()
            .input(inputSize)
            .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
            .withGlobalGradientClipping(0.0f) // Disabled
            .layer(Layers.hiddenDenseRelu(hiddenSize))
            .layer(Layers.hiddenDenseRelu(hiddenSize))
            .output(Layers.outputSoftmaxCrossEntropy(outputSize));
        
        // Create network WITH gradient clipping
        NeuralNet netWithClipping = NeuralNet.newBuilder()
            .input(inputSize)
            .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
            .withGlobalGradientClipping(1.0f) // Enabled
            .layer(Layers.hiddenDenseRelu(hiddenSize))
            .layer(Layers.hiddenDenseRelu(hiddenSize))
            .output(Layers.outputSoftmaxCrossEntropy(outputSize));
        
        // Prepare data
        float[] input = new float[inputSize];
        float[] target = new float[outputSize];
        for (int i = 0; i < inputSize; i++) {
            input[i] = (float) Math.random();
        }
        target[0] = 1.0f; // One-hot
        
        // Warmup both networks
        System.out.println("Warming up...");
        for (int i = 0; i < 100; i++) {
            netWithoutClipping.train(input, target);
            netWithClipping.train(input, target);
        }
        
        // Benchmark WITHOUT clipping
        System.out.println("\nBenchmarking WITHOUT gradient clipping:");
        long startTime = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            netWithoutClipping.train(input, target);
        }
        long withoutClippingTime = System.nanoTime() - startTime;
        double withoutClippingMs = withoutClippingTime / 1_000_000.0;
        System.out.printf("Time for %d iterations: %.2f ms (%.2f µs per iteration)\n", 
            iterations, withoutClippingMs, (withoutClippingTime / 1000.0) / iterations);
        
        // Benchmark WITH clipping
        System.out.println("\nBenchmarking WITH gradient clipping:");
        startTime = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            netWithClipping.train(input, target);
        }
        long withClippingTime = System.nanoTime() - startTime;
        double withClippingMs = withClippingTime / 1_000_000.0;
        System.out.printf("Time for %d iterations: %.2f ms (%.2f µs per iteration)\n", 
            iterations, withClippingMs, (withClippingTime / 1000.0) / iterations);
        
        // Calculate overhead
        double overhead = ((double) withClippingTime / withoutClippingTime - 1.0) * 100;
        System.out.printf("\nGradient clipping overhead: %.1f%%\n", overhead);
        
        // Memory allocation test
        System.out.println("\nTesting allocation behavior:");
        
        // Force GC and measure memory before
        System.gc();
        Thread.yield();
        long memBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        
        // Train some more
        for (int i = 0; i < 100; i++) {
            netWithClipping.train(input, target);
        }
        
        // Measure memory after
        long memAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        long memDiff = memAfter - memBefore;
        
        System.out.printf("Memory difference after 100 iterations: %d bytes (%.2f KB)\n", 
            memDiff, memDiff / 1024.0);
        System.out.println("Near-zero allocation indicates buffer pooling is working correctly.");
    }
    
    @Test
    public void profileBatchTraining() {
        int batchSize = 32;
        int inputSize = 50;
        int hiddenSize = 128;
        int outputSize = 10;
        int iterations = 100;
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(inputSize)
            .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
            .withGlobalGradientClipping(1.0f)
            .layer(Layers.hiddenDenseRelu(hiddenSize))
            .layer(Layers.hiddenDenseRelu(hiddenSize))
            .output(Layers.outputSoftmaxCrossEntropy(outputSize));
        
        // Prepare batch data
        float[][] batchInputs = new float[batchSize][inputSize];
        float[][] batchTargets = new float[batchSize][outputSize];
        
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < inputSize; i++) {
                batchInputs[b][i] = (float) Math.random();
            }
            batchTargets[b][b % outputSize] = 1.0f;
        }
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            net.trainBatch(batchInputs, batchTargets);
        }
        
        // Benchmark
        System.out.println("\nBatch training performance (batch size = " + batchSize + "):");
        long startTime = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            net.trainBatch(batchInputs, batchTargets);
        }
        long totalTime = System.nanoTime() - startTime;
        
        double totalMs = totalTime / 1_000_000.0;
        double perBatchMs = totalMs / iterations;
        double perSampleUs = (totalTime / 1000.0) / (iterations * batchSize);
        
        System.out.printf("Total time for %d batches: %.2f ms\n", iterations, totalMs);
        System.out.printf("Time per batch: %.2f ms\n", perBatchMs);
        System.out.printf("Time per sample: %.2f µs\n", perSampleUs);
        System.out.printf("Throughput: %.0f samples/second\n", 
            (iterations * batchSize) / (totalMs / 1000.0));
    }
}