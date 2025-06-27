package dev.neuronic.net;

import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import java.util.concurrent.*;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test concurrent trainBatch calls work correctly.
 */
public class ConcurrentTrainBatchTest {
    
    @Test
    public void testConcurrentTrainBatch() throws Exception {
        // Create a simple network
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .input(10)
            .layer(Layers.hiddenDenseRelu(20, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(5, optimizer));
            
        // Test data
        Random rng = new Random(42);
        int numThreads = 4;
        int batchesPerThread = 10;
        int batchSize = 8;
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);
        
        // Launch threads
        for (int t = 0; t < numThreads; t++) {
            executor.submit(() -> {
                try {
                    startLatch.await(); // Wait for all threads to be ready
                    
                    // Each thread trains multiple batches
                    for (int b = 0; b < batchesPerThread; b++) {
                        float[][] inputs = new float[batchSize][10];
                        float[][] targets = new float[batchSize][5];
                        
                        // Random data
                        for (int i = 0; i < batchSize; i++) {
                            for (int j = 0; j < 10; j++)
                                inputs[i][j] = rng.nextFloat() * 2 - 1;
                            
                            // One-hot target
                            int targetClass = rng.nextInt(5);
                            targets[i][targetClass] = 1.0f;
                        }
                        
                        // Train batch - should be lock-free during forward/backward
                        net.trainBatch(inputs, targets);
                    }
                    
                    doneLatch.countDown();
                } catch (Exception e) {
                    e.printStackTrace();
                    fail("Thread failed: " + e.getMessage());
                }
            });
        }
        
        // Start all threads simultaneously
        startLatch.countDown();
        
        // Wait for completion
        assertTrue(doneLatch.await(10, TimeUnit.SECONDS), "Training took too long");
        
        executor.shutdown();
        
        // Verify network still works
        float[] testInput = new float[10];
        for (int i = 0; i < 10; i++)
            testInput[i] = rng.nextFloat() * 2 - 1;
            
        float[] output = net.predict(testInput);
        assertEquals(5, output.length);
        
        // Verify output is valid probability distribution
        float sum = 0;
        for (float val : output) {
            assertTrue(val >= 0 && val <= 1, "Invalid probability: " + val);
            sum += val;
        }
        assertEquals(1.0f, sum, 0.01f, "Probabilities should sum to 1");
    }
}