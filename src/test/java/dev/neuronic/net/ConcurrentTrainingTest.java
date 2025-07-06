package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Concurrent training test to detect race conditions and thread safety issues.
 * 
 * This test creates multiple threads that simultaneously:
 * - Perform forward passes (predictions)
 * - Perform backward passes (training) 
 * - Use shared embedding tables
 * - Update AUTO_NORMALIZE statistics
 * 
 * Any race conditions would typically manifest as:
 * - ArrayIndexOutOfBoundsException
 * - NullPointerException
 * - Inconsistent results
 * - Deadlocks (caught by timeout)
 */
class ConcurrentTrainingTest {
    
    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void testConcurrentTrainingWithMixedFeatures() throws Exception {
        // Create a model with mixed features to stress test all code paths
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet model = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(6)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(1000, 32, "user_id"),        // Shared embeddings
                Feature.hashedEmbedding(10000, 16, "domain"),   // Hashed embeddings
                Feature.oneHot(10, "device_type"),              // One-hot encoding
                Feature.passthrough("ctr"),                     // Pass-through
                Feature.autoScale(0.0f, 100.0f, "bid_price"),   // Scaled feature
                Feature.autoNormalize("user_age")               // Auto-normalized (thread-safe stats)
            ))
            .layer(Layers.hiddenDenseRelu(64))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
        
        // Test parameters
        int numThreads = 8;
        int iterationsPerThread = 1000;
        int batchSize = 16;
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch completionLatch = new CountDownLatch(numThreads);
        
        AtomicBoolean errorOccurred = new AtomicBoolean(false);
        AtomicInteger successfulIterations = new AtomicInteger(0);
        ConcurrentLinkedQueue<Exception> exceptions = new ConcurrentLinkedQueue<>();
        
        // Create worker threads
        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    // Wait for all threads to be ready
                    startLatch.await();
                    
                    Random rand = new Random(threadId);
                    
                    for (int iter = 0; iter < iterationsPerThread; iter++) {
                        // Mix of single training and batch training
                        if (iter % 3 == 0) {
                            // Single sample training
                            float[] input = generateRandomInput(rand);
                            float[] target = {rand.nextFloat() * 100};
                            
                            model.train(input, target);
                        } else {
                            // Batch training
                            float[][] inputs = new float[batchSize][];
                            float[][] targets = new float[batchSize][];
                            
                            for (int i = 0; i < batchSize; i++) {
                                inputs[i] = generateRandomInput(rand);
                                targets[i] = new float[]{rand.nextFloat() * 100};
                            }
                            
                            model.trainBatch(inputs, targets);
                        }
                        
                        // Also do some predictions to stress forward pass
                        if (iter % 5 == 0) {
                            float[] testInput = generateRandomInput(rand);
                            float[] prediction = model.predict(testInput);
                            
                            // Basic sanity check
                            assertNotNull(prediction);
                            assertEquals(1, prediction.length);
                            assertFalse(Float.isNaN(prediction[0]));
                            assertFalse(Float.isInfinite(prediction[0]));
                        }
                        
                        successfulIterations.incrementAndGet();
                    }
                } catch (Exception e) {
                    errorOccurred.set(true);
                    exceptions.add(e);
                    e.printStackTrace();
                } finally {
                    completionLatch.countDown();
                }
            });
        }
        
        // Start all threads simultaneously
        startLatch.countDown();
        
        // Wait for completion
        boolean completed = completionLatch.await(25, TimeUnit.SECONDS);
        assertTrue(completed, "Training should complete within timeout");
        
        executor.shutdown();
        assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        
        // Check results
        assertFalse(errorOccurred.get(), 
            "No exceptions should occur during concurrent training. Exceptions: " + exceptions);
        
        int expectedIterations = numThreads * iterationsPerThread;
        assertEquals(expectedIterations, successfulIterations.get(),
            "All iterations should complete successfully");
        
        // Verify model still works after concurrent training
        float[] finalInput = generateRandomInput(new Random(42));
        float[] finalPrediction = model.predict(finalInput);
        assertNotNull(finalPrediction);
        assertFalse(Float.isNaN(finalPrediction[0]));
    }
    
    @Test 
    @Timeout(value = 20, unit = TimeUnit.SECONDS)
    void testConcurrentPredictionsOnly() throws Exception {
        // Test that concurrent predictions (no training) work correctly
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet model = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(3)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(100, 16, "category"),
                Feature.passthrough("value1"),
                Feature.autoNormalize("value2")
            ))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
        
        // Train the model first to set up some state
        for (int i = 0; i < 100; i++) {
            float[] input = {i % 100, i * 0.1f, i * 0.01f};
            float[] target = {i * 0.5f};
            model.train(input, target);
        }
        
        // Now test concurrent predictions
        int numThreads = 10;
        int predictionsPerThread = 10000;
        
        ExecutorService executor = Executors.newCachedThreadPool();
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch completionLatch = new CountDownLatch(numThreads);
        
        AtomicBoolean errorOccurred = new AtomicBoolean(false);
        ConcurrentLinkedQueue<Exception> exceptions = new ConcurrentLinkedQueue<>();
        
        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    startLatch.await();
                    
                    Random rand = new Random(threadId);
                    for (int i = 0; i < predictionsPerThread; i++) {
                        float[] input = {
                            rand.nextInt(100),
                            rand.nextFloat() * 10,
                            rand.nextFloat() * 5
                        };
                        
                        float[] prediction = model.predict(input);
                        
                        // Verify prediction is valid
                        assertNotNull(prediction);
                        assertEquals(1, prediction.length);
                        assertFalse(Float.isNaN(prediction[0]));
                        assertFalse(Float.isInfinite(prediction[0]));
                    }
                } catch (Exception e) {
                    errorOccurred.set(true);
                    exceptions.add(e);
                    e.printStackTrace();
                } finally {
                    completionLatch.countDown();
                }
            });
        }
        
        startLatch.countDown();
        boolean completed = completionLatch.await(15, TimeUnit.SECONDS);
        assertTrue(completed, "Predictions should complete within timeout");
        
        executor.shutdown();
        assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        
        assertFalse(errorOccurred.get(),
            "No exceptions should occur during concurrent predictions. Exceptions: " + exceptions);
    }
    
    @Test
    @Timeout(value = 20, unit = TimeUnit.SECONDS)
    void testAutoNormalizeStatisticsConcurrency() throws Exception {
        // Specifically test AUTO_NORMALIZE feature concurrent updates
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet model = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(3)
            .layer(Layers.inputMixed(optimizer,
                Feature.passthrough("id"),
                Feature.autoNormalize("value1"),
                Feature.autoNormalize("value2")
            ))
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputLinearRegression(1));
        
        int numThreads = 10;
        int samplesPerThread = 1000;
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch latch = new CountDownLatch(numThreads);
        AtomicBoolean errorOccurred = new AtomicBoolean(false);
        
        // Generate deterministic values for statistics validation
        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    Random rand = new Random(threadId);
                    for (int i = 0; i < samplesPerThread; i++) {
                        // Use deterministic values that should produce known statistics
                        float value1 = threadId + i * 0.01f;
                        float value2 = threadId * 10 + i * 0.1f;
                        
                        float[] input = {threadId, value1, value2};
                        float[] target = {value1 + value2};
                        
                        model.train(input, target);
                    }
                } catch (Exception e) {
                    errorOccurred.set(true);
                    e.printStackTrace();
                } finally {
                    latch.countDown();
                }
            });
        }
        
        boolean completed = latch.await(15, TimeUnit.SECONDS);
        assertTrue(completed, "Statistics updates should complete");
        
        executor.shutdown();
        assertFalse(errorOccurred.get(), "No errors during concurrent statistics updates");
        
        // Verify predictions still work and produce reasonable outputs
        for (int i = 0; i < 10; i++) {
            float[] input = {i, i * 1.0f, i * 10.0f};
            float[] pred = model.predict(input);
            assertFalse(Float.isNaN(pred[0]), "Predictions should not be NaN after concurrent updates");
            assertFalse(Float.isInfinite(pred[0]), "Predictions should not be infinite");
        }
    }
    
    private float[] generateRandomInput(Random rand) {
        return new float[] {
            rand.nextInt(1000),                    // user_id for embedding
            (float) "domain.com".hashCode(),       // domain hash
            rand.nextInt(10),                      // device_type
            rand.nextFloat(),                      // ctr (0-1)
            rand.nextFloat() * 100,                // bid_price (0-100)
            18 + rand.nextFloat() * 47             // user_age (18-65)
        };
    }
}