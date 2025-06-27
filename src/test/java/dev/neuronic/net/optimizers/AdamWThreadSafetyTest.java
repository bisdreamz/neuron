package dev.neuronic.net.optimizers;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import org.junit.jupiter.api.Test;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive thread safety test for AdamWOptimizer.
 * 
 * This test catches the race condition that was causing gradient explosion.
 * The bug manifested as extreme gradient clipping warnings because concurrent
 * updates to timeStep were causing incorrect bias correction calculations.
 */
public class AdamWThreadSafetyTest {
    
    @Test
    public void testConcurrentTrainingStability() throws Exception {
        // Create a network prone to gradient explosion if optimizer is broken
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .input(100)
            .layer(Layers.hiddenDenseRelu(256, optimizer))
            .layer(Layers.hiddenDenseRelu(256, optimizer))
            .layer(Layers.hiddenDenseRelu(128, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(10, optimizer));
            
        // Track gradient explosion indicators
        AtomicInteger extremeClippingEvents = new AtomicInteger(0);
        AtomicBoolean gradientExploded = new AtomicBoolean(false);
        List<Float> lossHistory = new CopyOnWriteArrayList<>();
        
        // Override System.err to catch gradient clipping warnings
        java.io.PrintStream originalErr = System.err;
        java.io.ByteArrayOutputStream errCapture = new java.io.ByteArrayOutputStream();
        System.setErr(new java.io.PrintStream(errCapture));
        
        try {
            // Run parallel training
            int numThreads = 8;
            int batchesPerThread = 50;
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch doneLatch = new CountDownLatch(numThreads);
            
            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        // Wait for all threads to be ready
                        startLatch.await();
                        
                        Random rand = new Random(threadId);
                        
                        for (int batch = 0; batch < batchesPerThread; batch++) {
                            // Create random batch data
                            float[][] inputs = new float[32][100];
                            float[][] targets = new float[32][10];
                            
                            for (int i = 0; i < 32; i++) {
                                // Random inputs
                                for (int j = 0; j < 100; j++) {
                                    inputs[i][j] = (float) (rand.nextGaussian() * 0.1);
                                }
                                // One-hot targets
                                targets[i][rand.nextInt(10)] = 1.0f;
                            }
                            
                            // Train and measure loss
                            float lossBefore = computeLoss(net, inputs, targets);
                            net.trainBatch(inputs, targets);
                            float lossAfter = computeLoss(net, inputs, targets);
                            
                            lossHistory.add(lossAfter);
                            
                            // Check for gradient explosion
                            if (Float.isNaN(lossAfter) || Float.isInfinite(lossAfter)) {
                                gradientExploded.set(true);
                            }
                            
                            // Check for extreme loss increase (another sign of instability)
                            if (lossAfter > lossBefore * 10) {
                                gradientExploded.set(true);
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                        gradientExploded.set(true);
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }
            
            // Start all threads simultaneously for maximum contention
            startLatch.countDown();
            
            // Wait for completion
            assertTrue(doneLatch.await(30, TimeUnit.SECONDS), "Training took too long");
            executor.shutdown();
            
            // Check for gradient clipping warnings
            String errors = errCapture.toString();
            if (errors.contains("Large gradient norm")) {
                // Count extreme clipping events
                for (String line : errors.split("\n")) {
                    if (line.contains("Large gradient norm")) {
                        extremeClippingEvents.incrementAndGet();
                    }
                }
            }
            
            // Verify training stability
            assertFalse(gradientExploded.get(), "Gradients exploded during training");
            assertEquals(0, extremeClippingEvents.get(), 
                "Extreme gradient clipping detected - indicates race condition in optimizer");
            
            // Verify loss is decreasing on average
            if (lossHistory.size() > 10) {
                float avgFirstTen = 0;
                float avgLastTen = 0;
                
                for (int i = 0; i < 10; i++) {
                    avgFirstTen += lossHistory.get(i);
                    avgLastTen += lossHistory.get(lossHistory.size() - 10 + i);
                }
                avgFirstTen /= 10;
                avgLastTen /= 10;
                
                assertTrue(avgLastTen < avgFirstTen * 1.5, 
                    String.format("Loss increased significantly: %.4f -> %.4f", avgFirstTen, avgLastTen));
            }
            
        } finally {
            System.setErr(originalErr);
        }
    }
    
    @Test
    public void testTimeStepConsistency() throws Exception {
        // Test that concurrent training maintains optimizer state consistency
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .input(10)
            .layer(Layers.hiddenDenseRelu(20, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(5, optimizer));
            
        int numThreads = 10;
        int trainingsPerThread = 100;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);
        
        // Track if training remains stable
        AtomicBoolean stableTraining = new AtomicBoolean(true);
        AtomicInteger totalUpdates = new AtomicInteger(0);
        
        for (int t = 0; t < numThreads; t++) {
            executor.submit(() -> {
                try {
                    startLatch.await();
                    
                    for (int i = 0; i < trainingsPerThread; i++) {
                        // Same input/target for all threads
                        float[][] inputs = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
                        float[][] targets = {{0, 1, 0, 0, 0}};
                        
                        // Train and verify outputs remain valid
                        net.trainBatch(inputs, targets);
                        float[] output = net.predict(inputs[0]);
                        
                        // Check for NaN or invalid outputs
                        for (float val : output) {
                            if (Float.isNaN(val) || Float.isInfinite(val)) {
                                stableTraining.set(false);
                                return;
                            }
                        }
                        
                        totalUpdates.incrementAndGet();
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    stableTraining.set(false);
                } finally {
                    doneLatch.countDown();
                }
            });
        }
        
        startLatch.countDown();
        assertTrue(doneLatch.await(10, TimeUnit.SECONDS));
        executor.shutdown();
        
        // Verify training remained stable
        assertTrue(stableTraining.get(), "Training became unstable with concurrent updates");
        
        // Verify all updates completed
        assertEquals(numThreads * trainingsPerThread, totalUpdates.get(), 
            "Not all updates completed successfully");
    }
    
    private float computeLoss(NeuralNet net, float[][] inputs, float[][] targets) {
        float totalLoss = 0;
        for (int i = 0; i < inputs.length; i++) {
            float[] output = net.predict(inputs[i]);
            // Cross entropy loss
            for (int j = 0; j < output.length; j++) {
                if (targets[i][j] > 0) {
                    totalLoss -= (float) Math.log(Math.max(output[j], 1e-7));
                }
            }
        }
        return totalLoss / inputs.length;
    }
}