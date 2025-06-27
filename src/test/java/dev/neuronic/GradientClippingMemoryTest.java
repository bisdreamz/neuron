package dev.neuronic;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.common.PooledFloatArray;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class GradientClippingMemoryTest {
    
    @Test
    public void testThreadLocalBuffersNoMemoryLeak() throws Exception {
        // Test that ThreadLocal buffers used by gradient computation don't leak memory
        // This replaces the old buffer pool test since we now use ThreadLocal buffers
        
        // Create a network with gradient clipping
        NeuralNet net = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
            .withGlobalGradientClipping(1.0f)
            .layer(Layers.hiddenDenseRelu(20))
            .layer(Layers.hiddenDenseRelu(15))
            .output(Layers.outputSoftmaxCrossEntropy(5));
        
        // Track memory usage before training
        System.gc();
        Thread.sleep(100);
        long memoryBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        
        // Train from multiple threads to test ThreadLocal buffer management
        int numThreads = 4;
        Thread[] threads = new Thread[numThreads];
        for (int t = 0; t < numThreads; t++) {
            threads[t] = new Thread(() -> {
                float[] input = new float[10];
                float[] target = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
                
                for (int iter = 0; iter < 100; iter++) {
                    for (int j = 0; j < 10; j++) {
                        input[j] = (float) Math.random();
                    }
                    net.train(input, target);
                }
            });
            threads[t].start();
        }
        
        // Wait for all threads to complete
        for (Thread thread : threads) {
            thread.join();
        }
        
        // Allow ThreadLocal cleanup
        System.gc();
        Thread.sleep(100);
        long memoryAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        
        // Memory increase should be reasonable (< 10MB for ThreadLocal buffers)
        long memoryIncrease = memoryAfter - memoryBefore;
        assertTrue(memoryIncrease < 10 * 1024 * 1024, 
            "Memory increase too large: " + (memoryIncrease / 1024 / 1024) + " MB");
    }
    
    @Test
    public void testNoMemoryLeakWithManyNetworks() {
        // Test that creating many networks doesn't cause memory leaks
        
        System.gc();
        long memoryBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        
        // Create many networks with same layer sizes
        for (int i = 0; i < 100; i++) {
            NeuralNet net = NeuralNet.newBuilder()
                .input(10)
                .setDefaultOptimizer(new SgdOptimizer(0.1f))
                .withGlobalGradientClipping(1.0f)
                .layer(Layers.hiddenDenseRelu(32))
                .layer(Layers.hiddenDenseRelu(32))
                .output(Layers.outputSoftmaxCrossEntropy(10));
            
            // Train once to trigger buffer allocation
            net.train(new float[10], new float[10]);
        }
        
        System.gc();
        long memoryAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        
        // Memory increase should be reasonable (< 50MB for 100 networks)
        long memoryIncrease = memoryAfter - memoryBefore;
        assertTrue(memoryIncrease < 50 * 1024 * 1024, 
            "Memory increase too large: " + (memoryIncrease / 1024 / 1024) + " MB");
    }
    
    @Test
    public void testBufferPoolThreadSafety() throws Exception {
        // Create network with gradient clipping
        NeuralNet net = NeuralNet.newBuilder()
            .input(50)
            .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
            .withGlobalGradientClipping(1.0f)
            .layer(Layers.hiddenDenseRelu(100))
            .layer(Layers.hiddenDenseRelu(50))
            .output(Layers.outputSoftmaxCrossEntropy(10));
        
        // Run parallel training from multiple threads
        int numThreads = 8;
        Thread[] threads = new Thread[numThreads];
        boolean[] errors = new boolean[numThreads];
        
        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            threads[t] = new Thread(() -> {
                try {
                    float[] input = new float[50];
                    float[] target = new float[10];
                    target[threadId % 10] = 1.0f;
                    
                    for (int iter = 0; iter < 100; iter++) {
                        // Random input
                        for (int i = 0; i < input.length; i++) {
                            input[i] = (float) Math.random();
                        }
                        
                        net.train(input, target);
                        
                        // Verify no NaN
                        float[] pred = net.predict(input);
                        for (float p : pred) {
                            if (Float.isNaN(p) || Float.isInfinite(p)) {
                                errors[threadId] = true;
                                return;
                            }
                        }
                    }
                } catch (Exception e) {
                    errors[threadId] = true;
                    e.printStackTrace();
                }
            });
            threads[t].start();
        }
        
        // Wait for all threads
        for (Thread thread : threads) {
            thread.join();
        }
        
        // Check no errors
        for (int t = 0; t < numThreads; t++) {
            assertFalse(errors[t], "Thread " + t + " encountered an error");
        }
    }
    
    @Test
    public void testBufferContentIsolation() {
        // Test that buffers returned to pool don't contaminate future uses
        PooledFloatArray pool = new PooledFloatArray(100);
        
        // Get buffer, modify it, return it
        float[] buffer1 = pool.getBuffer(false);
        for (int i = 0; i < buffer1.length; i++) {
            buffer1[i] = 999.0f; // Contaminate
        }
        pool.releaseBuffer(buffer1);
        
        // Get buffer with zeroing - should be clean
        float[] buffer2 = pool.getBuffer(true);
        for (int i = 0; i < buffer2.length; i++) {
            assertEquals(0.0f, buffer2[i], "Buffer should be zeroed");
        }
        pool.releaseBuffer(buffer2);
        
        // Get buffer without zeroing - will have old data but that's OK
        // because applyAccumulatedGradients immediately overwrites with arraycopy
        float[] buffer3 = pool.getBuffer(false);
        // This buffer might have old data, but that's fine since we copy over it
        pool.releaseBuffer(buffer3);
    }
}