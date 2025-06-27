package dev.neuronic.net.optimizers;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests to verify that Adam/AdamW optimizers perform zero allocations during training.
 * This test ensures we follow the codebase standards of using ThreadLocal buffers.
 */
class ZeroAllocationTest {

    private float[][] weights;
    private float[] biases;
    private float[][] weightGradients;
    private float[] biasGradients;

    @BeforeEach
    void setUp() {
        // Create test data
        weights = new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
        biases = new float[]{0.1f, 0.2f, 0.3f};
        weightGradients = new float[][]{{0.01f, 0.02f, 0.03f}, {0.04f, 0.05f, 0.06f}};
        biasGradients = new float[]{0.001f, 0.002f, 0.003f};
    }

    @Test
    void testAdamZeroAllocationAfterWarmup() {
        AdamOptimizer optimizer = new AdamOptimizer(0.01f);
        
        // Warm up - this will allocate buffers initially
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        
        // Force garbage collection to clean up any allocations
        System.gc();
        Thread.yield();
        
        // Get memory usage before optimization
        Runtime runtime = Runtime.getRuntime();
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
        
        // Perform many optimization steps
        for (int i = 0; i < 100; i++) {
            optimizer.optimize(weights, biases, weightGradients, biasGradients);
        }
        
        // Force garbage collection again
        System.gc();
        Thread.yield();
        
        // Get memory usage after optimization
        long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
        long memoryIncrease = memoryAfter - memoryBefore;
        
        // Memory increase should be minimal (allowing for GC overhead and measurement noise)
        assertTrue(memoryIncrease < 10_000, // 10KB threshold
            "Memory increase after 100 optimizations should be minimal. Actual increase: " + memoryIncrease + " bytes");
    }
    
    @Test
    void testAdamWZeroAllocationAfterWarmup() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        
        // Warm up
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        
        System.gc();
        Thread.yield();
        
        Runtime runtime = Runtime.getRuntime();
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
        
        // Perform many optimization steps
        for (int i = 0; i < 100; i++) {
            optimizer.optimize(weights, biases, weightGradients, biasGradients);
        }
        
        System.gc();
        Thread.yield();
        
        long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
        long memoryIncrease = memoryAfter - memoryBefore;
        
        assertTrue(memoryIncrease < 10_000, // 10KB threshold
            "AdamW memory increase after 100 optimizations should be minimal. Actual increase: " + memoryIncrease + " bytes");
    }
    
    @Test
    void testMultipleLayersShareNoState() {
        // Test that different layers get independent buffers but don't allocate during optimization
        AdamOptimizer optimizer = new AdamOptimizer(0.01f);
        
        // Create second layer with different dimensions
        float[][] weights2 = new float[][]{{10.0f, 11.0f}, {12.0f, 13.0f}, {14.0f, 15.0f}};
        float[] biases2 = new float[]{1.0f, 2.0f};
        float[][] weightGradients2 = new float[][]{{0.1f, 0.2f}, {0.3f, 0.4f}, {0.5f, 0.6f}};
        float[] biasGradients2 = new float[]{0.01f, 0.02f};
        
        // Warm up both layers
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        optimizer.optimize(weights2, biases2, weightGradients2, biasGradients2);
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        optimizer.optimize(weights2, biases2, weightGradients2, biasGradients2);
        
        System.gc();
        Thread.yield();
        
        Runtime runtime = Runtime.getRuntime();
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
        
        // Alternate between layers many times
        for (int i = 0; i < 50; i++) {
            optimizer.optimize(weights, biases, weightGradients, biasGradients);
            optimizer.optimize(weights2, biases2, weightGradients2, biasGradients2);
        }
        
        System.gc();
        Thread.yield();
        
        long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
        long memoryIncrease = memoryAfter - memoryBefore;
        
        assertTrue(memoryIncrease < 15_000, // 15KB threshold for two layers
            "Multi-layer memory increase should be minimal. Actual increase: " + memoryIncrease + " bytes");
    }
    
    @Test
    void testBuffersScaleWithLayerSize() {
        // Test that buffers are properly sized for different layer dimensions
        AdamOptimizer optimizer = new AdamOptimizer(0.01f);
        
        // Start with small layer
        float[][] smallWeights = new float[][]{{1.0f}};
        float[] smallBiases = new float[]{0.1f};
        float[][] smallWeightGradients = new float[][]{{0.01f}};
        float[] smallBiasGradients = new float[]{0.001f};
        
        // Then large layer
        float[][] largeWeights = new float[5][10];
        float[] largeBiases = new float[10];
        float[][] largeWeightGradients = new float[5][10];
        float[] largeBiasGradients = new float[10];
        
        // Initialize large arrays
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 10; j++) {
                largeWeights[i][j] = i + j;
                largeWeightGradients[i][j] = 0.01f;
            }
        }
        for (int j = 0; j < 10; j++) {
            largeBiases[j] = j * 0.1f;
            largeBiasGradients[j] = 0.001f;
        }
        
        // Both should work without errors (buffers should scale appropriately)
        assertDoesNotThrow(() -> {
            optimizer.optimize(smallWeights, smallBiases, smallWeightGradients, smallBiasGradients);
            optimizer.optimize(largeWeights, largeBiases, largeWeightGradients, largeBiasGradients);
            optimizer.optimize(smallWeights, smallBiases, smallWeightGradients, smallBiasGradients);
        }, "Optimizers should handle layers of different sizes");
    }
    
    @Test
    void testOptimizedMethodsUseBuffers() {
        // This test verifies that the optimized methods use reusable buffers correctly
        // by ensuring state objects contain properly initialized buffer containers
        AdamOptimizer optimizer = new AdamOptimizer(0.01f);
        
        // First optimization call initializes state and buffers
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        
        // Multiple subsequent calls should reuse the same buffers
        // We verify this by checking that operations complete without errors
        // and that memory usage stays stable
        
        Runtime runtime = Runtime.getRuntime();
        System.gc();
        Thread.yield();
        
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
        
        // Perform many optimization steps
        for (int i = 0; i < 50; i++) {
            optimizer.optimize(weights, biases, weightGradients, biasGradients);
        }
        
        System.gc();
        Thread.yield();
        
        long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
        long memoryIncrease = memoryAfter - memoryBefore;
        
        // Memory should be stable after buffer initialization
        assertTrue(memoryIncrease < 5_000, // 5KB threshold for noise tolerance
            "Memory should be stable after buffer initialization. Increase: " + memoryIncrease + " bytes");
        
        // Verify that all operations complete successfully (no exceptions from buffer reuse)
        assertDoesNotThrow(() -> {
            for (int i = 0; i < 10; i++) {
                optimizer.optimize(weights, biases, weightGradients, biasGradients);
            }
        }, "Buffer reuse should not cause any exceptions");
    }
}