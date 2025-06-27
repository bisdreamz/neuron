package dev.neuronic.net.optimizers;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests to verify that optimizers reuse buffers and minimize memory allocations.
 */
class BufferReuseTest {

    private float[][] weights;
    private float[] biases;
    private float[][] weightGradients;
    private float[] biasGradients;

    @BeforeEach
    void setUp() {
        // Create test data
        weights = new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}};
        biases = new float[]{0.5f, 1.0f};
        weightGradients = new float[][]{{0.1f, 0.2f}, {0.3f, 0.4f}};
        biasGradients = new float[]{0.05f, 0.1f};
    }

    @Test
    void testAdamBufferConsistency() {
        // Test that multiple calls with the same optimizer produce consistent results
        AdamOptimizer optimizer = new AdamOptimizer(0.01f);
        
        // Make first call to initialize state and buffers
        float[][] weights1 = copyWeights(weights);
        float[] biases1 = biases.clone();
        optimizer.optimize(weights1, biases1, weightGradients, biasGradients);
        
        // Make second call - should reuse buffers internally
        float[][] weights2 = copyWeights(weights);
        float[] biases2 = biases.clone();
        optimizer.optimize(weights2, biases2, weightGradients, biasGradients);
        
        // Results should be identical (deterministic updates)
        for (int i = 0; i < weights1.length; i++) {
            assertArrayEquals(weights1[i], weights2[i], 1e-6f, 
                "Buffer reuse should not affect deterministic results");
        }
        assertArrayEquals(biases1, biases2, 1e-6f, 
            "Buffer reuse should not affect bias results");
    }
    
    @Test
    void testAdamWBufferConsistency() {
        // Test that AdamW also reuses buffers correctly
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        
        // Make first call
        float[][] weights1 = copyWeights(weights);
        float[] biases1 = biases.clone();
        optimizer.optimize(weights1, biases1, weightGradients, biasGradients);
        
        // Make second call - should reuse buffers internally  
        float[][] weights2 = copyWeights(weights);
        float[] biases2 = biases.clone();
        optimizer.optimize(weights2, biases2, weightGradients, biasGradients);
        
        // Results should be identical
        for (int i = 0; i < weights1.length; i++) {
            assertArrayEquals(weights1[i], weights2[i], 1e-6f, 
                "AdamW buffer reuse should not affect results");
        }
        assertArrayEquals(biases1, biases2, 1e-6f, 
            "AdamW buffer reuse should not affect bias results");
    }
    
    @Test
    void testMultipleLayersIndependentBuffers() {
        // Test that different layers get independent buffer sets
        AdamOptimizer optimizer = new AdamOptimizer(0.01f);
        
        // Create second layer with different dimensions
        float[][] weights2 = new float[][]{{5.0f, 6.0f, 7.0f}, {8.0f, 9.0f, 10.0f}};
        float[] biases2 = new float[]{1.5f, 2.0f, 2.5f};
        float[][] weightGradients2 = new float[][]{{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}};
        float[] biasGradients2 = new float[]{0.05f, 0.1f, 0.15f};
        
        // Store original values
        float originalWeights1 = weights[0][0];
        float originalWeights2 = weights2[0][0];
        
        // Optimize both layers
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        optimizer.optimize(weights2, biases2, weightGradients2, biasGradients2);
        
        // Both should have changed from their original values
        assertNotEquals(originalWeights1, weights[0][0], 1e-6f, 
            "First layer should be updated");
        assertNotEquals(originalWeights2, weights2[0][0], 1e-6f, 
            "Second layer should be updated independently");
    }
    
    @Test
    void testStateResetDoesNotAffectBuffers() {
        // Test that state management doesn't interfere with buffer reuse
        AdamOptimizer optimizer = new AdamOptimizer(0.01f);
        
        // Create arrays of different sizes to test buffer size adaptation
        float[][] smallWeights = new float[][]{{1.0f}};
        float[] smallBiases = new float[]{0.5f};
        float[][] smallWeightGradients = new float[][]{{0.1f}};
        float[] smallBiasGradients = new float[]{0.05f};
        
        float[][] largeWeights = new float[][]{{1.0f, 2.0f, 3.0f, 4.0f}};
        float[] largeBiases = new float[]{0.5f, 1.0f, 1.5f, 2.0f};
        float[][] largeWeightGradients = new float[][]{{0.1f, 0.2f, 0.3f, 0.4f}};
        float[] largeBiasGradients = new float[]{0.05f, 0.1f, 0.15f, 0.2f};
        
        // Optimize small arrays first
        assertDoesNotThrow(() -> {
            optimizer.optimize(smallWeights, smallBiases, smallWeightGradients, smallBiasGradients);
        }, "Small array optimization should work");
        
        // Then optimize large arrays - buffers should handle the size change
        assertDoesNotThrow(() -> {
            optimizer.optimize(largeWeights, largeBiases, largeWeightGradients, largeBiasGradients);
        }, "Large array optimization should work with existing buffers");
        
        // Go back to small arrays - should still work
        assertDoesNotThrow(() -> {
            optimizer.optimize(smallWeights, smallBiases, smallWeightGradients, smallBiasGradients);
        }, "Small array optimization should still work after large arrays");
    }
    
    @Test
    void testThreadSafetyOfBuffers() {
        // Test that buffers are thread-safe (each thread gets its own)
        AdamOptimizer optimizer = new AdamOptimizer(0.01f);
        
        // Create large weights to trigger parallelization
        float[][] largeWeights = new float[3000][100]; // > 2048 threshold
        float[] largeBiases = new float[100];
        float[][] largeWeightGradients = new float[3000][100];
        float[] largeBiasGradients = new float[100];
        
        // Initialize with small values
        for (int i = 0; i < largeWeights.length; i++) {
            for (int j = 0; j < largeWeights[i].length; j++) {
                largeWeights[i][j] = 0.1f;
                largeWeightGradients[i][j] = 0.01f;
            }
        }
        for (int j = 0; j < largeBiases.length; j++) {
            largeBiases[j] = 0.1f;
            largeBiasGradients[j] = 0.01f;
        }
        
        // This should use parallel execution with thread-local buffers
        assertDoesNotThrow(() -> {
            optimizer.optimize(largeWeights, largeBiases, largeWeightGradients, largeBiasGradients);
        }, "Parallel optimization with thread-local buffers should work");
        
        // Verify optimization occurred
        assertTrue(largeWeights[0][0] < 0.1f, "Weights should have been updated");
    }
    
    private float[][] copyWeights(float[][] original) {
        float[][] copy = new float[original.length][];
        for (int i = 0; i < original.length; i++) {
            copy[i] = original[i].clone();
        }
        return copy;
    }
}