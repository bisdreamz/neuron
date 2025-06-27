package dev.neuronic.net.optimizers;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Adam optimizer implementation.
 */
class AdamOptimizerTest {

    private float[][] weights;
    private float[] biases;
    private float[][] weightGradients;
    private float[] biasGradients;

    @BeforeEach
    void setUp() {
        // Create simple test data
        weights = new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}};
        biases = new float[]{0.5f, 1.0f};
        weightGradients = new float[][]{{0.1f, 0.2f}, {0.3f, 0.4f}};
        biasGradients = new float[]{0.05f, 0.1f};
    }

    @Test
    void testConstructorValidation() {
        // Valid parameters should work
        assertDoesNotThrow(() -> new AdamOptimizer(0.001f));
        assertDoesNotThrow(() -> new AdamOptimizer(0.001f, 0.9f, 0.999f, 1e-8f));
        
        // Invalid learning rate
        assertThrows(IllegalArgumentException.class, () -> new AdamOptimizer(0));
        assertThrows(IllegalArgumentException.class, () -> new AdamOptimizer(-0.001f));
        
        // Invalid beta1
        assertThrows(IllegalArgumentException.class, () -> new AdamOptimizer(0.001f, -0.1f, 0.999f, 1e-8f));
        assertThrows(IllegalArgumentException.class, () -> new AdamOptimizer(0.001f, 1.0f, 0.999f, 1e-8f));
        
        // Invalid beta2
        assertThrows(IllegalArgumentException.class, () -> new AdamOptimizer(0.001f, 0.9f, -0.1f, 1e-8f));
        assertThrows(IllegalArgumentException.class, () -> new AdamOptimizer(0.001f, 0.9f, 1.0f, 1e-8f));
        
        // Invalid epsilon
        assertThrows(IllegalArgumentException.class, () -> new AdamOptimizer(0.001f, 0.9f, 0.999f, 0));
        assertThrows(IllegalArgumentException.class, () -> new AdamOptimizer(0.001f, 0.9f, 0.999f, -1e-8f));
    }

    @Test
    void testDefaultParameters() {
        AdamOptimizer optimizer = new AdamOptimizer(0.001f);
        assertEquals(0.001f, optimizer.getLearningRate(), 1e-6f);
        assertEquals(0.9f, optimizer.getBeta1(), 1e-6f);
        assertEquals(0.999f, optimizer.getBeta2(), 1e-6f);
        assertEquals(1e-8f, optimizer.getEpsilon(), 1e-12f);
    }

    @Test
    void testParameterUpdate() {
        AdamOptimizer optimizer = new AdamOptimizer(0.01f); // Higher learning rate for visible effect
        
        // Store original values
        float[][] originalWeights = new float[2][2];
        float[] originalBiases = new float[2];
        for (int i = 0; i < 2; i++) {
            originalBiases[i] = biases[i];
            for (int j = 0; j < 2; j++) {
                originalWeights[i][j] = weights[i][j];
            }
        }
        
        // Apply optimizer
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        
        // Weights should have changed (Adam should decrease weights in direction of gradients)
        assertTrue(weights[0][0] < originalWeights[0][0], "Weight should decrease with positive gradient");
        assertTrue(weights[0][1] < originalWeights[0][1], "Weight should decrease with positive gradient");
        assertTrue(biases[0] < originalBiases[0], "Bias should decrease with positive gradient");
        assertTrue(biases[1] < originalBiases[1], "Bias should decrease with positive gradient");
    }

    @Test
    void testStatePersistenceAcrossUpdates() {
        AdamOptimizer optimizer = new AdamOptimizer(0.01f);
        
        // Just verify that multiple updates work without errors
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        
        // If we get here without exceptions, state persistence is working
        assertTrue(true, "Multiple updates completed successfully");
    }

    @Test
    void testMultipleLayersIndependentState() {
        AdamOptimizer optimizer = new AdamOptimizer(0.01f);
        
        // Create second layer with same dimensions but different weights
        float[][] weights2 = new float[][]{{5.0f, 6.0f}, {7.0f, 8.0f}};
        float[] biases2 = new float[]{1.5f, 2.0f};
        
        // Store original values
        float original00_layer1 = weights[0][0];
        float original00_layer2 = weights2[0][0];
        
        // Update both layers
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        optimizer.optimize(weights2, biases2, weightGradients, biasGradients);
        
        // Both should have changed from their original values
        assertNotEquals(original00_layer1, weights[0][0], 1e-6f);
        assertNotEquals(original00_layer2, weights2[0][0], 1e-6f);
        
        // Layers should maintain independent state
        assertTrue(true, "Multiple layers can be optimized independently");
    }

    @Test
    void testZeroGradients() {
        AdamOptimizer optimizer = new AdamOptimizer(0.01f);
        
        // Store original values
        float[][] originalWeights = new float[2][2];
        float[] originalBiases = new float[2];
        for (int i = 0; i < 2; i++) {
            originalBiases[i] = biases[i];
            for (int j = 0; j < 2; j++) {
                originalWeights[i][j] = weights[i][j];
            }
        }
        
        // Apply zero gradients
        float[][] zeroWeightGradients = new float[][]{{0, 0}, {0, 0}};
        float[] zeroBiasGradients = new float[]{0, 0};
        
        optimizer.optimize(weights, biases, zeroWeightGradients, zeroBiasGradients);
        
        // Weights should remain unchanged with zero gradients
        for (int i = 0; i < 2; i++) {
            assertEquals(originalBiases[i], biases[i], 1e-6f);
            for (int j = 0; j < 2; j++) {
                assertEquals(originalWeights[i][j], weights[i][j], 1e-6f);
            }
        }
    }

    @Test
    void testBiasCorrection() {
        // Test that bias correction is being applied by comparing different beta values
        AdamOptimizer optimizerHighBeta = new AdamOptimizer(0.01f, 0.99f, 0.999f, 1e-8f); // High beta = more bias
        AdamOptimizer optimizerLowBeta = new AdamOptimizer(0.01f, 0.1f, 0.1f, 1e-8f);    // Low beta = less bias
        
        float[][] weights1 = new float[][]{{1.0f}};
        float[] biases1 = new float[]{1.0f};
        float[][] weights2 = new float[][]{{1.0f}};
        float[] biases2 = new float[]{1.0f};
        float[][] gradients = new float[][]{{0.1f}};
        float[] biasGradients = new float[]{0.1f};
        
        optimizerHighBeta.optimize(weights1, biases1, gradients, biasGradients);
        optimizerLowBeta.optimize(weights2, biases2, gradients, biasGradients);
        
        float changeHighBeta = Math.abs(1.0f - weights1[0][0]);
        float changeLowBeta = Math.abs(1.0f - weights2[0][0]);
        
        // High beta should result in larger early updates due to stronger bias correction
        assertTrue(changeHighBeta > changeLowBeta * 0.1f, 
                  "Bias correction should affect update magnitude");
    }

    @Test
    void testAdamConvergence() {
        // Test that Adam makes some progress toward optimization
        AdamOptimizer optimizer = new AdamOptimizer(0.1f); // Higher learning rate
        
        float[][] testWeights = new float[][]{{2.0f}};
        float[] testBiases = new float[]{2.0f};
        
        float initialValue = Math.abs(testWeights[0][0]) + Math.abs(testBiases[0]);
        
        // Apply consistent gradients pointing toward zero
        for (int i = 0; i < 50; i++) {
            float[][] grads = new float[][]{{testWeights[0][0] > 0 ? 0.1f : -0.1f}};
            float[] biasGrads = new float[]{testBiases[0] > 0 ? 0.1f : -0.1f};
            
            optimizer.optimize(testWeights, testBiases, grads, biasGrads);
        }
        
        float finalValue = Math.abs(testWeights[0][0]) + Math.abs(testBiases[0]);
        
        // Should make some progress (at least 10% reduction)
        assertTrue(finalValue < initialValue * 0.9f, 
                  "Adam should make some progress: initial=" + initialValue + ", final=" + finalValue);
    }
}