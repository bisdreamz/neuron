package dev.neuronic.net.optimizers;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SgdOptimizerTest {
    
    private static final float DELTA = 1e-6f;
    
    @Test
    void testBasicOptimization() {
        SgdOptimizer sgd = new SgdOptimizer(0.1f);
        
        float[][] weights = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        float[] biases = {0.5f, 1.5f};
        float[][] weightGradients = {{0.1f, 0.2f}, {0.3f, 0.4f}};
        float[] biasGradients = {0.05f, 0.15f};
        
        sgd.optimize(weights, biases, weightGradients, biasGradients);
        
        // weights = weights - lr * gradients
        assertArrayEquals(new float[]{0.99f, 1.98f}, weights[0], DELTA);
        assertArrayEquals(new float[]{2.97f, 3.96f}, weights[1], DELTA);
        
        // biases = biases - lr * gradients  
        assertArrayEquals(new float[]{0.495f, 1.485f}, biases, DELTA);
    }
    
    @Test
    void testGetLearningRate() {
        SgdOptimizer sgd = new SgdOptimizer(0.05f);
        assertEquals(0.05f, sgd.getLearningRate(), DELTA);
    }
    
    @Test
    void testZeroLearningRate() {
        SgdOptimizer sgd = new SgdOptimizer(0.0001f); // Very small but positive
        
        float[][] originalWeights = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        float[] originalBiases = {0.5f, 1.5f};
        
        float[][] weights = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        float[] biases = {0.5f, 1.5f};
        float[][] weightGradients = {{1.0f, 1.0f}, {1.0f, 1.0f}};
        float[] biasGradients = {1.0f, 1.0f};
        
        sgd.optimize(weights, biases, weightGradients, biasGradients);
        
        // Very small changes
        assertTrue(Math.abs(weights[0][0] - originalWeights[0][0]) < 0.001f);
        assertTrue(Math.abs(biases[0] - originalBiases[0]) < 0.001f);
    }
    
    @Test
    void testNegativeGradients() {
        SgdOptimizer sgd = new SgdOptimizer(0.1f);
        
        float[][] weights = {{1.0f, 2.0f}};
        float[] biases = {0.5f};
        float[][] weightGradients = {{-0.1f, -0.2f}};
        float[] biasGradients = {-0.05f};
        
        sgd.optimize(weights, biases, weightGradients, biasGradients);
        
        // Negative gradients should increase parameters
        assertArrayEquals(new float[]{1.01f, 2.02f}, weights[0], DELTA);
        assertEquals(0.505f, biases[0], DELTA);
    }
    
    @Test
    void testMultipleUpdates() {
        SgdOptimizer sgd = new SgdOptimizer(0.1f);
        
        float[][] weights = {{1.0f}};
        float[] biases = {0.0f};
        float[][] weightGradients = {{0.1f}};
        float[] biasGradients = {0.1f};
        
        // First update
        sgd.optimize(weights, biases, weightGradients, biasGradients);
        assertEquals(0.99f, weights[0][0], DELTA);
        assertEquals(-0.01f, biases[0], DELTA);
        
        // Second update
        sgd.optimize(weights, biases, weightGradients, biasGradients);
        assertEquals(0.98f, weights[0][0], DELTA);
        assertEquals(-0.02f, biases[0], DELTA);
    }
    
    @Test
    void testInvalidLearningRate() {
        assertThrows(IllegalArgumentException.class, () -> 
            new SgdOptimizer(0.0f));
        assertThrows(IllegalArgumentException.class, () -> 
            new SgdOptimizer(-0.1f));
    }
    
    @Test
    void testWeightDimensionMismatch() {
        SgdOptimizer sgd = new SgdOptimizer(0.1f);
        
        float[][] weights = {{1.0f, 2.0f}};
        float[] biases = {0.5f};
        float[][] weightGradients = {{0.1f, 0.2f}, {0.3f, 0.4f}}; // Extra row
        float[] biasGradients = {0.05f};
        
        assertThrows(IllegalArgumentException.class, () -> 
            sgd.optimize(weights, biases, weightGradients, biasGradients));
    }
    
    @Test
    void testBiasDimensionMismatch() {
        SgdOptimizer sgd = new SgdOptimizer(0.1f);
        
        float[][] weights = {{1.0f, 2.0f}};
        float[] biases = {0.5f};
        float[][] weightGradients = {{0.1f, 0.2f}};
        float[] biasGradients = {0.05f, 0.15f}; // Extra element
        
        assertThrows(IllegalArgumentException.class, () -> 
            sgd.optimize(weights, biases, weightGradients, biasGradients));
    }
    
    @Test
    void testLargeArrays() {
        SgdOptimizer sgd = new SgdOptimizer(0.01f);
        
        // Test with larger arrays to verify vectorization path
        int rows = 64, cols = 64;
        float[][] weights = new float[rows][cols];
        float[] biases = new float[cols];
        float[][] weightGradients = new float[rows][cols];
        float[] biasGradients = new float[cols];
        
        // Initialize with known values
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[i][j] = 1.0f;
                weightGradients[i][j] = 0.1f;
            }
        }
        for (int j = 0; j < cols; j++) {
            biases[j] = 0.5f;
            biasGradients[j] = 0.05f;
        }
        
        sgd.optimize(weights, biases, weightGradients, biasGradients);
        
        // Verify all elements updated correctly
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                assertEquals(0.999f, weights[i][j], DELTA);
            }
        }
        for (int j = 0; j < cols; j++) {
            assertEquals(0.4995f, biases[j], DELTA);
        }
    }
}