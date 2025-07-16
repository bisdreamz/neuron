package dev.neuronic.net.optimizers;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for AdamW optimizer implementation.
 */
class AdamWOptimizerTest {

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
        assertDoesNotThrow(() -> new AdamWOptimizer());
        assertDoesNotThrow(() -> new AdamWOptimizer(0.001f, 0.01f));
        assertDoesNotThrow(() -> new AdamWOptimizer(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f));
        
        // Invalid learning rate
        assertThrows(IllegalArgumentException.class, () -> new AdamWOptimizer(0, 0.01f));
        assertThrows(IllegalArgumentException.class, () -> new AdamWOptimizer(-0.001f, 0.01f));
        
        // Invalid weight decay
        assertThrows(IllegalArgumentException.class, () -> new AdamWOptimizer(0.001f, -0.01f));
        
        // Invalid beta parameters
        assertThrows(IllegalArgumentException.class, () -> new AdamWOptimizer(0.001f, -0.1f, 0.999f, 1e-8f, 0.01f));
        assertThrows(IllegalArgumentException.class, () -> new AdamWOptimizer(0.001f, 1.0f, 0.999f, 1e-8f, 0.01f));
    }

    @Test
    void testDefaultParameters() {
        AdamWOptimizer optimizer = new AdamWOptimizer();
        assertEquals(0.001f, optimizer.getLearningRate(), 1e-6f);
        assertEquals(0.01f, optimizer.getWeightDecay(), 1e-6f);
        assertEquals(0.9f, optimizer.getBeta1(), 1e-6f);
        assertEquals(0.999f, optimizer.getBeta2(), 1e-6f);
        assertEquals(1e-8f, optimizer.getEpsilon(), 1e-12f);
    }

    @Test
    void testWeightDecayEffect() {
        // Test that weight decay actually reduces weights
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.1f); // High weight decay for visible effect
        
        // Store original values
        float originalWeight = weights[0][0];
        float originalBias = biases[0];
        
        // Apply optimizer with zero gradients to isolate weight decay effect
        float[][] zeroWeightGradients = new float[][]{{0, 0}, {0, 0}};
        float[] zeroBiasGradients = new float[]{0, 0};
        
        optimizer.optimize(weights, biases, zeroWeightGradients, zeroBiasGradients);
        
        // Weights should be smaller due to weight decay (weights *= (1 - decay))
        assertTrue(weights[0][0] < originalWeight, "Weight should decrease due to weight decay");
        assertTrue(weights[0][1] < 2.0f, "Weight should decrease due to weight decay");
        
        // Biases should also decay in AdamW (matching PyTorch/TensorFlow behavior)
        assertTrue(biases[0] < originalBias, "Biases should also be affected by weight decay in AdamW");
    }

    @Test
    void testNoWeightDecay() {
        // Test with zero weight decay should behave like regular Adam
        AdamWOptimizer optimizerNoDecay = new AdamWOptimizer(0.01f, 0.0f);
        AdamOptimizer adamOptimizer = new AdamOptimizer(0.01f);
        
        // Create identical test data
        float[][] weights1 = new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}};
        float[] biases1 = new float[]{0.5f, 1.0f};
        float[][] weights2 = new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}};
        float[] biases2 = new float[]{0.5f, 1.0f};
        
        // Apply same gradients
        optimizerNoDecay.optimize(weights1, biases1, weightGradients, biasGradients);
        adamOptimizer.optimize(weights2, biases2, weightGradients, biasGradients);
        
        // Results should be very similar (small differences due to implementation details)
        assertEquals(weights1[0][0], weights2[0][0], 1e-4f, "AdamW with no decay should match Adam");
        assertEquals(weights1[0][1], weights2[0][1], 1e-4f, "AdamW with no decay should match Adam");
        assertEquals(biases1[0], biases2[0], 1e-4f, "AdamW with no decay should match Adam");
    }

    @Test
    void testRegularizationEffect() {
        // Test that AdamW provides better regularization than Adam
        AdamWOptimizer adamW = new AdamWOptimizer(0.01f, 0.01f);
        AdamOptimizer adam = new AdamOptimizer(0.01f);
        
        // Start with identical large weights
        float[][] weightsAdamW = new float[][]{{5.0f, 5.0f}};
        float[][] weightsAdam = new float[][]{{5.0f, 5.0f}};
        float[] biases1 = new float[]{0.0f};
        float[] biases2 = new float[]{0.0f};
        
        // Apply small gradients multiple times
        float[][] smallGradients = new float[][]{{0.01f, 0.01f}};
        float[] smallBiasGradients = new float[]{0.01f};
        
        for (int i = 0; i < 20; i++) {
            adamW.optimize(weightsAdamW, biases1, smallGradients, smallBiasGradients);
            adam.optimize(weightsAdam, biases2, smallGradients, smallBiasGradients);
        }
        
        // AdamW weights should be smaller due to weight decay regularization
        assertTrue(Math.abs(weightsAdamW[0][0]) < Math.abs(weightsAdam[0][0]), 
                  "AdamW should have smaller weights due to regularization");
        assertTrue(Math.abs(weightsAdamW[0][1]) < Math.abs(weightsAdam[0][1]), 
                  "AdamW should have smaller weights due to regularization");
    }

    @Test
    void testParameterUpdate() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        
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
        
        // Weights should have changed
        assertNotEquals(originalWeights[0][0], weights[0][0], 1e-6f);
        assertNotEquals(originalWeights[0][1], weights[0][1], 1e-6f);
        assertNotEquals(originalBiases[0], biases[0], 1e-6f);
        
        // All weights should have decreased (both from gradients and weight decay)
        assertTrue(weights[0][0] < originalWeights[0][0]);
        assertTrue(weights[0][1] < originalWeights[0][1]);
        assertTrue(biases[0] < originalBiases[0]); // Only gradient effect, no weight decay
    }

    @Test
    void testWeightDecayAppliedToBothWeightsAndBiases() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.1f); // Small learning rate to see weight decay effect
        
        float originalWeight = weights[0][0];
        float originalBias = biases[0];
        
        // Apply with zero gradients
        float[][] zeroGradients = new float[][]{{0, 0}, {0, 0}};
        float[] zeroBiasGradients = new float[]{0, 0};
        
        optimizer.optimize(weights, biases, zeroGradients, zeroBiasGradients);
        
        // Both weights and biases should decay (matching PyTorch/TensorFlow AdamW)
        assertTrue(weights[0][0] < originalWeight, "Weights should decay");
        assertTrue(biases[0] < originalBias, "Biases should also decay in AdamW");
    }

    @Test
    void testStatePersistence() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        
        // Multiple updates should accumulate momentum/velocity
        float initialWeight = weights[0][0];
        
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        float afterFirstUpdate = weights[0][0];
        
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
        float afterSecondUpdate = weights[0][0];
        
        // Each update should produce different changes due to accumulated state
        float firstChange = Math.abs(afterFirstUpdate - initialWeight);
        float secondChange = Math.abs(afterSecondUpdate - afterFirstUpdate);
        
        // Changes should be different (not exactly equal) due to momentum/velocity accumulation
        assertNotEquals(firstChange, secondChange, 1e-6f);
    }

    @Test
    void testAdamWConvergenceWithRegularization() {
        // Test that AdamW can still converge despite weight decay
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.01f);
        
        float[][] testWeights = new float[][]{{3.0f}};
        float[] testBiases = new float[]{2.0f};
        
        // Optimize toward smaller values with consistent gradients
        for (int i = 0; i < 100; i++) {
            float[][] grads = new float[][]{{testWeights[0][0] * 0.5f}}; // Gradient proportional to weight
            float[] biasGrads = new float[]{testBiases[0] * 0.5f};
            
            optimizer.optimize(testWeights, testBiases, grads, biasGrads);
        }
        
        // Should converge to smaller values (but not necessarily zero due to weight decay)
        assertTrue(Math.abs(testWeights[0][0]) < 1.0f, "Weight should converge to smaller value");
        assertTrue(Math.abs(testBiases[0]) < 1.0f, "Bias should converge to smaller value");
    }
}