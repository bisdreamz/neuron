package dev.neuronic.net.losses;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class HuberLossTest {
    
    private static final float EPSILON = 1e-6f;
    
    @Test
    public void testSmallErrors() {
        // For small errors (|error| <= delta), Huber loss = 0.5 * error^2
        HuberLoss loss = HuberLoss.createDefault(); // delta = 1.0
        
        float[] predictions = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] targets = {1.2f, 2.3f, 2.7f, 3.5f};
        
        // Errors: -0.2, -0.3, 0.3, 0.5 (all <= 1.0)
        // Expected loss: 0.5 * (0.04 + 0.09 + 0.09 + 0.25) / 4 = 0.05875
        
        float actualLoss = loss.loss(predictions, targets);
        assertEquals(0.05875f, actualLoss, EPSILON);
    }
    
    @Test
    public void testLargeErrors() {
        // For large errors (|error| > delta), Huber loss = delta * |error| - 0.5 * delta^2
        HuberLoss loss = HuberLoss.createDefault(); // delta = 1.0
        
        float[] predictions = {1.0f, 5.0f};
        float[] targets = {3.5f, 1.0f};
        
        // Errors: -2.5, 4.0 (both > 1.0)
        // Loss for -2.5: 1.0 * 2.5 - 0.5 * 1.0 = 2.0
        // Loss for 4.0: 1.0 * 4.0 - 0.5 * 1.0 = 3.5
        // Average: (2.0 + 3.5) / 2 = 2.75
        
        float actualLoss = loss.loss(predictions, targets);
        assertEquals(2.75f, actualLoss, EPSILON);
    }
    
    @Test
    public void testMixedErrors() {
        HuberLoss loss = HuberLoss.create(1.5f); // delta = 1.5
        
        float[] predictions = {1.0f, 2.0f, 3.0f, 10.0f};
        float[] targets = {2.0f, 2.5f, 0.0f, 12.0f};
        
        // Errors: -1.0, -0.5, 3.0, -2.0
        // |Errors|: 1.0, 0.5, 3.0, 2.0
        // Small (<=1.5): 1.0, 0.5 → 0.5 * (1.0 + 0.25) = 0.625
        // Large (>1.5): 3.0, 2.0 → (1.5*3.0 - 0.5*1.5^2) + (1.5*2.0 - 0.5*1.5^2) = 3.375 + 1.875 = 5.25
        // Total: (0.625 + 5.25) / 4 = 1.46875
        
        float actualLoss = loss.loss(predictions, targets);
        assertEquals(1.46875f, actualLoss, EPSILON);
    }
    
    @Test
    public void testDerivatives() {
        HuberLoss loss = HuberLoss.createDefault(); // delta = 1.0
        
        float[] predictions = {1.0f, 2.0f, 3.0f, 5.0f};
        float[] targets = {1.5f, 4.0f, 2.0f, 1.0f};
        
        // Errors: -0.5, -2.0, 1.0, 4.0
        // Expected derivatives:
        // |error| <= 1.0: derivative = error
        // |error| > 1.0: derivative = delta * sign(error)
        // So: -0.5, -1.0, 1.0, 1.0
        
        float[] derivatives = loss.derivatives(predictions, targets);
        
        assertEquals(-0.5f, derivatives[0], EPSILON);
        assertEquals(-1.0f, derivatives[1], EPSILON);
        assertEquals(1.0f, derivatives[2], EPSILON);
        assertEquals(1.0f, derivatives[3], EPSILON);
    }
    
    @Test
    public void testZeroError() {
        HuberLoss loss = HuberLoss.createDefault();
        
        float[] predictions = {1.0f, 2.0f, 3.0f};
        float[] targets = {1.0f, 2.0f, 3.0f};
        
        assertEquals(0.0f, loss.loss(predictions, targets), EPSILON);
        
        float[] derivatives = loss.derivatives(predictions, targets);
        for (float d : derivatives) {
            assertEquals(0.0f, d, EPSILON);
        }
    }
    
    @Test
    public void testDifferentDeltas() {
        float[] predictions = {1.0f, 2.0f};
        float[] targets = {3.0f, 4.0f}; // Both errors = -2.0
        
        // Delta = 0.5: Both errors are large
        HuberLoss loss1 = HuberLoss.create(0.5f);
        float expected1 = 0.5f * 2.0f - 0.5f * 0.5f * 0.5f; // 1.0 - 0.125 = 0.875 per element
        assertEquals(0.875f, loss1.loss(predictions, targets), EPSILON);
        
        // Delta = 3.0: Both errors are small
        HuberLoss loss2 = HuberLoss.create(3.0f);
        float expected2 = 0.5f * 2.0f * 2.0f; // 0.5 * 4 = 2.0 per element
        assertEquals(2.0f, loss2.loss(predictions, targets), EPSILON);
    }
    
    @Test
    public void testInvalidDelta() {
        assertThrows(IllegalArgumentException.class, () -> HuberLoss.create(0.0f));
        assertThrows(IllegalArgumentException.class, () -> HuberLoss.create(-1.0f));
    }
    
    @Test
    public void testFactoryMethods() {
        HuberLoss default1 = HuberLoss.createDefault();
        HuberLoss default2 = HuberLoss.createDefault();
        assertNotNull(default1);
        assertNotNull(default2);
        assertEquals(1.0f, default1.getDelta(), EPSILON);
        
        HuberLoss custom = HuberLoss.create(2.5f);
        assertNotNull(custom);
        assertEquals(2.5f, custom.getDelta(), EPSILON);
    }
    
    @Test
    public void testLargeArrayPerformance() {
        // Test that vectorized path works correctly on large arrays
        HuberLoss loss = HuberLoss.createDefault();
        
        int size = 10000;
        float[] predictions = new float[size];
        float[] targets = new float[size];
        
        // Create data with mix of small and large errors
        for (int i = 0; i < size; i++) {
            predictions[i] = (float)(Math.random() * 10 - 5);
            targets[i] = predictions[i] + (float)(Math.random() * 4 - 2);
        }
        
        // Compute loss
        float lossValue = loss.loss(predictions, targets);
        assertTrue(lossValue >= 0, "Loss should be non-negative");
        assertTrue(lossValue < 10, "Loss seems too large");
        
        // Compute derivatives
        float[] derivatives = loss.derivatives(predictions, targets);
        assertEquals(size, derivatives.length);
        
        // Spot check some derivatives
        for (int i = 0; i < 10; i++) {
            int idx = (int)(Math.random() * size);
            float error = predictions[idx] - targets[idx];
            float expectedDeriv = Math.abs(error) <= 1.0f ? error : Math.signum(error);
            assertEquals(expectedDeriv, derivatives[idx], EPSILON);
        }
    }
    
    @Test
    public void testGradientContinuity() {
        // Verify that gradients are continuous at the transition point
        HuberLoss loss = HuberLoss.create(1.0f);
        
        // Test points around delta
        float[] predictions = {1.99f, 2.0f, 2.01f};
        float[] targets = {1.0f, 1.0f, 1.0f}; // errors: 0.99, 1.0, 1.01
        
        float[] derivatives = loss.derivatives(predictions, targets);
        
        // At delta, both formulas should give same result
        assertEquals(0.99f, derivatives[0], EPSILON);  // < delta: use error
        assertEquals(1.0f, derivatives[1], EPSILON);   // = delta: both give 1.0
        assertEquals(1.0f, derivatives[2], EPSILON);   // > delta: use sign * delta
    }
    
    @Test
    public void testSymmetry() {
        // Huber loss should be symmetric
        HuberLoss loss = HuberLoss.createDefault();
        
        float[] pred1 = {5.0f};
        float[] targ1 = {3.0f}; // error = 2.0
        
        float[] pred2 = {3.0f};
        float[] targ2 = {5.0f}; // error = -2.0
        
        float loss1 = loss.loss(pred1, targ1);
        float loss2 = loss.loss(pred2, targ2);
        
        assertEquals(loss1, loss2, EPSILON, "Huber loss should be symmetric");
        
        // Derivatives should have opposite signs but same magnitude
        float[] deriv1 = loss.derivatives(pred1, targ1);
        float[] deriv2 = loss.derivatives(pred2, targ2);
        
        assertEquals(-deriv1[0], deriv2[0], EPSILON, "Derivatives should be opposite");
    }
}