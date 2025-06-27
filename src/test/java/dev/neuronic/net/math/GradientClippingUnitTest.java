package dev.neuronic.net.math;

import dev.neuronic.net.math.ops.VectorNorm;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for gradient clipping functionality.
 */
public class GradientClippingUnitTest {
    
    private static final float DELTA = 1e-6f;
    
    @Test
    public void testL2NormComputation() {
        // Test case: [3, 4] should have L2 norm of 5
        float[] gradient = {3.0f, 4.0f};
        float norm = VectorNorm.computeL2(gradient);
        assertEquals(5.0f, norm, DELTA);
        
        // Test with larger vector
        float[] gradient2 = {1.0f, 2.0f, 2.0f};
        float norm2 = VectorNorm.computeL2(gradient2);
        assertEquals(3.0f, norm2, DELTA);
        
        // Test with zero vector
        float[] gradient3 = {0.0f, 0.0f, 0.0f};
        float norm3 = VectorNorm.computeL2(gradient3);
        assertEquals(0.0f, norm3, DELTA);
    }
    
    @Test
    public void testGradientNormClipping() {
        // Test gradient clipping logic
        float[] gradient = {3.0f, 4.0f}; // norm = 5
        float maxNorm = 2.5f;
        
        float originalNorm = VectorNorm.computeL2(gradient);
        assertEquals(5.0f, originalNorm, DELTA);
        
        // Simulate clipping
        if (originalNorm > maxNorm) {
            float scaleFactor = maxNorm / originalNorm;
            assertEquals(0.5f, scaleFactor, DELTA);
            
            // Apply scaling
            for (int i = 0; i < gradient.length; i++) {
                gradient[i] *= scaleFactor;
            }
        }
        
        // Verify clipped values
        assertEquals(1.5f, gradient[0], DELTA);
        assertEquals(2.0f, gradient[1], DELTA);
        
        // Verify new norm
        float clippedNorm = VectorNorm.computeL2(gradient);
        assertEquals(2.5f, clippedNorm, DELTA);
    }
    
    @Test
    public void testGradientClippingWithSmallNorm() {
        // Test that gradients with norm <= maxNorm are not modified
        float[] gradient = {1.0f, 1.0f}; // norm = sqrt(2) ≈ 1.414
        float maxNorm = 2.0f;
        
        float originalNorm = VectorNorm.computeL2(gradient);
        float[] originalValues = gradient.clone();
        
        // Should not clip
        if (originalNorm > maxNorm) {
            float scaleFactor = maxNorm / originalNorm;
            for (int i = 0; i < gradient.length; i++) {
                gradient[i] *= scaleFactor;
            }
        }
        
        // Verify no change
        assertArrayEquals(originalValues, gradient, DELTA);
    }
    
    @Test
    public void testGradientClippingWithLargeNorm() {
        // Test extreme gradient clipping
        float[] gradient = new float[100];
        for (int i = 0; i < gradient.length; i++) {
            gradient[i] = 10.0f; // Very large gradients
        }
        
        float maxNorm = 5.0f;
        float originalNorm = VectorNorm.computeL2(gradient);
        
        // Should be large
        assertTrue(originalNorm > 50.0f);
        
        // Apply clipping
        float scaleFactor = maxNorm / originalNorm;
        for (int i = 0; i < gradient.length; i++) {
            gradient[i] *= scaleFactor;
        }
        
        // Verify clipped norm
        float clippedNorm = VectorNorm.computeL2(gradient);
        assertEquals(maxNorm, clippedNorm, DELTA);
    }
    
    /**
     * Helper method to simulate gradient clipping as done in NeuralNet.
     */
    private void clipGradientsByNorm(float[] gradients, float maxNorm) {
        float norm = VectorNorm.computeL2(gradients);
        if (norm > maxNorm) {
            float scaleFactor = maxNorm / norm;
            for (int i = 0; i < gradients.length; i++) {
                gradients[i] *= scaleFactor;
            }
        }
    }
    
    @Test
    public void testClipGradientsHelper() {
        // Test the helper method
        float[] gradients = {6.0f, 8.0f}; // norm = 10
        clipGradientsByNorm(gradients, 5.0f);
        
        assertEquals(3.0f, gradients[0], DELTA);
        assertEquals(4.0f, gradients[1], DELTA);
        
        float newNorm = VectorNorm.computeL2(gradients);
        assertEquals(5.0f, newNorm, DELTA);
    }
}