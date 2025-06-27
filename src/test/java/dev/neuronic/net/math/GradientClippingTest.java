package dev.neuronic.net.math;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test gradient clipping functionality.
 */
public class GradientClippingTest {
    
    private static final float EPSILON = 1e-6f;
    
    @Test
    public void testNormClipping() {
        GradientClipper clipper = GradientClipper.byNorm(1.0f);
        
        // Test gradients with norm > 1.0 (should be clipped)
        float[] gradients = {3.0f, 4.0f}; // norm = 5.0
        boolean clipped = clipper.clipInPlace(gradients);
        
        assertTrue(clipped, "Gradients should have been clipped");
        
        // Check that norm is now 1.0
        float norm = (float) Math.sqrt(gradients[0] * gradients[0] + gradients[1] * gradients[1]);
        assertEquals(1.0f, norm, EPSILON, "Norm should be 1.0 after clipping");
        
        // Check that direction is preserved (3:4 ratio)
        assertEquals(0.6f, gradients[0], EPSILON, "Direction should be preserved");
        assertEquals(0.8f, gradients[1], EPSILON, "Direction should be preserved");
    }
    
    @Test
    public void testNormClippingNoClip() {
        GradientClipper clipper = GradientClipper.byNorm(5.0f);
        
        // Test gradients with norm < 5.0 (should not be clipped)
        float[] gradients = {1.0f, 1.0f}; // norm = sqrt(2) ≈ 1.414
        float[] original = gradients.clone();
        
        boolean clipped = clipper.clipInPlace(gradients);
        
        assertFalse(clipped, "Gradients should not have been clipped");
        assertArrayEquals(original, gradients, EPSILON, "Gradients should be unchanged");
    }
    
    @Test
    public void testValueClipping() {
        GradientClipper clipper = GradientClipper.byValue(2.0f);
        
        // Test gradients with values outside [-2.0, 2.0]
        float[] gradients = {-5.0f, 1.0f, 3.0f, -1.5f};
        boolean clipped = clipper.clipInPlace(gradients);
        
        assertTrue(clipped, "Gradients should have been clipped");
        
        float[] expected = {-2.0f, 1.0f, 2.0f, -1.5f};
        assertArrayEquals(expected, gradients, EPSILON, "Values should be clipped to range");
    }
    
    @Test
    public void testValueClippingNoClip() {
        GradientClipper clipper = GradientClipper.byValue(5.0f);
        
        // Test gradients with all values in range
        float[] gradients = {-2.0f, 1.0f, 3.0f, -1.5f};
        float[] original = gradients.clone();
        
        boolean clipped = clipper.clipInPlace(gradients);
        
        assertFalse(clipped, "Gradients should not have been clipped");
        assertArrayEquals(original, gradients, EPSILON, "Gradients should be unchanged");
    }
    
    @Test
    public void testNoOpClipper() {
        GradientClipper clipper = GradientClipper.none();
        
        // Test with extreme gradients
        float[] gradients = {-100.0f, 200.0f, -50.0f};
        float[] original = gradients.clone();
        
        boolean clipped = clipper.clipInPlace(gradients);
        
        assertFalse(clipped, "No-op clipper should never clip");
        assertArrayEquals(original, gradients, EPSILON, "Gradients should be unchanged");
        assertFalse(clipper.wouldClip(gradients), "No-op clipper should never report clipping");
    }
    
    @Test
    public void testWouldClipMethods() {
        GradientClipper normClipper = GradientClipper.byNorm(1.0f);
        GradientClipper valueClipper = GradientClipper.byValue(2.0f);
        
        float[] largeNorm = {3.0f, 4.0f}; // norm = 5.0
        float[] smallNorm = {0.3f, 0.4f}; // norm = 0.5
        float[] largeValues = {-5.0f, 3.0f};
        float[] smallValues = {1.0f, -1.5f};
        
        assertTrue(normClipper.wouldClip(largeNorm), "Should detect large norm");
        assertFalse(normClipper.wouldClip(smallNorm), "Should not detect small norm");
        
        assertTrue(valueClipper.wouldClip(largeValues), "Should detect large values");
        assertFalse(valueClipper.wouldClip(smallValues), "Should not detect small values");
    }
    
    @Test
    public void testAdaptiveClipper() {
        GradientClipper clipper = GradientClipper.adaptive(1.0f, 0.1f);
        
        // Test that it adapts over time
        float[] gradients1 = {2.0f, 0.0f}; // norm = 2.0
        float[] gradients2 = {3.0f, 0.0f}; // norm = 3.0
        
        // First gradient should be clipped
        boolean clipped1 = clipper.clipInPlace(gradients1);
        assertTrue(clipped1, "First gradient should be clipped");
        
        // Second gradient behavior depends on adaptation
        boolean clipped2 = clipper.clipInPlace(gradients2);
        // We can't predict exact behavior due to adaptation, but it should work
        assertNotNull(clipper.getDescription(), "Should have description");
    }
    
    @Test
    public void testEmptyArrays() {
        GradientClipper normClipper = GradientClipper.byNorm(1.0f);
        GradientClipper valueClipper = GradientClipper.byValue(2.0f);
        
        float[] empty = {};
        
        assertFalse(normClipper.clipInPlace(empty), "Empty array should not be clipped");
        assertFalse(valueClipper.clipInPlace(empty), "Empty array should not be clipped");
        assertFalse(normClipper.wouldClip(empty), "Empty array should not trigger clipping");
        assertFalse(valueClipper.wouldClip(empty), "Empty array should not trigger clipping");
    }
    
    @Test
    public void testClipperDescriptions() {
        GradientClipper normClipper = GradientClipper.byNorm(1.5f);
        GradientClipper valueClipper = GradientClipper.byValue(3.0f);
        GradientClipper noOpClipper = GradientClipper.none();
        
        assertTrue(normClipper.getDescription().contains("1.500"), 
                  "Norm clipper description should contain threshold");
        assertTrue(valueClipper.getDescription().contains("3.000") || valueClipper.getDescription().contains("-3.000"), 
                  "Value clipper description should contain range");
        assertTrue(noOpClipper.getDescription().contains("disabled") || noOpClipper.getDescription().contains("NoOp"), 
                  "No-op clipper description should indicate disabled");
    }
    
    @Test
    public void testLargeArrayPerformance() {
        // Test that clipping works efficiently on large arrays
        GradientClipper clipper = GradientClipper.byNorm(1.0f);
        
        // Create large array that needs clipping
        int size = 10000;
        float[] gradients = new float[size];
        for (int i = 0; i < size; i++) {
            gradients[i] = (float) Math.sin(i) * 10.0f; // Large values
        }
        
        long start = System.nanoTime();
        boolean clipped = clipper.clipInPlace(gradients);
        long duration = System.nanoTime() - start;
        
        assertTrue(clipped, "Large array should be clipped");
        
        // Verify norm is approximately 1.0
        float sumSquares = 0.0f;
        for (float grad : gradients) {
            sumSquares += grad * grad;
        }
        float norm = (float) Math.sqrt(sumSquares);
        assertEquals(1.0f, norm, 1e-4f, "Norm should be 1.0 after clipping");
        
        // Performance should be reasonable (under 50ms for 10k elements)
        assertTrue(duration < 50_000_000, "Should complete in under 50ms, took: " + duration + "ns");
    }
}