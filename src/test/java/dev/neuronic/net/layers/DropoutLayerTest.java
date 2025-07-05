package dev.neuronic.net.layers;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class DropoutLayerTest {
    
    private static final float EPSILON = 1e-6f;
    
    @Test
    void testConstruction() {
        DropoutLayer layer = new DropoutLayer(0.5f, 100);
        assertEquals(100, layer.getOutputSize());
        assertEquals(0.5f, layer.getDropoutRate(), EPSILON);
    }
    
    @Test
    void testInvalidDropoutRate() {
        assertThrows(IllegalArgumentException.class, () -> new DropoutLayer(-0.1f, 100));
        assertThrows(IllegalArgumentException.class, () -> new DropoutLayer(1.0f, 100));
        assertThrows(IllegalArgumentException.class, () -> new DropoutLayer(1.5f, 100));
    }
    
    @Test
    void testInvalidSize() {
        assertThrows(IllegalArgumentException.class, () -> new DropoutLayer(0.5f, 0));
        assertThrows(IllegalArgumentException.class, () -> new DropoutLayer(0.5f, -2)); // -1 is valid for dynamic
    }
    
    @Test
    void testForwardWithDropout() {
        // Dropout now always applies the dropout mask
        DropoutLayer layer = new DropoutLayer(0.5f, 1000); // Large size for statistical test
        
        float[] input = new float[1000];
        for (int i = 0; i < 1000; i++) {
            input[i] = 1.0f;
        }
        
        Layer.LayerContext context = layer.forward(input, true);
        float[] output = context.outputs();
        
        // With dropout, some values should be zeroed and others scaled
        int zeros = 0;
        float expectedScale = 2.0f; // 1 / (1 - 0.5)
        
        for (float val : output) {
            if (val == 0.0f) {
                zeros++;
            } else {
                assertEquals(expectedScale, val, EPSILON); // Check scaling
            }
        }
        
        // With 50% dropout, we expect roughly 50% zeros (with some variance)
        assertTrue(zeros > 400 && zeros < 600, 
            "Expected ~500 zeros but got " + zeros);
    }
    
    @Test
    void testForwardZeroDropout() {
        // With 0% dropout, all values should pass through scaled by 1
        DropoutLayer layer = new DropoutLayer(0.0f, 5);
        
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Layer.LayerContext context = layer.forward(input, true);
        
        assertArrayEquals(input, context.outputs(), EPSILON);
    }
    
    @Test
    void testForwardTrainingMode() {
        // In training mode with dropout, some values should be zeroed and others scaled
        DropoutLayer layer = new DropoutLayer(0.5f, 1000); // Large size for statistical test
        
        float[] input = new float[1000];
        for (int i = 0; i < 1000; i++) {
            input[i] = 1.0f;
        }
        
        Layer.LayerContext context = layer.forward(input, true);
        float[] output = context.outputs();
        
        // Count zeros and non-zeros
        int zeros = 0;
        int nonZeros = 0;
        float expectedScale = 2.0f; // 1 / (1 - 0.5)
        
        for (float val : output) {
            if (val == 0.0f) {
                zeros++;
            } else {
                assertEquals(expectedScale, val, EPSILON); // Check scaling
                nonZeros++;
            }
        }
        
        // With 50% dropout, we expect roughly 50% zeros (with some variance)
        assertTrue(zeros > 400 && zeros < 600, 
            "Expected ~500 zeros but got " + zeros);
        assertTrue(nonZeros > 400 && nonZeros < 600, 
            "Expected ~500 non-zeros but got " + nonZeros);
    }
    
    @Test
    void testBackwardWithDropout() {
        // Gradients should follow the same mask as forward pass
        DropoutLayer layer = new DropoutLayer(0.5f, 5);
        
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        DropoutLayer.DropoutContext context = (DropoutLayer.DropoutContext) layer.forward(input, true);
        
        float[] upstreamGradient = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        Layer.LayerContext[] stack = {context};
        float[] downstreamGradient = layer.backward(stack, 0, upstreamGradient);
        
        // Check that gradient mask matches forward mask
        for (int i = 0; i < 5; i++) {
            if (context.mask[i]) {
                assertEquals(upstreamGradient[i] * 2.0f, downstreamGradient[i], EPSILON);
            } else {
                assertEquals(0.0f, downstreamGradient[i], EPSILON);
            }
        }
    }
    
    @Test
    void testBackwardTrainingMode() {
        // In training mode, gradients should follow the same mask as forward pass
        DropoutLayer layer = new DropoutLayer(0.5f, 5);
        
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        DropoutLayer.DropoutContext context = (DropoutLayer.DropoutContext) layer.forward(input, true);
        
        float[] upstreamGradient = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        Layer.LayerContext[] stack = {context};
        float[] downstreamGradient = layer.backward(stack, 0, upstreamGradient);
        
        // Check that gradient mask matches forward mask
        for (int i = 0; i < 5; i++) {
            if (context.mask[i]) {
                assertEquals(upstreamGradient[i] * 2.0f, downstreamGradient[i], EPSILON);
            } else {
                assertEquals(0.0f, downstreamGradient[i], EPSILON);
            }
        }
    }
    
    @Test
    void testExpectedValuePreservation() {
        // Verify that expected value is preserved with inverted dropout
        DropoutLayer layer = new DropoutLayer(0.3f, 1000); // 30% dropout
        
        // Use fixed values for more predictable test
        float[] input = new float[1000];
        float inputSum = 0.0f;
        java.util.Random rng = new java.util.Random(12345); // Fixed seed
        for (int i = 0; i < input.length; i++) {
            input[i] = rng.nextFloat() * 2.0f - 1.0f; // Random values [-1, 1]
            inputSum += input[i];
        }
        
        // Run multiple forward passes and average
        float outputSumTotal = 0.0f;
        int numTrials = 1000;
        
        for (int trial = 0; trial < numTrials; trial++) {
            Layer.LayerContext context = layer.forward(input, true);
            float[] output = context.outputs();
            
            float outputSum = 0.0f;
            for (float val : output) {
                outputSum += val;
            }
            outputSumTotal += outputSum;
        }
        
        float avgOutputSum = outputSumTotal / numTrials;
        
        // Expected value should be preserved (within statistical variance)
        // For 1000 samples with 1000 trials, we expect good convergence
        float tolerance = Math.max(1.0f, Math.abs(inputSum) * 0.05f); // 5% tolerance or at least 1.0
        assertEquals(inputSum, avgOutputSum, tolerance);
    }
    
    @Test
    void testLayerSpec() {
        Layer.Spec spec = DropoutLayer.spec(0.5f);
        assertEquals(-1, spec.getOutputSize()); // Indicates output size matches input
        
        Layer layer = spec.create(100);
        assertTrue(layer instanceof DropoutLayer);
        assertEquals(100, layer.getOutputSize());
    }
    
    @Test
    void testSerialization() throws Exception {
        DropoutLayer original = new DropoutLayer(0.3f, 50);
        
        // Serialize
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        java.io.DataOutputStream out = new java.io.DataOutputStream(baos);
        original.writeTo(out, 1);
        
        // Deserialize
        byte[] data = baos.toByteArray();
        java.io.DataInputStream in = new java.io.DataInputStream(
            new java.io.ByteArrayInputStream(data));
        DropoutLayer loaded = DropoutLayer.deserialize(in, 1);
        
        // Verify
        assertEquals(original.getDropoutRate(), loaded.getDropoutRate(), EPSILON);
        assertEquals(original.getOutputSize(), loaded.getOutputSize());
    }
    
    @Test
    void testInputSizeMismatch() {
        DropoutLayer layer = new DropoutLayer(0.5f, 5);
        float[] wrongSizeInput = {1.0f, 2.0f, 3.0f}; // Wrong size
        
        assertThrows(IllegalArgumentException.class, () -> layer.forward(wrongSizeInput, true));
    }
    
    @Test
    void testDropoutAlwaysActive() {
        // Test that dropout is always active (no training mode toggle)
        DropoutLayer layer = new DropoutLayer(0.5f, 100);
        
        float[] input = new float[100];
        for (int i = 0; i < 100; i++) {
            input[i] = 1.0f;
        }
        
        // Run multiple times to verify dropout is consistently applied
        boolean foundDifference = false;
        float[] firstOutput = layer.forward(input, true).outputs().clone();
        
        for (int trial = 0; trial < 10; trial++) {
            float[] output = layer.forward(input, true).outputs();
            if (!java.util.Arrays.equals(firstOutput, output)) {
                foundDifference = true;
                break;
            }
        }
        
        assertTrue(foundDifference, "Dropout should produce different outputs on different forward passes");
    }
    
    @Test
    void testDifferentDropoutRates() {
        // Test various dropout rates
        float[] rates = {0.1f, 0.2f, 0.3f, 0.5f, 0.7f, 0.9f};
        
        for (float rate : rates) {
            DropoutLayer layer = new DropoutLayer(rate, 1000);
            float[] input = new float[1000];
            for (int i = 0; i < 1000; i++) {
                input[i] = 1.0f;
            }
            
            Layer.LayerContext context = layer.forward(input, true);
            float[] output = context.outputs();
            
            int zeros = 0;
            for (float val : output) {
                if (val == 0.0f) zeros++;
            }
            
            float actualDropoutRate = zeros / 1000.0f;
            // Allow 20% relative error due to randomness (or minimum 0.05 absolute)
            float tolerance = Math.max(rate * 0.2f, 0.05f);
            assertEquals(rate, actualDropoutRate, tolerance);
        }
    }
}