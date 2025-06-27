package dev.neuronic.net.activators;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SoftmaxActivatorTest {
    
    private static final float DELTA = 1e-5f;
    private final SoftmaxActivator softmax = SoftmaxActivator.INSTANCE;
    
    @Test
    void testActivateBasic() {
        float[] input = {1.0f, 2.0f, 3.0f};
        float[] output = new float[3];
        
        softmax.activate(input, output);
        
        // Check probabilities sum to 1
        float sum = output[0] + output[1] + output[2];
        assertEquals(1.0f, sum, DELTA);
        
        // Check all values are positive
        for (float value : output) {
            assertTrue(value > 0.0f);
        }
        
        // Highest input should have highest probability
        assertTrue(output[2] > output[1] && output[1] > output[0]);
    }
    
    @Test
    void testActivateScalar() {
        float[] input = {0.0f, 1.0f, 0.0f};
        float[] output = new float[3];
        
        softmax.activateScalar(input, output);
        
        // Should sum to 1
        float sum = output[0] + output[1] + output[2];
        assertEquals(1.0f, sum, DELTA);
        
        // Middle element should have highest probability
        assertTrue(output[1] > output[0] && output[1] > output[2]);
    }
    
    @Test
    void testActivateVectorized() {
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] output = new float[8];
        
        softmax.activateVectorized(input, output);
        
        // Should sum to 1
        float sum = 0.0f;
        for (float value : output) {
            sum += value;
        }
        assertEquals(1.0f, sum, DELTA);
        
        // Should be monotonically increasing probabilities
        for (int i = 1; i < output.length; i++) {
            assertTrue(output[i] > output[i-1]);
        }
    }
    
    @Test
    void testScalarVsVectorizedConsistency() {
        float[] input = {2.1f, 1.5f, 3.2f, 0.8f, 2.9f, 1.1f, 3.8f, 2.4f};
        float[] output1 = new float[8];
        float[] output2 = new float[8];
        
        softmax.activateScalar(input, output1);
        softmax.activateVectorized(input, output2);
        
        assertArrayEquals(output1, output2, DELTA);
    }
    
    @Test
    void testNumericalStability() {
        // Large values that could cause overflow without max trick
        float[] input = {1000.0f, 1001.0f, 1002.0f};
        float[] output = new float[3];
        
        softmax.activate(input, output);
        
        // Should not produce NaN or infinity
        for (float value : output) {
            assertTrue(Float.isFinite(value));
            assertTrue(value >= 0.0f && value <= 1.0f);
        }
        
        // Should sum to 1
        float sum = output[0] + output[1] + output[2];
        assertEquals(1.0f, sum, DELTA);
    }
    
    @Test
    void testNegativeValues() {
        float[] input = {-1.0f, -2.0f, -3.0f};
        float[] output = new float[3];
        
        softmax.activate(input, output);
        
        // Should still work with negative inputs
        float sum = output[0] + output[1] + output[2];
        assertEquals(1.0f, sum, DELTA);
        
        // Less negative should have higher probability
        assertTrue(output[0] > output[1] && output[1] > output[2]);
    }
    
    @Test
    void testUniformInputs() {
        float[] input = {2.0f, 2.0f, 2.0f, 2.0f};
        float[] output = new float[4];
        
        softmax.activate(input, output);
        
        // All should be equal (uniform distribution)
        for (float value : output) {
            assertEquals(0.25f, value, DELTA);
        }
    }
    
    @Test
    void testSingleElement() {
        float[] input = {5.0f};
        float[] output = new float[1];
        
        softmax.activate(input, output);
        
        assertEquals(1.0f, output[0], DELTA);
    }
    
    @Test
    void testTwoElements() {
        float[] input = {0.0f, 1.0f};
        float[] output = new float[2];
        
        softmax.activate(input, output);
        
        assertEquals(1.0f, output[0] + output[1], DELTA);
        assertTrue(output[1] > output[0]); // exp(1) > exp(0)
    }
    
    @Test
    void testMNISTLikeOutput() {
        // Simulate typical MNIST output layer
        float[] input = new float[10];
        for (int i = 0; i < 10; i++) {
            input[i] = (float) (Math.random() * 10 - 5); // Random values -5 to 5
        }
        float[] output = new float[10];
        
        softmax.activate(input, output);
        
        // Should sum to 1
        float sum = 0.0f;
        for (float value : output) {
            sum += value;
        }
        assertEquals(1.0f, sum, DELTA);
        
        // All probabilities should be valid
        for (float value : output) {
            assertTrue(value >= 0.0f && value <= 1.0f);
        }
    }
    
    @Test
    void testDerivativeBasic() {
        // Test derivative with already-softmaxed input
        float[] input = {0.2f, 0.3f, 0.5f}; // Already probabilities
        float[] output = new float[3];
        
        softmax.derivative(input, output);
        
        // Check derivative formula: softmax_i * (1 - softmax_i)
        assertEquals(0.2f * 0.8f, output[0], DELTA);
        assertEquals(0.3f * 0.7f, output[1], DELTA);
        assertEquals(0.5f * 0.5f, output[2], DELTA);
    }
    
    @Test
    void testDerivativeScalarVsVectorized() {
        float[] input = {0.1f, 0.15f, 0.25f, 0.3f, 0.05f, 0.1f, 0.02f, 0.03f};
        float[] output1 = new float[8];
        float[] output2 = new float[8];
        
        softmax.derivativeScalar(input, output1);
        softmax.derivativeVectorized(input, output2);
        
        assertArrayEquals(output1, output2, DELTA);
    }
    
    @Test
    void testActivateDimensionMismatch() {
        float[] input = {1.0f, 2.0f};
        float[] output = new float[3];
        
        assertThrows(IllegalArgumentException.class, () -> 
            softmax.activate(input, output));
    }
    
    @Test
    void testDerivativeDimensionMismatch() {
        float[] input = {1.0f, 2.0f, 3.0f};
        float[] output = new float[2];
        
        assertThrows(IllegalArgumentException.class, () -> 
            softmax.derivative(input, output));
    }
    
    @Test
    void testMaxFinding() {
        float[] input = {1.5f, 3.2f, 2.1f, 4.8f, 1.9f};
        float[] output = new float[5];
        
        softmax.activate(input, output);
        
        // Element with max input (4.8f at index 3) should have highest probability
        int maxIndex = 0;
        for (int i = 1; i < output.length; i++) {
            if (output[i] > output[maxIndex])
                maxIndex = i;
        }
        assertEquals(3, maxIndex);
    }
    
    @Test
    void testExtremeValues() {
        float[] input = {-100.0f, 0.0f, 100.0f};
        float[] output = new float[3];
        
        softmax.activate(input, output);
        
        // Should handle extreme values gracefully
        assertTrue(Float.isFinite(output[0]));
        assertTrue(Float.isFinite(output[1]));
        assertTrue(Float.isFinite(output[2]));
        
        // Largest input should dominate
        assertTrue(output[2] > 0.99f);
        assertTrue(output[0] < 0.01f);
    }
}