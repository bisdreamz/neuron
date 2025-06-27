package dev.neuronic.net.activators;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TanhActivatorTest {

    @Test
    void testSingletonPattern() {
        TanhActivator activator1 = TanhActivator.INSTANCE;
        TanhActivator activator2 = TanhActivator.INSTANCE;
        assertSame(activator1, activator2, "Should return the same instance");
    }

    @Test
    void testActivateBasicValues() {
        TanhActivator activator = TanhActivator.INSTANCE;
        float[] input = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
        float[] output = new float[input.length];
        
        activator.activate(input, output);
        
        // tanh(0) = 0
        assertEquals(0.0f, output[0], 0.001f);
        
        // tanh(1) ≈ 0.7616
        assertEquals(Math.tanh(1.0), output[1], 0.001f);
        
        // tanh(-1) ≈ -0.7616
        assertEquals(Math.tanh(-1.0), output[2], 0.001f);
        
        // tanh(2) ≈ 0.9640
        assertEquals(Math.tanh(2.0), output[3], 0.001f);
        
        // tanh(-2) ≈ -0.9640
        assertEquals(Math.tanh(-2.0), output[4], 0.001f);
    }

    @Test
    void testActivateOutputRange() {
        TanhActivator activator = TanhActivator.INSTANCE;
        float[] input = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
        float[] output = new float[input.length];
        
        activator.activate(input, output);
        
        // All outputs should be in range (-1, 1), but at extremes can approach the bounds
        for (int i = 0; i < output.length; i++) {
            assertTrue(output[i] >= -1.0f, "Output should be >= -1: " + output[i]);
            assertTrue(output[i] <= 1.0f, "Output should be <= 1: " + output[i]);
        }
        
        // Large negative input should be close to -1
        assertTrue(output[0] < -0.99f, "tanh(-10) should be close to -1");
        
        // Large positive input should be close to 1
        assertTrue(output[4] > 0.99f, "tanh(10) should be close to 1");
    }

    @Test
    void testActivateZeroCentered() {
        TanhActivator activator = TanhActivator.INSTANCE;
        float[] input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        float[] output = new float[input.length];
        
        activator.activate(input, output);
        
        // tanh is odd function: tanh(-x) = -tanh(x)
        assertEquals(-output[4], output[0], 0.001f, "tanh(-2) should equal -tanh(2)");
        assertEquals(-output[3], output[1], 0.001f, "tanh(-1) should equal -tanh(1)");
        assertEquals(0.0f, output[2], 0.001f, "tanh(0) should be 0");
    }

    @Test
    void testDerivativeBasicValues() {
        TanhActivator activator = TanhActivator.INSTANCE;
        float[] tanhOutput = {0.0f, 0.7616f, -0.7616f, 0.9640f, -0.9640f};
        float[] derivative = new float[tanhOutput.length];
        
        activator.derivative(tanhOutput, derivative);
        
        // Derivative of tanh(x) = 1 - tanh²(x)
        for (int i = 0; i < tanhOutput.length; i++) {
            float expected = 1.0f - tanhOutput[i] * tanhOutput[i];
            assertEquals(expected, derivative[i], 0.001f, 
                "Derivative should be 1 - tanh²(x) for index " + i);
        }
    }

    @Test
    void testDerivativeRange() {
        TanhActivator activator = TanhActivator.INSTANCE;
        float[] tanhOutput = {-0.99f, -0.5f, 0.0f, 0.5f, 0.99f};
        float[] derivative = new float[tanhOutput.length];
        
        activator.derivative(tanhOutput, derivative);
        
        // All derivatives should be in range (0, 1]
        for (int i = 0; i < derivative.length; i++) {
            assertTrue(derivative[i] > 0.0f, "Derivative should be positive: " + derivative[i]);
            assertTrue(derivative[i] <= 1.0f, "Derivative should be <= 1: " + derivative[i]);
        }
        
        // Maximum derivative should be at tanh(x) = 0
        assertTrue(derivative[2] > 0.99f, "Maximum derivative should be at tanh=0");
        
        // Derivatives should be smaller for larger |tanh| values
        assertTrue(derivative[0] < derivative[2], "Derivative should be smaller for larger |tanh|");
        assertTrue(derivative[4] < derivative[2], "Derivative should be smaller for larger |tanh|");
    }

    @Test
    void testDerivativeSymmetry() {
        TanhActivator activator = TanhActivator.INSTANCE;
        float[] tanhOutput = {-0.7616f, 0.7616f, -0.9640f, 0.9640f};
        float[] derivative = new float[tanhOutput.length];
        
        activator.derivative(tanhOutput, derivative);
        
        // Derivative should be symmetric: f'(tanh(x)) = f'(tanh(-x))
        assertEquals(derivative[0], derivative[1], 0.001f, 
            "Derivative should be same for tanh(x) and tanh(-x)");
        assertEquals(derivative[2], derivative[3], 0.001f, 
            "Derivative should be same for tanh(x) and tanh(-x)");
    }

    @Test
    void testVectorizedVsScalar() {
        TanhActivator activator = TanhActivator.INSTANCE;
        float[] input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, -3.0f, 0.5f};
        
        float[] outputScalar = new float[input.length];
        float[] outputVectorized = new float[input.length];
        
        // Force scalar computation
        activator.activateScalar(input, outputScalar);
        
        // Force vectorized computation (if supported)
        activator.activateVectorized(input, outputVectorized);
        
        // Results should be nearly identical
        for (int i = 0; i < input.length; i++) {
            assertEquals(outputScalar[i], outputVectorized[i], 0.0001f, 
                "Scalar and vectorized results should match for index " + i);
        }
    }

    @Test
    void testDerivativeVectorizedVsScalar() {
        TanhActivator activator = TanhActivator.INSTANCE;
        float[] tanhOutput = {-0.9640f, -0.7616f, 0.0f, 0.7616f, 0.9640f, 0.9951f, -0.9951f, 0.4621f};
        
        float[] derivativeScalar = new float[tanhOutput.length];
        float[] derivativeVectorized = new float[tanhOutput.length];
        
        // Force scalar computation
        activator.derivativeScalar(tanhOutput, derivativeScalar);
        
        // Force vectorized computation (if supported)
        activator.derivativeVectorized(tanhOutput, derivativeVectorized);
        
        // Results should be nearly identical
        for (int i = 0; i < tanhOutput.length; i++) {
            assertEquals(derivativeScalar[i], derivativeVectorized[i], 0.0001f, 
                "Scalar and vectorized derivative results should match for index " + i);
        }
    }

    @Test
    void testInputOutputLengthMismatch() {
        TanhActivator activator = TanhActivator.INSTANCE;
        float[] input = {1.0f, 2.0f, 3.0f};
        float[] output = {0.0f, 0.0f}; // Wrong length
        
        assertThrows(IllegalArgumentException.class, () -> {
            activator.activate(input, output);
        }, "Should throw exception for mismatched array lengths");
        
        assertThrows(IllegalArgumentException.class, () -> {
            activator.derivative(input, output);
        }, "Should throw exception for mismatched array lengths");
    }

    @Test
    void testLargeArrayPerformance() {
        TanhActivator activator = TanhActivator.INSTANCE;
        int size = 1000;
        float[] input = new float[size];
        float[] output = new float[size];
        
        // Fill with random-ish values
        for (int i = 0; i < size; i++) {
            input[i] = (float) (Math.sin(i) * 5.0); // Values in range [-5, 5]
        }
        
        // Should not throw and should complete reasonably quickly
        long start = System.nanoTime();
        activator.activate(input, output);
        long duration = System.nanoTime() - start;
        
        // Verify results are reasonable
        for (int i = 0; i < size; i++) {
            assertTrue(output[i] >= -1.0f && output[i] <= 1.0f, 
                "Output should be in tanh range");
            assertEquals(Math.tanh(input[i]), output[i], 0.001f, 
                "Should match Math.tanh for index " + i);
        }
        
        // Performance check - should complete in reasonable time (< 10ms for 1000 elements)
        assertTrue(duration < 10_000_000, "Should complete in reasonable time: " + duration + "ns");
    }
}