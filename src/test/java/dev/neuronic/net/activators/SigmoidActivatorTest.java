package dev.neuronic.net.activators;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SigmoidActivatorTest {

    @Test
    void testSingletonPattern() {
        SigmoidActivator activator1 = SigmoidActivator.INSTANCE;
        SigmoidActivator activator2 = SigmoidActivator.INSTANCE;
        assertSame(activator1, activator2, "Should return the same instance");
    }

    @Test
    void testActivateBasicValues() {
        SigmoidActivator activator = SigmoidActivator.INSTANCE;
        float[] input = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
        float[] output = new float[input.length];
        
        activator.activate(input, output);
        
        // sigmoid(0) = 0.5
        assertEquals(0.5f, output[0], 0.001f);
        
        // sigmoid(1) ≈ 0.7311
        assertEquals(1.0f / (1.0f + Math.exp(-1.0f)), output[1], 0.001f);
        
        // sigmoid(-1) ≈ 0.2689
        assertEquals(1.0f / (1.0f + Math.exp(1.0f)), output[2], 0.001f);
        
        // sigmoid(2) ≈ 0.8808
        assertEquals(1.0f / (1.0f + Math.exp(-2.0f)), output[3], 0.001f);
        
        // sigmoid(-2) ≈ 0.1192
        assertEquals(1.0f / (1.0f + Math.exp(2.0f)), output[4], 0.001f);
    }

    @Test
    void testActivateOutputRange() {
        SigmoidActivator activator = SigmoidActivator.INSTANCE;
        float[] input = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
        float[] output = new float[input.length];
        
        activator.activate(input, output);
        
        // All outputs should be in range (0, 1)
        for (int i = 0; i < output.length; i++) {
            assertTrue(output[i] > 0.0f, "Output should be greater than 0: " + output[i]);
            assertTrue(output[i] < 1.0f, "Output should be less than 1: " + output[i]);
        }
        
        // Large negative input should be close to 0
        assertTrue(output[0] < 0.01f, "sigmoid(-10) should be close to 0");
        
        // Large positive input should be close to 1
        assertTrue(output[4] > 0.99f, "sigmoid(10) should be close to 1");
        
        // Zero input should be exactly 0.5
        assertEquals(0.5f, output[2], 0.001f, "sigmoid(0) should be 0.5");
    }

    @Test
    void testActivateNumericalStability() {
        SigmoidActivator activator = SigmoidActivator.INSTANCE;
        float[] input = {-100.0f, -50.0f, 50.0f, 100.0f};
        float[] output = new float[input.length];
        
        activator.activate(input, output);
        
        // Should not produce NaN or infinite values
        for (int i = 0; i < output.length; i++) {
            assertFalse(Float.isNaN(output[i]), "Output should not be NaN: " + output[i]);
            assertFalse(Float.isInfinite(output[i]), "Output should not be infinite: " + output[i]);
        }
        
        // Very large negative should be very close to 0
        assertTrue(output[0] < 1e-6f, "sigmoid(-100) should be very close to 0");
        assertTrue(output[1] < 1e-6f, "sigmoid(-50) should be very close to 0");
        
        // Very large positive should be very close to 1
        assertTrue(output[2] > 1.0f - 1e-6f, "sigmoid(50) should be very close to 1");
        assertTrue(output[3] > 1.0f - 1e-6f, "sigmoid(100) should be very close to 1");
    }

    @Test
    void testActivateSymmetry() {
        SigmoidActivator activator = SigmoidActivator.INSTANCE;
        float[] input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        float[] output = new float[input.length];
        
        activator.activate(input, output);
        
        // sigmoid(-x) = 1 - sigmoid(x)
        assertEquals(1.0f - output[4], output[0], 0.001f, "sigmoid(-2) should equal 1 - sigmoid(2)");
        assertEquals(1.0f - output[3], output[1], 0.001f, "sigmoid(-1) should equal 1 - sigmoid(1)");
        assertEquals(0.5f, output[2], 0.001f, "sigmoid(0) should be 0.5");
    }

    @Test
    void testDerivativeBasicValues() {
        SigmoidActivator activator = SigmoidActivator.INSTANCE;
        float[] sigmoidOutput = {0.5f, 0.7311f, 0.2689f, 0.8808f, 0.1192f};
        float[] derivative = new float[sigmoidOutput.length];
        
        activator.derivative(sigmoidOutput, derivative);
        
        // Derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        for (int i = 0; i < sigmoidOutput.length; i++) {
            float expected = sigmoidOutput[i] * (1.0f - sigmoidOutput[i]);
            assertEquals(expected, derivative[i], 0.001f, 
                "Derivative should be sigmoid * (1 - sigmoid) for index " + i);
        }
    }

    @Test
    void testDerivativeRange() {
        SigmoidActivator activator = SigmoidActivator.INSTANCE;
        float[] sigmoidOutput = {0.01f, 0.25f, 0.5f, 0.75f, 0.99f};
        float[] derivative = new float[sigmoidOutput.length];
        
        activator.derivative(sigmoidOutput, derivative);
        
        // All derivatives should be in range (0, 0.25]
        for (int i = 0; i < derivative.length; i++) {
            assertTrue(derivative[i] > 0.0f, "Derivative should be positive: " + derivative[i]);
            assertTrue(derivative[i] <= 0.25f, "Derivative should be <= 0.25: " + derivative[i]);
        }
        
        // Maximum derivative should be at sigmoid(x) = 0.5
        assertTrue(derivative[2] > 0.24f, "Maximum derivative should be at sigmoid=0.5");
        
        // Derivatives should be smaller for values closer to 0 or 1
        assertTrue(derivative[0] < derivative[2], "Derivative should be smaller near 0");
        assertTrue(derivative[4] < derivative[2], "Derivative should be smaller near 1");
    }

    @Test
    void testDerivativeSymmetry() {
        SigmoidActivator activator = SigmoidActivator.INSTANCE;
        // Use symmetric sigmoid values: sigmoid(-x) and sigmoid(x)
        float[] sigmoidOutput = {0.2689f, 0.7311f, 0.1192f, 0.8808f};
        float[] derivative = new float[sigmoidOutput.length];
        
        activator.derivative(sigmoidOutput, derivative);
        
        // Derivative should be same for sigmoid(x) and sigmoid(-x)
        assertEquals(derivative[0], derivative[1], 0.001f, 
            "Derivative should be same for sigmoid(x) and sigmoid(-x)");
        assertEquals(derivative[2], derivative[3], 0.001f, 
            "Derivative should be same for sigmoid(x) and sigmoid(-x)");
    }

    @Test
    void testDerivativeExtremeValues() {
        SigmoidActivator activator = SigmoidActivator.INSTANCE;
        float[] sigmoidOutput = {0.0001f, 0.9999f, 0.5f};
        float[] derivative = new float[sigmoidOutput.length];
        
        activator.derivative(sigmoidOutput, derivative);
        
        // Derivatives for extreme values should be very small
        assertTrue(derivative[0] < 0.001f, "Derivative should be very small for sigmoid ≈ 0");
        assertTrue(derivative[1] < 0.001f, "Derivative should be very small for sigmoid ≈ 1");
        
        // Derivative at 0.5 should be maximum (0.25)
        assertEquals(0.25f, derivative[2], 0.001f, "Derivative at 0.5 should be 0.25");
    }

    @Test
    void testVectorizedVsScalar() {
        SigmoidActivator activator = SigmoidActivator.INSTANCE;
        float[] input = {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.5f};
        
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
        SigmoidActivator activator = SigmoidActivator.INSTANCE;
        float[] sigmoidOutput = {0.0474f, 0.1192f, 0.2689f, 0.5f, 0.7311f, 0.8808f, 0.9526f, 0.6225f};
        
        float[] derivativeScalar = new float[sigmoidOutput.length];
        float[] derivativeVectorized = new float[sigmoidOutput.length];
        
        // Force scalar computation
        activator.derivativeScalar(sigmoidOutput, derivativeScalar);
        
        // Force vectorized computation (if supported)
        activator.derivativeVectorized(sigmoidOutput, derivativeVectorized);
        
        // Results should be nearly identical
        for (int i = 0; i < sigmoidOutput.length; i++) {
            assertEquals(derivativeScalar[i], derivativeVectorized[i], 0.0001f, 
                "Scalar and vectorized derivative results should match for index " + i);
        }
    }

    @Test
    void testInputOutputLengthMismatch() {
        SigmoidActivator activator = SigmoidActivator.INSTANCE;
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
        SigmoidActivator activator = SigmoidActivator.INSTANCE;
        int size = 1000;
        float[] input = new float[size];
        float[] output = new float[size];
        
        // Fill with random-ish values
        for (int i = 0; i < size; i++) {
            input[i] = (float) (Math.sin(i) * 10.0); // Values in range [-10, 10]
        }
        
        // Should not throw and should complete reasonably quickly
        long start = System.nanoTime();
        activator.activate(input, output);
        long duration = System.nanoTime() - start;
        
        // Verify results are reasonable
        for (int i = 0; i < size; i++) {
            assertTrue(output[i] >= 0.0f && output[i] <= 1.0f, 
                "Output should be in sigmoid range");
            float expected = 1.0f / (1.0f + (float) Math.exp(-input[i]));
            assertEquals(expected, output[i], 0.001f, 
                "Should match manual sigmoid calculation for index " + i);
        }
        
        // Performance check - should complete in reasonable time (< 10ms for 1000 elements)
        assertTrue(duration < 10_000_000, "Should complete in reasonable time: " + duration + "ns");
    }
}