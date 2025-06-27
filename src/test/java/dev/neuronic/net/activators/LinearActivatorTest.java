package dev.neuronic.net.activators;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LinearActivatorTest {

    @Test
    void testSingletonPattern() {
        LinearActivator activator1 = LinearActivator.INSTANCE;
        LinearActivator activator2 = LinearActivator.INSTANCE;
        assertSame(activator1, activator2, "Should return the same instance");
    }

    @Test
    void testActivateIdentityFunction() {
        LinearActivator activator = LinearActivator.INSTANCE;
        float[] input = {-10.0f, -1.5f, 0.0f, 1.5f, 10.0f, 3.14159f, -2.71828f};
        float[] output = new float[input.length];
        
        activator.activate(input, output);
        
        // Linear activation should be identity: f(x) = x
        for (int i = 0; i < input.length; i++) {
            assertEquals(input[i], output[i], 0.0f, 
                "Linear activation should be identity for index " + i);
        }
    }

    @Test
    void testActivatePreservesValues() {
        LinearActivator activator = LinearActivator.INSTANCE;
        float[] input = {Float.MAX_VALUE, Float.MIN_VALUE, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY};
        float[] output = new float[input.length];
        
        activator.activate(input, output);
        
        // Should preserve all values exactly, including special values
        for (int i = 0; i < input.length; i++) {
            assertEquals(input[i], output[i], 0.0f, 
                "Should preserve value exactly for index " + i);
        }
    }

    @Test
    void testActivateWithNaN() {
        LinearActivator activator = LinearActivator.INSTANCE;
        float[] input = {1.0f, Float.NaN, 3.0f};
        float[] output = new float[input.length];
        
        activator.activate(input, output);
        
        // Should preserve NaN
        assertEquals(1.0f, output[0]);
        assertTrue(Float.isNaN(output[1]), "Should preserve NaN");
        assertEquals(3.0f, output[2]);
    }

    @Test
    void testDerivativeConstantOne() {
        LinearActivator activator = LinearActivator.INSTANCE;
        float[] input = {-100.0f, -1.0f, 0.0f, 1.0f, 100.0f, 3.14159f};
        float[] derivative = new float[input.length];
        
        activator.derivative(input, derivative);
        
        // Derivative of linear function f(x) = x is always 1
        for (int i = 0; i < derivative.length; i++) {
            assertEquals(1.0f, derivative[i], 0.0f, 
                "Linear derivative should always be 1 for index " + i);
        }
    }

    @Test
    void testDerivativeIgnoresInputValues() {
        LinearActivator activator = LinearActivator.INSTANCE;
        float[] input = {Float.MAX_VALUE, Float.MIN_VALUE, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY};
        float[] derivative = new float[input.length];
        
        activator.derivative(input, derivative);
        
        // Derivative should always be 1, regardless of input values
        for (int i = 0; i < derivative.length; i++) {
            assertEquals(1.0f, derivative[i], 0.0f, 
                "Linear derivative should always be 1, regardless of input for index " + i);
        }
    }

    @Test
    void testSimplicity() {
        LinearActivator activator = LinearActivator.INSTANCE;
        float[] input = {-5.5f, -1.0f, 0.0f, 1.0f, 5.5f, 2.718f, -3.14159f, 42.0f};
        
        float[] output = new float[input.length];
        float[] derivative = new float[input.length];
        
        activator.activate(input, output);
        activator.derivative(input, derivative);
        
        // Output should be identical to input
        for (int i = 0; i < input.length; i++) {
            assertEquals(input[i], output[i], 0.0f, 
                "Linear activation should be identity for index " + i);
        }
        
        // Derivative should always be 1
        for (int i = 0; i < derivative.length; i++) {
            assertEquals(1.0f, derivative[i], 0.0f, 
                "Linear derivative should always be 1 for index " + i);
        }
    }

    @Test
    void testMismatchedLengthsBehavior() {
        LinearActivator activator = LinearActivator.INSTANCE;
        float[] input = {1.0f, 2.0f};
        float[] output = {0.0f, 0.0f, 0.0f}; // Longer array
        
        // LinearActivator uses System.arraycopy which will copy input.length elements
        assertDoesNotThrow(() -> {
            activator.activate(input, output);
        }, "Should handle mismatched lengths when output is longer");
        
        // Should copy all input elements
        assertEquals(1.0f, output[0]);
        assertEquals(2.0f, output[1]);
        assertEquals(0.0f, output[2]); // Unchanged
    }

    @Test
    void testLargeArrayPerformance() {
        LinearActivator activator = LinearActivator.INSTANCE;
        int size = 10000;
        float[] input = new float[size];
        float[] output = new float[size];
        
        // Fill with various values
        for (int i = 0; i < size; i++) {
            input[i] = (float) (Math.sin(i) * 1000.0 + Math.cos(i * 0.1) * 100.0);
        }
        
        // Should not throw and should complete very quickly (linear is trivial)
        long start = System.nanoTime();
        activator.activate(input, output);
        long duration = System.nanoTime() - start;
        
        // Verify results are exact copies
        for (int i = 0; i < size; i++) {
            assertEquals(input[i], output[i], 0.0f, 
                "Should be exact copy for index " + i);
        }
        
        // Performance check - linear activation should be very fast
        assertTrue(duration < 5_000_000, "Should complete very quickly: " + duration + "ns");
    }

    @Test
    void testActivateInPlace() {
        LinearActivator activator = LinearActivator.INSTANCE;
        float[] data = {-2.5f, -1.0f, 0.0f, 1.0f, 2.5f};
        float[] expected = data.clone();
        
        // Use same array for input and output (in-place operation)
        activator.activate(data, data);
        
        // Should be unchanged (linear is identity)
        assertArrayEquals(expected, data, 0.0f, "In-place activation should not change values");
    }

    @Test
    void testDerivativeInPlace() {
        LinearActivator activator = LinearActivator.INSTANCE;
        float[] data = {-100.0f, -1.0f, 0.0f, 1.0f, 100.0f};
        float[] expected = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        
        // Use same array for input and output (in-place operation)
        activator.derivative(data, data);
        
        // Should all be 1.0
        assertArrayEquals(expected, data, 0.0f, "In-place derivative should all be 1.0");
    }

    @Test
    void testZeroLengthArrays() {
        LinearActivator activator = LinearActivator.INSTANCE;
        float[] input = {};
        float[] output = {};
        
        // Should handle empty arrays gracefully
        assertDoesNotThrow(() -> {
            activator.activate(input, output);
        }, "Should handle empty arrays");
        
        assertDoesNotThrow(() -> {
            activator.derivative(input, output);
        }, "Should handle empty arrays");
    }

    @Test
    void testSingleElementArray() {
        LinearActivator activator = LinearActivator.INSTANCE;
        float[] input = {42.0f};
        float[] output = new float[1];
        float[] derivative = new float[1];
        
        activator.activate(input, output);
        assertEquals(42.0f, output[0], 0.0f, "Single element should be preserved");
        
        activator.derivative(input, derivative);
        assertEquals(1.0f, derivative[0], 0.0f, "Single element derivative should be 1.0");
    }

    @Test
    void testConsistencyWithMathematicalDefinition() {
        LinearActivator activator = LinearActivator.INSTANCE;
        
        // Test that linear activation truly implements f(x) = x
        // and derivative implements f'(x) = 1
        
        float[] testValues = {
            -Float.MAX_VALUE, -1000.0f, -1.0f, -0.001f, 0.0f, 
            0.001f, 1.0f, 1000.0f, Float.MAX_VALUE
        };
        
        float[] outputs = new float[testValues.length];
        float[] derivatives = new float[testValues.length];
        
        activator.activate(testValues, outputs);
        activator.derivative(testValues, derivatives);
        
        for (int i = 0; i < testValues.length; i++) {
            // f(x) = x
            assertEquals(testValues[i], outputs[i], 0.0f, 
                "f(x) should equal x for index " + i);
            
            // f'(x) = 1
            assertEquals(1.0f, derivatives[i], 0.0f, 
                "f'(x) should equal 1 for index " + i);
        }
    }
}