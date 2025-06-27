package dev.neuronic.net.activators;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class LeakyReluActivatorTest {
    
    private static final float EPSILON = 1e-6f;
    
    @Test
    public void testDefaultAlpha() {
        LeakyReluActivator activator = LeakyReluActivator.createDefault();
        
        float[] input = {2.0f, -2.0f, 0.0f, 0.5f, -0.5f};
        float[] output = new float[5];
        
        activator.activate(input, output);
        
        // Positive values should pass through unchanged
        assertEquals(2.0f, output[0], EPSILON);
        assertEquals(0.5f, output[3], EPSILON);
        
        // Negative values should be multiplied by alpha (0.01)
        assertEquals(-0.02f, output[1], EPSILON);
        assertEquals(-0.005f, output[4], EPSILON);
        
        // Zero should remain zero
        assertEquals(0.0f, output[2], EPSILON);
    }
    
    @Test
    public void testCustomAlpha() {
        LeakyReluActivator activator = new LeakyReluActivator(0.2f);
        
        float[] input = {3.0f, -5.0f, -1.0f};
        float[] output = new float[3];
        
        activator.activate(input, output);
        
        assertEquals(3.0f, output[0], EPSILON);
        assertEquals(-1.0f, output[1], EPSILON);  // -5.0 * 0.2
        assertEquals(-0.2f, output[2], EPSILON);  // -1.0 * 0.2
    }
    
    @Test
    public void testDerivative() {
        LeakyReluActivator activator = new LeakyReluActivator(0.1f);
        
        float[] input = {2.0f, -2.0f, 0.0f, 0.5f, -0.5f};
        float[] output = new float[5];
        
        activator.derivative(input, output);
        
        // Derivative is 1 for positive values
        assertEquals(1.0f, output[0], EPSILON);
        assertEquals(1.0f, output[3], EPSILON);
        
        // Derivative is alpha for negative values
        assertEquals(0.1f, output[1], EPSILON);
        assertEquals(0.1f, output[4], EPSILON);
        
        // Derivative at zero is technically undefined, but we use alpha
        assertEquals(0.1f, output[2], EPSILON);
    }
    
    @Test
    public void testInvalidAlpha() {
        assertThrows(IllegalArgumentException.class, () -> new LeakyReluActivator(0.0f));
        assertThrows(IllegalArgumentException.class, () -> new LeakyReluActivator(-0.1f));
        assertThrows(IllegalArgumentException.class, () -> new LeakyReluActivator(1.0f));
        assertThrows(IllegalArgumentException.class, () -> new LeakyReluActivator(1.5f));
    }
    
    @Test
    public void testFactoryMethods() {
        // Test default factory method
        LeakyReluActivator default1 = LeakyReluActivator.createDefault();
        LeakyReluActivator default2 = LeakyReluActivator.createDefault();
        assertNotNull(default1);
        assertNotNull(default2);
        
        // Test custom alpha factory
        LeakyReluActivator custom1 = LeakyReluActivator.create(0.15f);
        LeakyReluActivator custom2 = LeakyReluActivator.create(0.15f);
        assertNotNull(custom1);
        assertNotNull(custom2);
    }
    
    @Test
    public void testLargeArray() {
        LeakyReluActivator activator = LeakyReluActivator.createDefault();
        
        int size = 10000;
        float[] input = new float[size];
        float[] output = new float[size];
        
        // Fill with alternating positive and negative values
        for (int i = 0; i < size; i++) {
            input[i] = (i % 2 == 0) ? (i * 0.001f) : (-i * 0.001f);
        }
        
        activator.activate(input, output);
        
        // Spot check some values
        assertEquals(0.0f, output[0], EPSILON);  // 0 * 0.001
        assertEquals(-0.00001f, output[1], EPSILON);  // -1 * 0.001 * 0.01
        assertEquals(10.0f * 0.001f, output[10], EPSILON);  // 10 * 0.001
        assertEquals(-11.0f * 0.001f * 0.01f, output[11], EPSILON);  // -11 * 0.001 * 0.01
    }
    
}