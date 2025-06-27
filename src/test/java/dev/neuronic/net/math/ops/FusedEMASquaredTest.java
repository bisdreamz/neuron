package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FusedEMASquaredTest {
    
    private static final float EPSILON = 1e-6f;
    
    @Test
    public void testFusedEMASquared() {
        float[] state = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] gradients = {0.1f, -0.2f, 0.3f, -0.4f};
        float beta = 0.9f;
        
        // Expected calculation: state = beta * state + (1 - beta) * gradient²
        float[] expected = new float[state.length];
        for (int i = 0; i < state.length; i++) {
            expected[i] = beta * state[i] + (1 - beta) * gradients[i] * gradients[i];
        }
        
        // Clone for testing
        float[] stateToUpdate = state.clone();
        
        // Apply fused operation
        FusedEMASquared.compute(stateToUpdate, gradients, beta);
        
        assertArrayEquals(expected, stateToUpdate, EPSILON);
    }
    
    @Test
    public void testScalarVsVectorized() {
        int[] sizes = {4, 16, 64, 256};
        
        for (int size : sizes) {
            float[] stateScalar = new float[size];
            float[] stateVectorized = new float[size];
            float[] gradients = new float[size];
            
            // Initialize with random values
            for (int i = 0; i < size; i++) {
                float value = (float)(Math.random() * 2);
                stateScalar[i] = stateVectorized[i] = value;
                gradients[i] = (float)(Math.random() * 0.2 - 0.1);
            }
            
            float beta = 0.999f;
            
            // Compute with scalar implementation
            FusedEMASquared.computeScalar(stateScalar, gradients, beta);
            
            // Compute with vectorized implementation
            FusedEMASquared.computeVectorized(stateVectorized, gradients, beta);
            
            assertArrayEquals(stateScalar, stateVectorized, EPSILON,
                            "Scalar and vectorized results should match for size " + size);
        }
    }
    
    @Test
    public void testBetaBounds() {
        float[] state = {1.0f};
        float[] gradients = {0.5f};
        
        // Test beta = 0 (no memory)
        float[] state0 = state.clone();
        FusedEMASquared.compute(state0, gradients, 0.0f);
        assertEquals(0.25f, state0[0], EPSILON); // Should be gradient²
        
        // Test beta close to 1 (strong memory)
        float[] state1 = state.clone();
        FusedEMASquared.compute(state1, gradients, 0.99999f);
        assertTrue(Math.abs(state1[0] - 1.0f) < 0.001f); // Should be very close to original
    }
    
    @Test
    public void testValidation() {
        // Test array length mismatch
        assertThrows(IllegalArgumentException.class, () -> {
            FusedEMASquared.compute(new float[4], new float[3], 0.9f);
        });
        
        // Test invalid beta
        assertThrows(IllegalArgumentException.class, () -> {
            FusedEMASquared.compute(new float[4], new float[4], -0.1f);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            FusedEMASquared.compute(new float[4], new float[4], 1.0f);
        });
    }
}