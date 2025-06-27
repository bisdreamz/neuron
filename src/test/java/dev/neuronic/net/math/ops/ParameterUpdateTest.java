package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ParameterUpdateTest {
    
    private static final float DELTA = 1e-6f;
    
    @Test
    void testBasicUpdate() {
        float[] params = {1.0f, 2.0f, 3.0f};
        float[] gradients = {0.1f, 0.2f, 0.3f};
        float learningRate = 0.5f;
        
        ParameterUpdate.compute(params, gradients, learningRate);
        
        assertArrayEquals(new float[]{0.95f, 1.9f, 2.85f}, params, DELTA);
    }
    
    @Test
    void testScalarImplementation() {
        float[] params = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] gradients = {0.1f, 0.2f, 0.3f, 0.4f};
        float learningRate = 0.1f;
        
        ParameterUpdate.computeScalar(params, gradients, learningRate);
        
        assertArrayEquals(new float[]{0.99f, 1.98f, 2.97f, 3.96f}, params, DELTA);
    }
    
    @Test
    void testVectorizedImplementation() {
        float[] params = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] gradients = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        float learningRate = 0.1f;
        
        ParameterUpdate.computeVectorized(params, gradients, learningRate);
        
        assertArrayEquals(new float[]{0.99f, 1.98f, 2.97f, 3.96f, 4.95f, 5.94f, 6.93f, 7.92f}, params, DELTA);
    }
    
    @Test
    void testScalarVsVectorizedConsistency() {
        float[] params1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] params2 = params1.clone();
        float[] gradients = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        float learningRate = 0.05f;
        
        ParameterUpdate.computeScalar(params1, gradients, learningRate);
        ParameterUpdate.computeVectorized(params2, gradients.clone(), learningRate);
        
        assertArrayEquals(params1, params2, DELTA);
    }
    
    @Test
    void testZeroLearningRate() {
        float[] originalParams = {1.0f, 2.0f, 3.0f};
        float[] params = originalParams.clone();
        float[] gradients = {0.1f, 0.2f, 0.3f};
        
        ParameterUpdate.compute(params, gradients, 0.0f);
        
        assertArrayEquals(originalParams, params, DELTA);
    }
    
    @Test
    void testNegativeGradients() {
        float[] params = {1.0f, 2.0f, 3.0f};
        float[] gradients = {-0.1f, -0.2f, -0.3f};
        float learningRate = 0.5f;
        
        ParameterUpdate.compute(params, gradients, learningRate);
        
        // Negative gradients should increase parameters
        assertArrayEquals(new float[]{1.05f, 2.1f, 3.15f}, params, DELTA);
    }
    
    @Test
    void testDimensionMismatch() {
        float[] params = {1.0f, 2.0f};
        float[] gradients = {0.1f, 0.2f, 0.3f};
        
        assertThrows(IllegalArgumentException.class, () -> 
            ParameterUpdate.compute(params, gradients, 0.1f));
    }
}