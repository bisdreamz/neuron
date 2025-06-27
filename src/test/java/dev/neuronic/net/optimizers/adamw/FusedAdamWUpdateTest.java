package dev.neuronic.net.optimizers.adamw;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FusedAdamWUpdateTest {
    
    private static final float EPSILON = 1e-6f;
    
    @Test
    public void testFusedAdamWUpdateMatchesStepByStep() {
        // Test data
        float[] params = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] gradients = {0.1f, -0.2f, 0.3f, -0.4f};
        float[] momentum = {0.05f, -0.1f, 0.15f, -0.2f};
        float[] velocity = {0.01f, 0.02f, 0.03f, 0.04f};
        
        // Create copies for step-by-step calculation
        float[] paramsStepByStep = params.clone();
        float[] momentumStepByStep = momentum.clone();
        float[] velocityStepByStep = velocity.clone();
        
        // AdamW parameters
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float learningRate = 0.001f;
        float epsilon = 1e-8f;
        float weightDecay = 0.01f;
        float momentumCorrection = 0.1f; // 1 - beta1^t for t=1
        float velocityCorrection = 0.001f; // 1 - beta2^t for t=1
        
        // Step-by-step calculation
        for (int i = 0; i < paramsStepByStep.length; i++) {
            // Update momentum
            momentumStepByStep[i] = beta1 * momentumStepByStep[i] + (1 - beta1) * gradients[i];
            
            // Update velocity
            velocityStepByStep[i] = beta2 * velocityStepByStep[i] + (1 - beta2) * gradients[i] * gradients[i];
            
            // Bias correction
            float mHat = momentumStepByStep[i] / momentumCorrection;
            float vHat = velocityStepByStep[i] / velocityCorrection;
            
            // Parameter update
            paramsStepByStep[i] -= learningRate * mHat / (float)(Math.sqrt(vHat) + epsilon);
            
            // Weight decay
            paramsStepByStep[i] *= (1 - weightDecay);
        }
        
        // Fused calculation
        float[] paramsFused = params.clone();
        float[] momentumFused = momentum.clone();
        float[] velocityFused = velocity.clone();
        
        FusedAdamWUpdate.compute(paramsFused, gradients, momentumFused, velocityFused,
                               beta1, beta2, learningRate, epsilon, weightDecay,
                               momentumCorrection, velocityCorrection, true);
        
        // Compare results
        assertArrayEquals(paramsStepByStep, paramsFused, EPSILON, "Parameters should match");
        assertArrayEquals(momentumStepByStep, momentumFused, EPSILON, "Momentum should match");
        assertArrayEquals(velocityStepByStep, velocityFused, EPSILON, "Velocity should match");
    }
    
    @Test
    public void testWithoutWeightDecay() {
        float[] params = {1.0f, 2.0f};
        float[] gradients = {0.1f, -0.2f};
        float[] momentum = new float[2];
        float[] velocity = new float[2];
        
        FusedAdamWUpdate.compute(params, gradients, momentum, velocity,
                               0.9f, 0.999f, 0.001f, 1e-8f, 0.01f,
                               0.1f, 0.001f, false); // applyWeightDecay = false
        
        // Should not apply weight decay
        assertTrue(params[0] < 1.0f); // Should decrease due to positive gradient
        assertTrue(params[1] > 2.0f); // Should increase due to negative gradient
    }
    
    @Test
    public void testScalarVsVectorized() {
        // Test with different sizes to ensure both paths work
        int[] sizes = {4, 16, 64, 256};
        
        for (int size : sizes) {
            float[] paramsScalar = new float[size];
            float[] paramsVectorized = new float[size];
            float[] gradients = new float[size];
            float[] momentumScalar = new float[size];
            float[] momentumVectorized = new float[size];
            float[] velocityScalar = new float[size];
            float[] velocityVectorized = new float[size];
            
            // Initialize with random values
            for (int i = 0; i < size; i++) {
                float value = (float)(Math.random() * 2 - 1);
                paramsScalar[i] = paramsVectorized[i] = value;
                gradients[i] = (float)(Math.random() * 0.2 - 0.1);
                momentumScalar[i] = momentumVectorized[i] = (float)(Math.random() * 0.1 - 0.05);
                velocityScalar[i] = velocityVectorized[i] = (float)(Math.random() * 0.01);
            }
            
            // Compute with scalar implementation
            FusedAdamWUpdate.computeScalar(paramsScalar, gradients, momentumScalar, velocityScalar,
                                         0.9f, 0.999f, 0.001f, 1e-8f, 0.01f,
                                         0.1f, 0.001f, true);
            
            // Compute with vectorized implementation (if available)
            FusedAdamWUpdate.computeVectorized(paramsVectorized, gradients, momentumVectorized, velocityVectorized,
                                             0.9f, 0.999f, 0.001f, 1e-8f, 0.01f,
                                             0.1f, 0.001f, true);
            
            // Results should be identical
            assertArrayEquals(paramsScalar, paramsVectorized, EPSILON, 
                            "Scalar and vectorized results should match for size " + size);
            assertArrayEquals(momentumScalar, momentumVectorized, EPSILON,
                            "Momentum should match for size " + size);
            assertArrayEquals(velocityScalar, velocityVectorized, EPSILON,
                            "Velocity should match for size " + size);
        }
    }
    
    @Test
    public void testEdgeCases() {
        // Test with zero gradients
        float[] params = {1.0f, 2.0f};
        float[] zeroGradients = {0.0f, 0.0f};
        float[] momentum = {0.1f, 0.2f};
        float[] velocity = {0.01f, 0.02f};
        
        float[] originalParams = params.clone();
        
        FusedAdamWUpdate.compute(params, zeroGradients, momentum, velocity,
                               0.9f, 0.999f, 0.001f, 1e-8f, 0.01f,
                               0.1f, 0.001f, true);
        
        // With zero gradients, only weight decay should affect parameters
        for (int i = 0; i < params.length; i++) {
            float expected = originalParams[i] * 0.99f; // 1 - weightDecay
            // Account for small changes due to existing momentum
            assertTrue(Math.abs(params[i] - expected) < 0.01f,
                      "Parameters should only be affected by weight decay and existing momentum");
        }
    }
    
    @Test
    public void testArrayLengthValidation() {
        float[] params = new float[4];
        float[] gradients = new float[3]; // Wrong size
        float[] momentum = new float[4];
        float[] velocity = new float[4];
        
        assertThrows(IllegalArgumentException.class, () -> {
            FusedAdamWUpdate.compute(params, gradients, momentum, velocity,
                                   0.9f, 0.999f, 0.001f, 1e-8f, 0.01f,
                                   0.1f, 0.001f, true);
        });
    }
}