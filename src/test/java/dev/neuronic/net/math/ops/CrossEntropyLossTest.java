package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CrossEntropyLossTest {
    
    private static final float DELTA = 1e-5f;
    
    @Test
    void testComputeBasic() {
        // Perfect prediction
        float[] trueLabels = {0.0f, 0.0f, 1.0f, 0.0f};
        float[] predictions = {0.1f, 0.1f, 0.8f, 0.0f};
        
        float loss = CrossEntropyLoss.compute(trueLabels, predictions);
        
        // Expected: -log(0.8) ≈ 0.223
        assertEquals(-Math.log(0.8), loss, DELTA);
    }
    
    @Test
    void testComputeScalar() {
        float[] trueLabels = {1.0f, 0.0f, 0.0f};
        float[] predictions = {0.7f, 0.2f, 0.1f};
        
        float loss = CrossEntropyLoss.computeScalar(trueLabels, predictions);
        
        // Expected: -log(0.7) ≈ 0.357
        assertEquals(-Math.log(0.7), loss, DELTA);
    }
    
    @Test
    void testComputeVectorized() {
        float[] trueLabels = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float[] predictions = {0.1f, 0.6f, 0.1f, 0.05f, 0.05f, 0.05f, 0.05f, 0.0f};
        
        float loss = CrossEntropyLoss.computeVectorized(trueLabels, predictions);
        
        // Expected: -log(0.6) ≈ 0.511
        assertEquals(-Math.log(0.6), loss, DELTA);
    }
    
    @Test
    void testScalarVsVectorizedConsistency() {
        float[] trueLabels = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float[] predictions = {0.05f, 0.1f, 0.7f, 0.05f, 0.03f, 0.02f, 0.03f, 0.02f};
        
        float scalarLoss = CrossEntropyLoss.computeScalar(trueLabels, predictions);
        float vectorizedLoss = CrossEntropyLoss.computeVectorized(trueLabels, predictions);
        
        assertEquals(scalarLoss, vectorizedLoss, DELTA);
    }
    
    @Test
    void testPerfectPrediction() {
        float[] trueLabels = {0.0f, 1.0f, 0.0f};
        float[] predictions = {0.0f, 1.0f, 0.0f};
        
        float loss = CrossEntropyLoss.compute(trueLabels, predictions);
        
        // Perfect prediction should have very low loss (clamped to prevent log(0))
        assertTrue(loss < 0.01f);
    }
    
    @Test
    void testWorstPrediction() {
        float[] trueLabels = {1.0f, 0.0f, 0.0f};
        float[] predictions = {0.0f, 0.5f, 0.5f}; // Wrong class gets all probability
        
        float loss = CrossEntropyLoss.compute(trueLabels, predictions);
        
        // Should be high loss due to clamping (log(1e-7))
        assertTrue(loss > 15.0f); // -log(1e-7) ≈ 16.1
    }
    
    @Test
    void testMNISTLikeScenario() {
        // 10-class classification (MNIST)
        float[] trueLabels = new float[10];
        trueLabels[7] = 1.0f; // True class is 7
        
        float[] predictions = {0.05f, 0.02f, 0.03f, 0.08f, 0.01f, 0.04f, 0.02f, 0.7f, 0.03f, 0.02f};
        
        float loss = CrossEntropyLoss.compute(trueLabels, predictions);
        
        // Good prediction: -log(0.7) ≈ 0.357
        assertEquals(-Math.log(0.7), loss, DELTA);
    }
    
    @Test
    void testUniformPrediction() {
        float[] trueLabels = {1.0f, 0.0f, 0.0f, 0.0f};
        float[] predictions = {0.25f, 0.25f, 0.25f, 0.25f}; // Uniform uncertainty
        
        float loss = CrossEntropyLoss.compute(trueLabels, predictions);
        
        // Expected: -log(0.25) = log(4) ≈ 1.386
        assertEquals(Math.log(4), loss, DELTA);
    }
    
    @Test
    void testGradientBasic() {
        float[] trueLabels = {0.0f, 1.0f, 0.0f};
        float[] predictions = {0.2f, 0.7f, 0.1f};
        float[] output = new float[3];
        
        CrossEntropyLoss.gradient(trueLabels, predictions, output);
        
        // Expected gradient: predictions - trueLabels
        assertEquals(0.2f, output[0], DELTA);  // 0.2 - 0.0
        assertEquals(-0.3f, output[1], DELTA); // 0.7 - 1.0
        assertEquals(0.1f, output[2], DELTA);  // 0.1 - 0.0
    }
    
    @Test
    void testGradientScalar() {
        float[] trueLabels = {1.0f, 0.0f, 0.0f, 0.0f};
        float[] predictions = {0.4f, 0.3f, 0.2f, 0.1f};
        float[] output = new float[4];
        
        CrossEntropyLoss.gradientScalar(trueLabels, predictions, output);
        
        assertEquals(-0.6f, output[0], DELTA); // 0.4 - 1.0
        assertEquals(0.3f, output[1], DELTA);  // 0.3 - 0.0
        assertEquals(0.2f, output[2], DELTA);  // 0.2 - 0.0
        assertEquals(0.1f, output[3], DELTA);  // 0.1 - 0.0
    }
    
    @Test
    void testGradientVectorized() {
        float[] trueLabels = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float[] predictions = {0.1f, 0.05f, 0.6f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f};
        float[] output = new float[8];
        
        CrossEntropyLoss.gradientVectorized(trueLabels, predictions, output);
        
        assertEquals(0.1f, output[0], DELTA);   // 0.1 - 0.0
        assertEquals(0.05f, output[1], DELTA);  // 0.05 - 0.0
        assertEquals(-0.4f, output[2], DELTA);  // 0.6 - 1.0
        assertEquals(0.05f, output[3], DELTA);  // 0.05 - 0.0
    }
    
    @Test
    void testGradientScalarVsVectorizedConsistency() {
        float[] trueLabels = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float[] predictions = {0.15f, 0.5f, 0.1f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f};
        float[] output1 = new float[8];
        float[] output2 = new float[8];
        
        CrossEntropyLoss.gradientScalar(trueLabels, predictions, output1);
        CrossEntropyLoss.gradientVectorized(trueLabels, predictions, output2);
        
        assertArrayEquals(output1, output2, DELTA);
    }
    
    @Test
    void testSoftmaxCrossEntropyGradient() {
        // Test that gradient is simply (predictions - trueLabels) for softmax + cross-entropy
        float[] trueLabels = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        float[] softmaxOutput = {0.1f, 0.2f, 0.1f, 0.5f, 0.1f}; // Sum = 1.0
        float[] gradients = new float[5];
        
        CrossEntropyLoss.gradient(trueLabels, softmaxOutput, gradients);
        
        // For softmax + cross-entropy, gradient is remarkably simple
        assertEquals(0.1f, gradients[0], DELTA);   // 0.1 - 0.0
        assertEquals(0.2f, gradients[1], DELTA);   // 0.2 - 0.0
        assertEquals(0.1f, gradients[2], DELTA);   // 0.1 - 0.0
        assertEquals(-0.5f, gradients[3], DELTA);  // 0.5 - 1.0
        assertEquals(0.1f, gradients[4], DELTA);   // 0.1 - 0.0
    }
    
    @Test
    void testZeroPredictionClamping() {
        // Test that zero predictions are clamped to prevent log(0)
        float[] trueLabels = {1.0f, 0.0f};
        float[] predictions = {0.0f, 1.0f}; // Zero prediction for true class
        
        float loss = CrossEntropyLoss.compute(trueLabels, predictions);
        
        // Should be clamped to 1e-7: -log(1e-7) ≈ 16.1
        assertTrue(loss > 15.0f && loss < 17.0f);
    }
    
    @Test
    void testSingleClass() {
        float[] trueLabels = {1.0f};
        float[] predictions = {0.9f};
        
        float loss = CrossEntropyLoss.compute(trueLabels, predictions);
        
        assertEquals(-Math.log(0.9), loss, DELTA);
    }
    
    @Test
    void testTwoClass() {
        float[] trueLabels = {0.0f, 1.0f};
        float[] predictions = {0.3f, 0.7f};
        
        float loss = CrossEntropyLoss.compute(trueLabels, predictions);
        
        assertEquals(-Math.log(0.7), loss, DELTA);
    }
    
    @Test
    void testLargeDimensions() {
        // Test with realistic neural network output dimensions
        int numClasses = 1000; // ImageNet-like
        float[] trueLabels = new float[numClasses];
        float[] predictions = new float[numClasses];
        
        trueLabels[42] = 1.0f; // True class
        for (int i = 0; i < numClasses; i++) {
            predictions[i] = (i == 42) ? 0.8f : 0.0002f; // Concentrate on true class
        }
        
        float loss = CrossEntropyLoss.compute(trueLabels, predictions);
        
        assertEquals(-Math.log(0.8), loss, DELTA);
    }
    
    @Test
    void testComputeDimensionMismatch() {
        float[] trueLabels = {1.0f, 0.0f};
        float[] predictions = {0.5f, 0.3f, 0.2f};
        
        assertThrows(IllegalArgumentException.class, () -> 
            CrossEntropyLoss.compute(trueLabels, predictions));
    }
    
    @Test
    void testGradientDimensionMismatch() {
        float[] trueLabels = {1.0f, 0.0f, 0.0f};
        float[] predictions = {0.5f, 0.5f};
        float[] output = new float[2];
        
        assertThrows(IllegalArgumentException.class, () -> 
            CrossEntropyLoss.gradient(trueLabels, predictions, output));
    }
    
    @Test
    void testNumericalStability() {
        // Test with very small and very large values
        float[] trueLabels = {1.0f, 0.0f, 0.0f};
        float[] predictions = {1e-10f, 0.5f, 0.5f}; // Very small probability for true class
        
        float loss = CrossEntropyLoss.compute(trueLabels, predictions);
        
        // Should be finite (clamped)
        assertTrue(Float.isFinite(loss));
        assertTrue(loss > 0.0f);
    }
}