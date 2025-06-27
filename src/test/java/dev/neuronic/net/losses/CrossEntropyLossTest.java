package dev.neuronic.net.losses;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CrossEntropyLossTest {
    
    private static final float DELTA = 1e-5f;
    private final CrossEntropyLoss loss = CrossEntropyLoss.INSTANCE;
    
    @Test
    void testLossBasic() {
        float[] predictions = {0.2f, 0.7f, 0.1f};
        float[] trueLabels = {0.0f, 1.0f, 0.0f}; // One-hot for class 1
        
        float lossValue = loss.loss(predictions, trueLabels);
        
        // Expected: -log(0.7) ≈ 0.357
        assertEquals(-Math.log(0.7), lossValue, DELTA);
    }
    
    @Test
    void testLossPerfectPrediction() {
        float[] predictions = {0.0f, 1.0f, 0.0f};
        float[] trueLabels = {0.0f, 1.0f, 0.0f};
        
        float lossValue = loss.loss(predictions, trueLabels);
        
        // Perfect prediction should have very low loss (due to clamping)
        assertTrue(lossValue < 0.01f);
    }
    
    @Test
    void testLossMNISTScenario() {
        // 10-class MNIST scenario
        float[] predictions = new float[10];
        float[] trueLabels = new float[10];
        
        // Softmax-like output with highest probability for class 3
        predictions[0] = 0.05f; predictions[1] = 0.03f; predictions[2] = 0.08f;
        predictions[3] = 0.6f;  predictions[4] = 0.04f; predictions[5] = 0.02f;
        predictions[6] = 0.07f; predictions[7] = 0.05f; predictions[8] = 0.03f; predictions[9] = 0.03f;
        
        trueLabels[3] = 1.0f; // True class is 3
        
        float lossValue = loss.loss(predictions, trueLabels);
        
        assertEquals(-Math.log(0.6), lossValue, DELTA);
    }
    
    @Test
    void testDerivativesBasic() {
        float[] predictions = {0.3f, 0.5f, 0.2f};
        float[] trueLabels = {0.0f, 1.0f, 0.0f};
        
        float[] derivatives = loss.derivatives(predictions, trueLabels);
        
        // Expected: predictions - trueLabels
        assertEquals(0.3f, derivatives[0], DELTA);  // 0.3 - 0.0
        assertEquals(-0.5f, derivatives[1], DELTA); // 0.5 - 1.0
        assertEquals(0.2f, derivatives[2], DELTA);  // 0.2 - 0.0
    }
    
    @Test
    void testDerivativesPerfectPrediction() {
        float[] predictions = {0.0f, 1.0f, 0.0f};
        float[] trueLabels = {0.0f, 1.0f, 0.0f};
        
        float[] derivatives = loss.derivatives(predictions, trueLabels);
        
        // Perfect prediction: gradients should be zero
        assertEquals(0.0f, derivatives[0], DELTA);
        assertEquals(0.0f, derivatives[1], DELTA);
        assertEquals(0.0f, derivatives[2], DELTA);
    }
    
    @Test
    void testDerivativesWorstPrediction() {
        float[] predictions = {0.0f, 0.0f, 1.0f}; // Wrong class
        float[] trueLabels = {1.0f, 0.0f, 0.0f}; // True class
        
        float[] derivatives = loss.derivatives(predictions, trueLabels);
        
        assertEquals(-1.0f, derivatives[0], DELTA); // 0.0 - 1.0
        assertEquals(0.0f, derivatives[1], DELTA);  // 0.0 - 0.0
        assertEquals(1.0f, derivatives[2], DELTA);  // 1.0 - 0.0
    }
    
    @Test
    void testSoftmaxCrossEntropyGradientProperty() {
        // Test the beautiful mathematical property: gradient of softmax + cross-entropy
        // is simply (predictions - trueLabels)
        float[] softmaxOutput = {0.1f, 0.6f, 0.2f, 0.1f}; // Sums to 1.0
        float[] oneHotLabels = {0.0f, 1.0f, 0.0f, 0.0f};
        
        float[] gradients = loss.derivatives(softmaxOutput, oneHotLabels);
        
        // Should be exactly predictions - labels (no complex chain rule needed!)
        assertEquals(0.1f, gradients[0], DELTA);   // 0.1 - 0.0
        assertEquals(-0.4f, gradients[1], DELTA);  // 0.6 - 1.0
        assertEquals(0.2f, gradients[2], DELTA);   // 0.2 - 0.0
        assertEquals(0.1f, gradients[3], DELTA);   // 0.1 - 0.0
    }
    
    @Test
    void testClassificationExample() {
        // Real classification scenario: network predicts digit "7" when true digit is "2"
        float[] predictions = new float[10];
        float[] trueLabels = new float[10];
        
        // Network is confident but wrong
        predictions[7] = 0.8f; // Thinks it's a 7
        predictions[2] = 0.1f; // Low confidence in true class 2
        for (int i = 0; i < 10; i++) {
            if (i != 7 && i != 2) predictions[i] = 0.0125f; // Distribute remaining 0.1
        }
        
        trueLabels[2] = 1.0f; // Actually a 2
        
        float lossValue = loss.loss(predictions, trueLabels);
        
        // High loss because network is confident but wrong: -log(0.1) ≈ 2.303
        assertEquals(-Math.log(0.1), lossValue, DELTA);
        assertTrue(lossValue > 2.0f); // Should be significantly penalized
    }
    
    @Test
    void testUncertainPrediction() {
        // Network is uncertain (uniform distribution)
        float[] predictions = {0.25f, 0.25f, 0.25f, 0.25f};
        float[] trueLabels = {1.0f, 0.0f, 0.0f, 0.0f};
        
        float lossValue = loss.loss(predictions, trueLabels);
        
        // Uncertainty penalty: -log(0.25) = log(4) ≈ 1.386
        assertEquals(Math.log(4), lossValue, DELTA);
    }
    
    @Test
    void testSingletonPattern() {
        CrossEntropyLoss instance1 = CrossEntropyLoss.INSTANCE;
        CrossEntropyLoss instance2 = CrossEntropyLoss.INSTANCE;
        
        assertSame(instance1, instance2);
    }
    
    @Test
    void testGradientDimensionConsistency() {
        float[] predictions = {0.4f, 0.3f, 0.2f, 0.1f};
        float[] trueLabels = {0.0f, 1.0f, 0.0f, 0.0f};
        
        float[] derivatives = loss.derivatives(predictions, trueLabels);
        
        assertEquals(predictions.length, derivatives.length);
        assertEquals(trueLabels.length, derivatives.length);
    }
    
    @Test
    void testLargeScale() {
        // Test with ImageNet-scale classification (1000 classes)
        int numClasses = 1000;
        float[] predictions = new float[numClasses];
        float[] trueLabels = new float[numClasses];
        
        // Realistic softmax output with concentration on true class
        trueLabels[123] = 1.0f;
        predictions[123] = 0.6f;
        float remaining = 0.4f / (numClasses - 1);
        for (int i = 0; i < numClasses; i++) {
            if (i != 123) predictions[i] = remaining;
        }
        
        float lossValue = loss.loss(predictions, trueLabels);
        float[] derivatives = loss.derivatives(predictions, trueLabels);
        
        assertEquals(-Math.log(0.6), lossValue, DELTA);
        assertEquals(numClasses, derivatives.length);
        assertEquals(-0.4f, derivatives[123], DELTA); // 0.6 - 1.0
    }
    
    @Test
    void testEdgeCases() {
        // Test with minimum and maximum realistic probabilities
        float[] predictions = {0.999f, 0.001f};
        float[] trueLabels1 = {1.0f, 0.0f}; // Best case
        float[] trueLabels2 = {0.0f, 1.0f}; // Worst case
        
        float bestLoss = loss.loss(predictions, trueLabels1);
        float worstLoss = loss.loss(predictions, trueLabels2);
        
        assertTrue(bestLoss < 0.01f);  // Very low loss for good prediction
        assertTrue(worstLoss > 6.0f);  // High loss for bad prediction: -log(0.001) ≈ 6.9
    }
}