package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MeanSquaredErrorLossTest {
    
    private static final float DELTA = 1e-6f;
    
    @Test
    void testBasicLossComputation() {
        float[] predictions = {1.0f, 2.0f, 3.0f};
        float[] targets = {0.0f, 1.0f, 2.0f};
        
        float loss = MeanSquaredErrorLoss.computeLoss(predictions, targets);
        
        // ((1-0)^2 + (2-1)^2 + (3-2)^2) / 3 = (1 + 1 + 1) / 3 = 1.0
        assertEquals(1.0f, loss, DELTA);
    }
    
    @Test
    void testLossScalarImplementation() {
        float[] predictions = {5.0f, 0.0f};
        float[] targets = {0.0f, 5.0f};
        
        float loss = MeanSquaredErrorLoss.computeLossScalar(predictions, targets);
        
        // ((5-0)^2 + (0-5)^2) / 2 = (25 + 25) / 2 = 25.0
        assertEquals(25.0f, loss, DELTA);
    }
    
    @Test
    void testLossVectorizedImplementation() {
        float[] predictions = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] targets = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        
        float loss = MeanSquaredErrorLoss.computeLossVectorized(predictions, targets);
        
        // (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2) / 8 = (1+4+9+16+25+36+49+64) / 8 = 204/8 = 25.5
        assertEquals(25.5f, loss, DELTA);
    }
    
    @Test
    void testLossScalarVsVectorizedConsistency() {
        float[] predictions = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f};
        float[] targets = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        
        float loss1 = MeanSquaredErrorLoss.computeLossScalar(predictions, targets);
        float loss2 = MeanSquaredErrorLoss.computeLossVectorized(predictions, targets);
        
        assertEquals(loss1, loss2, DELTA);
    }
    
    @Test
    void testBasicDerivatives() {
        float[] predictions = {1.0f, 2.0f, 3.0f};
        float[] targets = {0.0f, 1.0f, 2.0f};
        float[] output = new float[3];
        
        MeanSquaredErrorLoss.computeDerivatives(predictions, targets, output);
        
        // dMSE/dpred[i] = (2/n) * (pred[i] - target[i])
        // (2/3) * (1-0) = 2/3, (2/3) * (2-1) = 2/3, (2/3) * (3-2) = 2/3
        assertArrayEquals(new float[]{2.0f/3, 2.0f/3, 2.0f/3}, output, DELTA);
    }
    
    @Test
    void testDerivativesScalarImplementation() {
        float[] predictions = {2.0f, 1.0f};
        float[] targets = {1.0f, 2.0f};
        float[] output = new float[2];
        
        MeanSquaredErrorLoss.computeDerivativesScalar(predictions, targets, output);
        
        // (2/2) * (2-1) = 1.0, (2/2) * (1-2) = -1.0
        assertArrayEquals(new float[]{1.0f, -1.0f}, output, DELTA);
    }
    
    @Test
    void testDerivativesVectorizedImplementation() {
        float[] predictions = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] targets = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
        float[] output = new float[8];
        
        MeanSquaredErrorLoss.computeDerivativesVectorized(predictions, targets, output);
        
        // All differences are 1.0, so all derivatives should be (2/8) * 1.0 = 0.25
        assertArrayEquals(new float[]{0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f}, output, DELTA);
    }
    
    @Test
    void testDerivativesScalarVsVectorizedConsistency() {
        float[] predictions = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f};
        float[] targets = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] output1 = new float[8];
        float[] output2 = new float[8];
        
        MeanSquaredErrorLoss.computeDerivativesScalar(predictions, targets, output1);
        MeanSquaredErrorLoss.computeDerivativesVectorized(predictions, targets, output2);
        
        assertArrayEquals(output1, output2, DELTA);
    }
    
    @Test
    void testPerfectPredictions() {
        float[] predictions = {1.0f, 2.0f, 3.0f};
        float[] targets = {1.0f, 2.0f, 3.0f};
        float[] output = new float[3];
        
        float loss = MeanSquaredErrorLoss.computeLoss(predictions, targets);
        MeanSquaredErrorLoss.computeDerivatives(predictions, targets, output);
        
        assertEquals(0.0f, loss, DELTA);
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f}, output, DELTA);
    }
    
    @Test
    void testSingleElement() {
        float[] predictions = {3.0f};
        float[] targets = {1.0f};
        float[] output = new float[1];
        
        float loss = MeanSquaredErrorLoss.computeLoss(predictions, targets);
        MeanSquaredErrorLoss.computeDerivatives(predictions, targets, output);
        
        // Loss: (3-1)^2 / 1 = 4.0
        assertEquals(4.0f, loss, DELTA);
        
        // Derivative: (2/1) * (3-1) = 4.0
        assertArrayEquals(new float[]{4.0f}, output, DELTA);
    }
    
    @Test
    void testLossDimensionMismatch() {
        float[] predictions = {1.0f, 2.0f};
        float[] targets = {1.0f, 2.0f, 3.0f};
        
        assertThrows(IllegalArgumentException.class, () -> 
            MeanSquaredErrorLoss.computeLoss(predictions, targets));
    }
    
    @Test
    void testDerivativesDimensionMismatch() {
        float[] predictions = {1.0f, 2.0f, 3.0f};
        float[] targets = {1.0f, 2.0f};
        float[] output = new float[3];
        
        assertThrows(IllegalArgumentException.class, () -> 
            MeanSquaredErrorLoss.computeDerivatives(predictions, targets, output));
    }
    
    @Test
    void testDerivativesOutputDimensionMismatch() {
        float[] predictions = {1.0f, 2.0f, 3.0f};
        float[] targets = {1.0f, 2.0f, 3.0f};
        float[] output = new float[2];
        
        assertThrows(IllegalArgumentException.class, () -> 
            MeanSquaredErrorLoss.computeDerivatives(predictions, targets, output));
    }
}