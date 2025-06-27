package dev.neuronic.net.losses;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MseLossTest {
    
    private static final float DELTA = 1e-6f;
    private final MseLoss mse = MseLoss.INSTANCE;
    
    @Test
    void testPerfectPrediction() {
        float[] prediction = {1.0f, 2.0f, 3.0f};
        float[] labels = {1.0f, 2.0f, 3.0f};
        
        float loss = mse.loss(prediction, labels);
        
        assertEquals(0.0f, loss, DELTA);
    }
    
    @Test
    void testBasicLoss() {
        float[] prediction = {1.0f, 2.0f, 3.0f};
        float[] labels = {0.0f, 1.0f, 2.0f};
        
        float loss = mse.loss(prediction, labels);
        
        // ((1-0)^2 + (2-1)^2 + (3-2)^2) / 3 = (1 + 1 + 1) / 3 = 1.0
        assertEquals(1.0f, loss, DELTA);
    }
    
    @Test
    void testLargerErrors() {
        float[] prediction = {5.0f, 0.0f};
        float[] labels = {0.0f, 5.0f};
        
        float loss = mse.loss(prediction, labels);
        
        // ((5-0)^2 + (0-5)^2) / 2 = (25 + 25) / 2 = 25.0
        assertEquals(25.0f, loss, DELTA);
    }
    
    @Test
    void testPerfectDerivatives() {
        float[] prediction = {1.0f, 2.0f, 3.0f};
        float[] labels = {1.0f, 2.0f, 3.0f};
        
        float[] derivatives = mse.derivatives(prediction, labels);
        
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f}, derivatives, DELTA);
    }
    
    @Test
    void testBasicDerivatives() {
        float[] prediction = {1.0f, 2.0f, 3.0f};
        float[] labels = {0.0f, 1.0f, 2.0f};
        
        float[] derivatives = mse.derivatives(prediction, labels);
        
        // dMSE/dpred[i] = (2/n) * (pred[i] - label[i])
        // (2/3) * (1-0) = 2/3, (2/3) * (2-1) = 2/3, (2/3) * (3-2) = 2/3
        assertArrayEquals(new float[]{2.0f/3, 2.0f/3, 2.0f/3}, derivatives, DELTA);
    }
    
    @Test
    void testNegativeDerivatives() {
        float[] prediction = {0.0f, 1.0f, 2.0f};
        float[] labels = {1.0f, 2.0f, 3.0f};
        
        float[] derivatives = mse.derivatives(prediction, labels);
        
        // (2/3) * (0-1) = -2/3, (2/3) * (1-2) = -2/3, (2/3) * (2-3) = -2/3
        assertArrayEquals(new float[]{-2.0f/3, -2.0f/3, -2.0f/3}, derivatives, DELTA);
    }
    
    @Test
    void testMixedDerivatives() {
        float[] prediction = {2.0f, 1.0f};
        float[] labels = {1.0f, 2.0f};
        
        float[] derivatives = mse.derivatives(prediction, labels);
        
        // (2/2) * (2-1) = 1.0, (2/2) * (1-2) = -1.0
        assertArrayEquals(new float[]{1.0f, -1.0f}, derivatives, DELTA);
    }
    
    @Test
    void testSingleElement() {
        float[] prediction = {3.0f};
        float[] labels = {1.0f};
        
        float loss = mse.loss(prediction, labels);
        float[] derivatives = mse.derivatives(prediction, labels);
        
        // Loss: (3-1)^2 / 1 = 4.0
        assertEquals(4.0f, loss, DELTA);
        
        // Derivative: (2/1) * (3-1) = 4.0
        assertArrayEquals(new float[]{4.0f}, derivatives, DELTA);
    }
    
    @Test
    void testLossDimensionMismatch() {
        float[] prediction = {1.0f, 2.0f};
        float[] labels = {1.0f, 2.0f, 3.0f};
        
        assertThrows(IllegalArgumentException.class, () -> 
            mse.loss(prediction, labels));
    }
    
    @Test
    void testDerivativesDimensionMismatch() {
        float[] prediction = {1.0f, 2.0f, 3.0f};
        float[] labels = {1.0f, 2.0f};
        
        assertThrows(IllegalArgumentException.class, () -> 
            mse.derivatives(prediction, labels));
    }
    
    @Test
    void testZeroValues() {
        float[] prediction = {0.0f, 0.0f, 0.0f};
        float[] labels = {0.0f, 0.0f, 0.0f};
        
        float loss = mse.loss(prediction, labels);
        float[] derivatives = mse.derivatives(prediction, labels);
        
        assertEquals(0.0f, loss, DELTA);
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f}, derivatives, DELTA);
    }
}