package dev.neuronic.net.activators;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ReluActivatorTest {
    
    private static final float DELTA = 1e-6f;
    private final ReluActivator relu = ReluActivator.INSTANCE;
    
    @Test
    void testActivateBasic() {
        float[] input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        float[] output = new float[5];
        
        relu.activate(input, output);
        
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f, 1.0f, 2.0f}, output, DELTA);
    }
    
    @Test
    void testActivateScalar() {
        float[] input = {-1.5f, 0.5f, 2.5f, -0.1f};
        float[] output = new float[4];
        
        relu.activateScalar(input, output);
        
        assertArrayEquals(new float[]{0.0f, 0.5f, 2.5f, 0.0f}, output, DELTA);
    }
    
    @Test
    void testActivateVectorized() {
        float[] input = {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 5.0f, -2.0f, 3.0f};
        float[] output = new float[8];
        
        relu.activateVectorized(input, output);
        
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 5.0f, 0.0f, 3.0f}, output, DELTA);
    }
    
    @Test
    void testDerivativeBasic() {
        float[] input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        float[] output = new float[5];
        
        relu.derivative(input, output);
        
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f, 1.0f, 1.0f}, output, DELTA);
    }
    
    @Test
    void testDerivativeScalar() {
        float[] input = {-1.5f, 0.5f, 2.5f, -0.1f, 0.0f};
        float[] output = new float[5];
        
        relu.derivativeScalar(input, output);
        
        assertArrayEquals(new float[]{0.0f, 1.0f, 1.0f, 0.0f, 0.0f}, output, DELTA);
    }
    
    @Test
    void testDerivativeVectorized() {
        float[] input = {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 5.0f, -2.0f, 3.0f};
        float[] output = new float[8];
        
        relu.derivativeVectorized(input, output);
        
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f}, output, DELTA);
    }
    
    @Test
    void testActivateVsScalarConsistency() {
        float[] input = {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 5.0f, -2.0f, 3.0f};
        float[] output1 = new float[8];
        float[] output2 = new float[8];
        
        relu.activateScalar(input, output1);
        relu.activateVectorized(input, output2);
        
        assertArrayEquals(output1, output2, DELTA);
    }
    
    @Test
    void testDerivativeVsScalarConsistency() {
        float[] input = {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 5.0f, -2.0f, 3.0f};
        float[] output1 = new float[8];
        float[] output2 = new float[8];
        
        relu.derivativeScalar(input, output1);
        relu.derivativeVectorized(input, output2);
        
        assertArrayEquals(output1, output2, DELTA);
    }
    
    @Test
    void testActivateDimensionMismatch() {
        float[] input = {1.0f, 2.0f};
        float[] output = new float[3];
        
        assertThrows(IllegalArgumentException.class, () -> 
            relu.activate(input, output));
    }
    
    @Test
    void testDerivativeDimensionMismatch() {
        float[] input = {1.0f, 2.0f, 3.0f};
        float[] output = new float[2];
        
        assertThrows(IllegalArgumentException.class, () -> 
            relu.derivative(input, output));
    }
    
    @Test
    void testZeroInput() {
        float[] input = {0.0f, 0.0f, 0.0f};
        float[] activationOutput = new float[3];
        float[] derivativeOutput = new float[3];
        
        relu.activate(input, activationOutput);
        relu.derivative(input, derivativeOutput);
        
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f}, activationOutput, DELTA);
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f}, derivativeOutput, DELTA);
    }
}