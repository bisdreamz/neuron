package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ScalarSubtractTest {
    
    @Test
    void testBasicScalarSubtraction() {
        float[] array = {0.2f, 0.7f, 0.9f, 0.1f};
        float[] output = new float[4];
        
        ScalarSubtract.compute(1.0f, array, output);
        
        assertArrayEquals(new float[]{0.8f, 0.3f, 0.1f, 0.9f}, output, 1e-6f);
    }
    
    @Test
    void testZeroScalar() {
        float[] array = {1.0f, 2.0f, 3.0f};
        float[] output = new float[3];
        
        ScalarSubtract.compute(0.0f, array, output);
        
        assertArrayEquals(new float[]{-1.0f, -2.0f, -3.0f}, output, 1e-6f);
    }
    
    @Test
    void testNegativeScalar() {
        float[] array = {1.0f, 2.0f, 3.0f};
        float[] output = new float[3];
        
        ScalarSubtract.compute(-5.0f, array, output);
        
        assertArrayEquals(new float[]{-6.0f, -7.0f, -8.0f}, output, 1e-6f);
    }
    
    @Test
    void testVectorizedScalarSubtraction() {
        float[] array = new float[16];
        float[] output = new float[16];
        
        for (int i = 0; i < 16; i++) {
            array[i] = i * 0.1f;
        }
        
        ScalarSubtract.computeVectorized(2.0f, array, output);
        
        for (int i = 0; i < 16; i++) {
            assertEquals(2.0f - i * 0.1f, output[i], 1e-6f);
        }
    }
    
    @Test
    void testScalarScalarSubtraction() {
        float[] array = {0.5f, 1.5f, 2.5f, 3.5f};
        float[] output = new float[4];
        
        ScalarSubtract.computeScalar(10.0f, array, output);
        
        assertArrayEquals(new float[]{9.5f, 8.5f, 7.5f, 6.5f}, output, 1e-6f);
    }
    
    @Test
    void testInPlaceScalarSubtraction() {
        float[] array = {0.1f, 0.2f, 0.3f, 0.4f};
        
        ScalarSubtract.compute(1.0f, array, array);  // In-place operation
        
        assertArrayEquals(new float[]{0.9f, 0.8f, 0.7f, 0.6f}, array, 1e-6f);
    }
    
    @Test
    void testSigmoidComplement() {
        // Test the common use case: 1.0 - sigmoid_output
        float[] sigmoidOutput = {0.1f, 0.5f, 0.9f, 0.99f};
        float[] output = new float[4];
        
        ScalarSubtract.compute(1.0f, sigmoidOutput, output);
        
        assertArrayEquals(new float[]{0.9f, 0.5f, 0.1f, 0.01f}, output, 1e-6f);
    }
    
    @Test
    void testMismatchedLengths() {
        float[] array = {1.0f, 2.0f};
        float[] output = new float[3];
        
        assertThrows(IllegalArgumentException.class, () -> {
            ScalarSubtract.compute(5.0f, array, output);
        });
    }
    
    @Test
    void testEmptyArray() {
        float[] array = {};
        float[] output = {};
        
        ScalarSubtract.compute(1.0f, array, output);
        
        assertEquals(0, output.length);
    }
}