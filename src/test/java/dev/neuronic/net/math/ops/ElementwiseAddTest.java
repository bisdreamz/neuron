package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ElementwiseAddTest {
    
    private static final float DELTA = 1e-6f;
    
    @Test
    void testBasicAddition() {
        float[] a = {1.0f, 2.0f, 3.0f};
        float[] b = {0.5f, 1.5f, 2.5f};
        float[] output = new float[3];
        
        ElementwiseAdd.compute(a, b, output);
        
        assertArrayEquals(new float[]{1.5f, 3.5f, 5.5f}, output, DELTA);
    }
    
    @Test
    void testScalarImplementation() {
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b = {0.1f, 0.2f, 0.3f, 0.4f};
        float[] output = new float[4];
        
        ElementwiseAdd.computeScalar(a, b, output);
        
        assertArrayEquals(new float[]{1.1f, 2.2f, 3.3f, 4.4f}, output, DELTA);
    }
    
    @Test
    void testVectorizedImplementation() {
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] b = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        float[] output = new float[8];
        
        ElementwiseAdd.computeVectorized(a, b, output);
        
        assertArrayEquals(new float[]{1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f}, output, DELTA);
    }
    
    @Test
    void testScalarVsVectorizedConsistency() {
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] b = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        float[] output1 = new float[8];
        float[] output2 = new float[8];
        
        ElementwiseAdd.computeScalar(a, b, output1);
        ElementwiseAdd.computeVectorized(a, b, output2);
        
        assertArrayEquals(output1, output2, DELTA);
    }
    
    @Test
    void testZeroValues() {
        float[] a = {0.0f, 0.0f, 0.0f};
        float[] b = {1.0f, 2.0f, 3.0f};
        float[] output = new float[3];
        
        ElementwiseAdd.compute(a, b, output);
        
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f}, output, DELTA);
    }
    
    @Test
    void testNegativeValues() {
        float[] a = {1.0f, -2.0f, 3.0f};
        float[] b = {-0.5f, 1.5f, -2.5f};
        float[] output = new float[3];
        
        ElementwiseAdd.compute(a, b, output);
        
        assertArrayEquals(new float[]{0.5f, -0.5f, 0.5f}, output, DELTA);
    }
    
    @Test
    void testDimensionMismatch() {
        float[] a = {1.0f, 2.0f};
        float[] b = {1.0f, 2.0f, 3.0f};
        float[] output = new float[2];
        
        assertThrows(IllegalArgumentException.class, () -> 
            ElementwiseAdd.compute(a, b, output));
    }
    
    @Test
    void testOutputDimensionMismatch() {
        float[] a = {1.0f, 2.0f, 3.0f};
        float[] b = {1.0f, 2.0f, 3.0f};
        float[] output = new float[2];
        
        assertThrows(IllegalArgumentException.class, () -> 
            ElementwiseAdd.compute(a, b, output));
    }
}