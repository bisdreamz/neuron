package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ElementwiseMultiplyTest {
    
    private static final float DELTA = 1e-6f;
    
    @Test
    void testBasicMultiplication() {
        float[] a = {2.0f, 3.0f, 4.0f};
        float[] b = {0.5f, 2.0f, 1.5f};
        float[] output = new float[3];
        
        ElementwiseMultiply.compute(a, b, output);
        
        assertArrayEquals(new float[]{1.0f, 6.0f, 6.0f}, output, DELTA);
    }
    
    @Test
    void testScalarImplementation() {
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b = {2.0f, 3.0f, 4.0f, 5.0f};
        float[] output = new float[4];
        
        ElementwiseMultiply.computeScalar(a, b, output);
        
        assertArrayEquals(new float[]{2.0f, 6.0f, 12.0f, 20.0f}, output, DELTA);
    }
    
    @Test
    void testVectorizedImplementation() {
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] b = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
        float[] output = new float[8];
        
        ElementwiseMultiply.computeVectorized(a, b, output);
        
        assertArrayEquals(new float[]{2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f}, output, DELTA);
    }
    
    @Test
    void testScalarVsVectorizedConsistency() {
        float[] a = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f};
        float[] b = {0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.2f, 1.4f, 1.6f};
        float[] output1 = new float[8];
        float[] output2 = new float[8];
        
        ElementwiseMultiply.computeScalar(a, b, output1);
        ElementwiseMultiply.computeVectorized(a, b, output2);
        
        assertArrayEquals(output1, output2, DELTA);
    }
    
    @Test
    void testZeroValues() {
        float[] a = {0.0f, 5.0f, 0.0f};
        float[] b = {3.0f, 0.0f, 7.0f};
        float[] output = new float[3];
        
        ElementwiseMultiply.compute(a, b, output);
        
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f}, output, DELTA);
    }
    
    @Test
    void testNegativeValues() {
        float[] a = {2.0f, -3.0f, 4.0f};
        float[] b = {-1.0f, 2.0f, -1.5f};
        float[] output = new float[3];
        
        ElementwiseMultiply.compute(a, b, output);
        
        assertArrayEquals(new float[]{-2.0f, -6.0f, -6.0f}, output, DELTA);
    }
    
    @Test
    void testOnes() {
        float[] a = {2.0f, 3.0f, 4.0f};
        float[] b = {1.0f, 1.0f, 1.0f};
        float[] output = new float[3];
        
        ElementwiseMultiply.compute(a, b, output);
        
        assertArrayEquals(a, output, DELTA);
    }
    
    @Test
    void testDimensionMismatch() {
        float[] a = {1.0f, 2.0f};
        float[] b = {1.0f, 2.0f, 3.0f};
        float[] output = new float[2];
        
        assertThrows(IllegalArgumentException.class, () -> 
            ElementwiseMultiply.compute(a, b, output));
    }
    
    @Test
    void testOutputDimensionMismatch() {
        float[] a = {1.0f, 2.0f, 3.0f};
        float[] b = {1.0f, 2.0f, 3.0f};
        float[] output = new float[4];
        
        assertThrows(IllegalArgumentException.class, () -> 
            ElementwiseMultiply.compute(a, b, output));
    }
}