package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ElementwiseSubtractTest {
    
    @Test
    void testBasicSubtraction() {
        float[] a = {5.0f, 3.0f, 8.0f, 2.0f};
        float[] b = {2.0f, 1.0f, 3.0f, 1.0f};
        float[] output = new float[4];
        
        ElementwiseSubtract.compute(a, b, output);
        
        assertArrayEquals(new float[]{3.0f, 2.0f, 5.0f, 1.0f}, output, 1e-6f);
    }
    
    @Test
    void testNegativeResults() {
        float[] a = {1.0f, 2.0f, 0.0f};
        float[] b = {3.0f, 1.0f, 2.0f};
        float[] output = new float[3];
        
        ElementwiseSubtract.compute(a, b, output);
        
        assertArrayEquals(new float[]{-2.0f, 1.0f, -2.0f}, output, 1e-6f);
    }
    
    @Test
    void testZeroSubtraction() {
        float[] a = {1.0f, 2.0f, 3.0f};
        float[] b = {0.0f, 0.0f, 0.0f};
        float[] output = new float[3];
        
        ElementwiseSubtract.compute(a, b, output);
        
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f}, output, 1e-6f);
    }
    
    @Test
    void testVectorizedSubtraction() {
        float[] a = new float[16];
        float[] b = new float[16];
        float[] output = new float[16];
        
        for (int i = 0; i < 16; i++) {
            a[i] = i * 2.0f;
            b[i] = i;
        }
        
        ElementwiseSubtract.computeVectorized(a, b, output);
        
        for (int i = 0; i < 16; i++) {
            assertEquals(i, output[i], 1e-6f);
        }
    }
    
    @Test
    void testScalarSubtraction() {
        float[] a = {10.0f, 20.0f, 30.0f, 40.0f};
        float[] b = {5.0f, 10.0f, 15.0f, 20.0f};
        float[] output = new float[4];
        
        ElementwiseSubtract.computeScalar(a, b, output);
        
        assertArrayEquals(new float[]{5.0f, 10.0f, 15.0f, 20.0f}, output, 1e-6f);
    }
    
    @Test
    void testInPlaceSubtraction() {
        float[] a = {10.0f, 8.0f, 6.0f, 4.0f};
        float[] b = {1.0f, 2.0f, 3.0f, 4.0f};
        
        ElementwiseSubtract.compute(a, b, a);  // In-place operation
        
        assertArrayEquals(new float[]{9.0f, 6.0f, 3.0f, 0.0f}, a, 1e-6f);
    }
    
    @Test
    void testMismatchedLengths() {
        float[] a = {1.0f, 2.0f};
        float[] b = {1.0f, 2.0f, 3.0f};
        float[] output = new float[2];
        
        assertThrows(IllegalArgumentException.class, () -> {
            ElementwiseSubtract.compute(a, b, output);
        });
    }
    
    @Test
    void testEmptyArrays() {
        float[] a = {};
        float[] b = {};
        float[] output = {};
        
        ElementwiseSubtract.compute(a, b, output);
        
        assertEquals(0, output.length);
    }
}