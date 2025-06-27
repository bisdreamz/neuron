package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class OuterProductTest {
    
    private static final float DELTA = 1e-6f;
    
    @Test
    void testBasicOuterProduct() {
        float[] a = {2.0f, 3.0f};
        float[] b = {4.0f, 5.0f};
        float[][] output = new float[2][2];
        
        OuterProduct.compute(a, b, output);
        
        float[][] expected = {{8.0f, 10.0f}, {12.0f, 15.0f}};
        assertArrayEquals(expected[0], output[0], DELTA);
        assertArrayEquals(expected[1], output[1], DELTA);
    }
    
    @Test
    void testScalarImplementation() {
        float[] a = {1.0f, 2.0f, 3.0f};
        float[] b = {4.0f, 5.0f};
        float[][] output = new float[3][2];
        
        OuterProduct.computeScalar(a, b, output);
        
        float[][] expected = {{4.0f, 5.0f}, {8.0f, 10.0f}, {12.0f, 15.0f}};
        assertArrayEquals(expected[0], output[0], DELTA);
        assertArrayEquals(expected[1], output[1], DELTA);
        assertArrayEquals(expected[2], output[2], DELTA);
    }
    
    @Test
    void testVectorizedImplementation() {
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b = {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        float[][] output = new float[4][8];
        
        OuterProduct.computeVectorized(a, b, output);
        
        // Check first row: 1 * [5,6,7,8,9,10,11,12]
        assertArrayEquals(new float[]{5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, output[0], DELTA);
        // Check second row: 2 * [5,6,7,8,9,10,11,12]
        assertArrayEquals(new float[]{10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f}, output[1], DELTA);
    }
    
    @Test
    void testScalarVsVectorizedConsistency() {
        float[] a = {1.5f, 2.5f, 3.5f, 4.5f};
        float[] b = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        float[][] output1 = new float[4][8];
        float[][] output2 = new float[4][8];
        
        OuterProduct.computeScalar(a, b, output1);
        OuterProduct.computeVectorized(a, b, output2);
        
        for (int i = 0; i < 4; i++) {
            assertArrayEquals(output1[i], output2[i], DELTA);
        }
    }
    
    @Test
    void testZeroValues() {
        float[] a = {0.0f, 2.0f};
        float[] b = {3.0f, 0.0f};
        float[][] output = new float[2][2];
        
        OuterProduct.compute(a, b, output);
        
        float[][] expected = {{0.0f, 0.0f}, {6.0f, 0.0f}};
        assertArrayEquals(expected[0], output[0], DELTA);
        assertArrayEquals(expected[1], output[1], DELTA);
    }
    
    @Test
    void testNegativeValues() {
        float[] a = {-1.0f, 2.0f};
        float[] b = {3.0f, -4.0f};
        float[][] output = new float[2][2];
        
        OuterProduct.compute(a, b, output);
        
        float[][] expected = {{-3.0f, 4.0f}, {6.0f, -8.0f}};
        assertArrayEquals(expected[0], output[0], DELTA);
        assertArrayEquals(expected[1], output[1], DELTA);
    }
    
    @Test
    void testOnes() {
        float[] a = {1.0f, 1.0f, 1.0f};
        float[] b = {1.0f, 1.0f};
        float[][] output = new float[3][2];
        
        OuterProduct.compute(a, b, output);
        
        float[][] expected = {{1.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 1.0f}};
        for (int i = 0; i < 3; i++) {
            assertArrayEquals(expected[i], output[i], DELTA);
        }
    }
    
    @Test
    void testSingleElements() {
        float[] a = {5.0f};
        float[] b = {3.0f};
        float[][] output = new float[1][1];
        
        OuterProduct.compute(a, b, output);
        
        assertEquals(15.0f, output[0][0], DELTA);
    }
    
    @Test
    void testDimensionMismatch() {
        float[] a = {1.0f, 2.0f};
        float[] b = {3.0f, 4.0f};
        float[][] output = new float[3][2]; // Wrong outer dimension
        
        assertThrows(IllegalArgumentException.class, () -> 
            OuterProduct.compute(a, b, output));
    }
    
    @Test
    void testInnerDimensionMismatch() {
        float[] a = {1.0f, 2.0f};
        float[] b = {3.0f, 4.0f};
        float[][] output = new float[2][3]; // Wrong inner dimension
        
        assertThrows(IllegalArgumentException.class, () -> 
            OuterProduct.compute(a, b, output));
    }
}