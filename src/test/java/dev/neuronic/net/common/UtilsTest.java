package dev.neuronic.net.common;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class UtilsTest {
    
    private static final float DELTA = 1e-6f;
    
    // ========== FLATTEN INT MATRIX TESTS ==========
    
    @Test
    void testFlattenIntMatrixBasic() {
        int[][] matrix = {{1, 2, 3}, {4, 5, 6}};
        float[] result = Utils.flatten(matrix, false);
        
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, result, DELTA);
    }
    
    @Test
    void testFlattenIntMatrixWithNormalization() {
        int[][] matrix = {{0, 255}, {128, 64}};
        float[] result = Utils.flatten(matrix, true);
        
        // 0/255=0.0, 255/255=1.0, 128/255≈0.502, 64/255≈0.251
        assertArrayEquals(new float[]{0.0f, 1.0f, 128/255.0f, 64/255.0f}, result, DELTA);
    }
    
    @Test
    void testFlattenIntMatrixSingleElement() {
        int[][] matrix = {{42}};
        float[] result = Utils.flatten(matrix, false);
        
        assertArrayEquals(new float[]{42.0f}, result, DELTA);
    }
    
    @Test
    void testFlattenIntMatrixMNISTSize() {
        int[][] matrix = new int[28][28];
        // Fill with test pattern
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                matrix[i][j] = (i * 28 + j) % 256;
            }
        }
        
        float[] result = Utils.flatten(matrix, true);
        
        assertEquals(784, result.length); // 28*28
        assertEquals(0.0f, result[0], DELTA); // First element
        assertEquals(1.0f, result[255], DELTA); // Element with value 255
    }
    
    @Test
    void testFlattenIntMatrixNullInput() {
        assertThrows(IllegalArgumentException.class, () -> 
            Utils.flatten((int[][])null, false));
    }
    
    @Test
    void testFlattenIntMatrixEmptyInput() {
        int[][] matrix = {};
        assertThrows(IllegalArgumentException.class, () -> 
            Utils.flatten(matrix, false));
    }
    
    @Test
    void testFlattenIntMatrixJaggedArray() {
        int[][] matrix = {{1, 2, 3}, {4, 5}}; // Different row lengths
        assertThrows(IllegalArgumentException.class, () -> 
            Utils.flatten(matrix, false));
    }
    
    // ========== FLATTEN FLOAT MATRIX TESTS ==========
    
    @Test
    void testFlattenFloatMatrixBasic() {
        float[][] matrix = {{1.5f, 2.5f}, {3.5f, 4.5f}};
        float[] result = Utils.flatten(matrix);
        
        assertArrayEquals(new float[]{1.5f, 2.5f, 3.5f, 4.5f}, result, DELTA);
    }
    
    @Test
    void testFlattenFloatMatrixSingleRow() {
        float[][] matrix = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}};
        float[] result = Utils.flatten(matrix);
        
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, result, DELTA);
    }
    
    @Test
    void testFlattenFloatMatrixSingleColumn() {
        float[][] matrix = {{1.0f}, {2.0f}, {3.0f}};
        float[] result = Utils.flatten(matrix);
        
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f}, result, DELTA);
    }
    
    @Test
    void testFlattenFloatMatrixNullInput() {
        assertThrows(IllegalArgumentException.class, () -> 
            Utils.flatten((float[][]) null));
    }
    
    @Test
    void testFlattenFloatMatrixEmptyInput() {
        float[][] matrix = {};
        assertThrows(IllegalArgumentException.class, () -> 
            Utils.flatten(matrix));
    }
    
    @Test
    void testFlattenFloatMatrixJaggedArray() {
        float[][] matrix = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f}}; // Different row lengths
        assertThrows(IllegalArgumentException.class, () -> 
            Utils.flatten(matrix));
    }
    
    // ========== ONE-HOT ENCODING TESTS ==========
    
    @Test
    void testOneHotBasic() {
        float[] result = Utils.oneHot(2, 5);
        
        assertArrayEquals(new float[]{0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, result, DELTA);
    }
    
    @Test
    void testOneHotFirstClass() {
        float[] result = Utils.oneHot(0, 3);
        
        assertArrayEquals(new float[]{1.0f, 0.0f, 0.0f}, result, DELTA);
    }
    
    @Test
    void testOneHotLastClass() {
        float[] result = Utils.oneHot(4, 5);
        
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, result, DELTA);
    }
    
    @Test
    void testOneHotMNISTClasses() {
        // Test all MNIST digit classes
        for (int digit = 0; digit < 10; digit++) {
            float[] result = Utils.oneHot(digit, 10);
            
            assertEquals(10, result.length);
            for (int i = 0; i < 10; i++) {
                if (i == digit) {
                    assertEquals(1.0f, result[i], DELTA);
                } else {
                    assertEquals(0.0f, result[i], DELTA);
                }
            }
        }
    }
    
    @Test
    void testOneHotSingleClass() {
        float[] result = Utils.oneHot(0, 1);
        
        assertArrayEquals(new float[]{1.0f}, result, DELTA);
    }
    
    @Test
    void testOneHotInvalidLabel() {
        assertThrows(IllegalArgumentException.class, () -> 
            Utils.oneHot(-1, 5)); // Negative label
        
        assertThrows(IllegalArgumentException.class, () -> 
            Utils.oneHot(5, 5)); // Label >= numClasses
        
        assertThrows(IllegalArgumentException.class, () -> 
            Utils.oneHot(10, 5)); // Label way out of bounds
    }
    
    @Test
    void testOneHotInvalidNumClasses() {
        assertThrows(IllegalArgumentException.class, () -> 
            Utils.oneHot(0, 0)); // Zero classes
        
        assertThrows(IllegalArgumentException.class, () -> 
            Utils.oneHot(0, -1)); // Negative classes
    }
    
    // ========== ARGMAX TESTS ==========
    
    @Test
    void testArgmaxBasic() {
        float[] array = {0.1f, 0.8f, 0.1f};
        assertEquals(1, Utils.argmax(array));
    }
    
    @Test
    void testArgmaxScalar() {
        float[] array = {0.2f, 0.5f, 0.1f, 0.2f};
        assertEquals(1, Utils.argmaxScalar(array));
    }
    
    @Test
    void testArgmaxVectorized() {
        float[] array = {0.1f, 0.05f, 0.8f, 0.02f, 0.01f, 0.01f, 0.01f, 0.0f};
        assertEquals(2, Utils.argmaxVectorized(array));
    }
    
    @Test
    void testArgmaxScalarVsVectorizedConsistency() {
        float[] array = {0.15f, 0.25f, 0.1f, 0.3f, 0.05f, 0.1f, 0.02f, 0.03f};
        
        int scalarResult = Utils.argmaxScalar(array);
        int vectorizedResult = Utils.argmaxVectorized(array);
        
        assertEquals(scalarResult, vectorizedResult);
        assertEquals(3, scalarResult); // 0.3f is at index 3
    }
    
    @Test
    void testArgmaxFirstElement() {
        float[] array = {0.9f, 0.05f, 0.03f, 0.02f};
        assertEquals(0, Utils.argmax(array));
    }
    
    @Test
    void testArgmaxLastElement() {
        float[] array = {0.1f, 0.2f, 0.15f, 0.55f};
        assertEquals(3, Utils.argmax(array));
    }
    
    @Test
    void testArgmaxSingleElement() {
        float[] array = {42.0f};
        assertEquals(0, Utils.argmax(array));
    }
    
    @Test
    void testArgmaxEqualValues() {
        // When values are equal, should return first occurrence
        float[] array = {0.5f, 0.5f, 0.3f};
        assertEquals(0, Utils.argmax(array));
    }
    
    @Test
    void testArgmaxMNISTScenario() {
        // Typical MNIST softmax output (10 classes)
        float[] predictions = new float[10];
        predictions[0] = 0.05f; predictions[1] = 0.02f; predictions[2] = 0.03f;
        predictions[3] = 0.01f; predictions[4] = 0.04f; predictions[5] = 0.02f;
        predictions[6] = 0.08f; predictions[7] = 0.7f;  predictions[8] = 0.03f; predictions[9] = 0.02f;
        
        assertEquals(7, Utils.argmax(predictions)); // Model predicts digit 7
    }
    
    @Test
    void testArgmaxEmptyArray() {
        float[] empty = {};
        assertThrows(IllegalArgumentException.class, () -> Utils.argmax(empty));
    }
    
    @Test
    void testArgmaxNullArray() {
        assertThrows(IllegalArgumentException.class, () -> Utils.argmax(null));
    }
    
    @Test
    void testArgmaxPerformanceComparison() {
        // Test that both implementations give same result for various sizes
        for (int size : new int[]{10, 50, 100, 500}) {
            float[] array = new float[size];
            int maxIndex = size / 2; // Put max in middle
            
            for (int i = 0; i < size; i++) {
                array[i] = (float) (Math.random() * 0.8); // Values 0-0.8
            }
            array[maxIndex] = 1.0f; // Clear maximum
            
            int scalarResult = Utils.argmaxScalar(array);
            int vectorizedResult = Utils.argmaxVectorized(array);
            int generalResult = Utils.argmax(array);
            
            assertEquals(maxIndex, scalarResult, "Scalar failed for size " + size);
            assertEquals(maxIndex, vectorizedResult, "Vectorized failed for size " + size);
            assertEquals(maxIndex, generalResult, "General failed for size " + size);
        }
    }
    
    // ========== INTEGRATION TESTS ==========
    
    @Test
    void testMNISTWorkflow() {
        // Simulate MNIST data processing workflow
        int[][] mockImage = new int[28][28];
        
        // Fill with gradient pattern (like edge detection)
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                mockImage[i][j] = Math.min(255, i * 9); // Gradient
            }
        }
        
        int mockLabel = 7;
        
        // Process like in Main.java
        float[] input = Utils.flatten(mockImage, true);
        float[] labels = Utils.oneHot(mockLabel, 10);
        
        // Verify results
        assertEquals(784, input.length);
        assertEquals(10, labels.length);
        assertEquals(1.0f, labels[7], DELTA);
        
        // Check normalization worked
        assertTrue(input[0] >= 0.0f && input[0] <= 1.0f);
        assertTrue(input[783] >= 0.0f && input[783] <= 1.0f);
    }
}