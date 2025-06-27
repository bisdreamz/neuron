package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class WeightInitXavierTest {
    
    private static final float DELTA = 1e-6f;
    
    @Test
    void testComputeBasic() {
        float[][] weights = new float[3][4];
        int fanIn = 3;
        int fanOut = 4;
        
        WeightInitXavier.compute(weights, fanIn, fanOut);
        
        // Verify all weights are initialized (not zero)
        boolean hasNonZero = false;
        for (float[] row : weights) {
            for (float value : row) {
                if (value != 0.0f) {
                    hasNonZero = true;
                    break;
                }
            }
        }
        assertTrue(hasNonZero, "Weights should be initialized to non-zero values");
    }
    
    @Test
    void testSmallArray() {
        // Test with array too small for vectorization
        float[][] weights = new float[1][4];
        int fanIn = 10;
        int fanOut = 20;
        
        WeightInitXavier.compute(weights, fanIn, fanOut);
        
        // Verify all elements are initialized
        boolean hasNonZero = false;
        for (float value : weights[0]) {
            if (value != 0.0f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue(hasNonZero, "Small array should be initialized to non-zero values");
    }
    
    @Test
    void testLargeArrayVectorized() {
        // Test with array large enough for vectorization
        float[][] weights = new float[1][64];
        int fanIn = 10;
        int fanOut = 20;
        
        WeightInitXavier.compute(weights, fanIn, fanOut);
        
        // Verify all elements are initialized
        boolean hasNonZero = false;
        for (float value : weights[0]) {
            if (value != 0.0f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue(hasNonZero, "Large array should be initialized to non-zero values");
    }
    
    @Test
    void testScalarVsVectorizedConsistency() {
        // Test that scalar path (small arrays) and vectorized path (large arrays) 
        // produce similar statistical distributions
        int fanIn = 50;
        int fanOut = 100;
        float expectedLimit = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        
        // Small array - will use scalar path
        float[][] smallWeights = new float[100][4];
        WeightInitXavier.compute(smallWeights, fanIn, fanOut);
        
        // Large array - will use vectorized path
        float[][] largeWeights = new float[100][64];
        WeightInitXavier.compute(largeWeights, fanIn, fanOut);
        
        // Calculate statistics for both
        double smallMean = calculateMean2D(smallWeights);
        double largeMean = calculateMean2D(largeWeights);
        double smallStd = calculateStd2D(smallWeights, smallMean);
        double largeStd = calculateStd2D(largeWeights, largeMean);
        
        // Both should have mean close to 0 (uniform distribution centered at 0)
        assertTrue(Math.abs(smallMean) < 0.1, "Small array mean should be close to 0");
        assertTrue(Math.abs(largeMean) < 0.1, "Large array mean should be close to 0");
        
        // For uniform distribution in [-limit, +limit], std = limit / sqrt(3)
        float expectedStd = expectedLimit / (float)Math.sqrt(3.0);
        assertEquals(expectedStd, smallStd, expectedStd * 0.3, 
                    "Small array std should match uniform distribution");
        assertEquals(expectedStd, largeStd, expectedStd * 0.3, 
                    "Large array std should match uniform distribution");
    }
    
    @Test
    void testXavierScaling() {
        int fanIn = 50;
        int fanOut = 100;
        float expectedLimit = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        float[][] weights = new float[10][20];
        
        WeightInitXavier.compute(weights, fanIn, fanOut);
        
        // Check bounds - all values should be within [-limit, +limit]
        for (float[] row : weights) {
            for (float value : row) {
                assertTrue(Math.abs(value) <= expectedLimit + DELTA, 
                          "Xavier values should be within bounds: " + value + " vs " + expectedLimit);
            }
        }
        
        // Calculate statistics
        double mean = calculateMean2D(weights);
        double std = calculateStd2D(weights, mean);
        
        // Mean should be close to 0
        assertTrue(Math.abs(mean) < 0.05, "Mean should be close to 0");
        
        // For uniform distribution in [-limit, +limit], std = limit / sqrt(3)
        float expectedStd = expectedLimit / (float)Math.sqrt(3.0);
        assertEquals(expectedStd, std, expectedStd * 0.2, 
                    "Sample std should match uniform distribution");
    }
    
    @Test
    void testSingleElement() {
        float[][] weights = new float[1][1];
        int fanIn = 10;
        int fanOut = 20;
        
        WeightInitXavier.compute(weights, fanIn, fanOut);
        assertNotEquals(0.0f, weights[0][0], "Single element should be initialized");
    }
    
    @Test
    void testLargeArrayWithRemainder() {
        // Test array size that's not a multiple of vector length
        float[][] weights = new float[1][1001];  // Odd size to ensure remainder
        int fanIn = 50;
        int fanOut = 100;
        
        WeightInitXavier.compute(weights, fanIn, fanOut);
        
        // Check that all elements are initialized, including remainder
        boolean allInitialized = true;
        for (float value : weights[0]) {
            if (value == 0.0f) {
                allInitialized = false;
                break;
            }
        }
        assertTrue(allInitialized, "All elements should be initialized even with remainder");
    }
    
    @Test
    void testInvalidFanInOut() {
        float[][] weights = new float[2][2];
        
        // Test zero fanIn
        assertThrows(IllegalArgumentException.class, 
                    () -> WeightInitXavier.compute(weights, 0, 10),
                    "Should throw exception for zero fanIn");
        
        // Test zero fanOut
        assertThrows(IllegalArgumentException.class, 
                    () -> WeightInitXavier.compute(weights, 10, 0),
                    "Should throw exception for zero fanOut");
        
        // Test negative fanIn
        assertThrows(IllegalArgumentException.class, 
                    () -> WeightInitXavier.compute(weights, -10, 10),
                    "Should throw exception for negative fanIn");
        
        // Test negative fanOut
        assertThrows(IllegalArgumentException.class, 
                    () -> WeightInitXavier.compute(weights, 10, -10),
                    "Should throw exception for negative fanOut");
    }
    
    @Test
    void testDifferentFanInOut() {
        float[][] weights1 = new float[10][10];
        float[][] weights2 = new float[10][10];
        
        WeightInitXavier.compute(weights1, 10, 10);    // Small fan-in/out
        WeightInitXavier.compute(weights2, 100, 100);  // Large fan-in/out
        
        // Larger fan-in/out should produce smaller weights on average
        double avg1 = calculateAbsoluteMean(weights1);
        double avg2 = calculateAbsoluteMean(weights2);
        
        assertTrue(avg1 > avg2, "Smaller fan-in/out should produce larger weights on average");
    }
    
    @Test
    void testAsymmetricFanInOut() {
        float[][] weights1 = new float[10][10];
        float[][] weights2 = new float[10][10];
        
        WeightInitXavier.compute(weights1, 10, 100);  // fanIn << fanOut
        WeightInitXavier.compute(weights2, 100, 10);  // fanIn >> fanOut
        
        // Both should have similar scale since Xavier uses (fanIn + fanOut)
        double avg1 = calculateAbsoluteMean(weights1);
        double avg2 = calculateAbsoluteMean(weights2);
        
        assertTrue(Math.abs(avg1 - avg2) < Math.max(avg1, avg2) * 0.3, 
                  "Asymmetric fan-in/out should produce similar scales");
    }
    
    @Test
    void testMixedRowSizes() {
        // Test with different row sizes to ensure both paths work correctly
        float[][] weights = new float[4][];
        weights[0] = new float[4];   // Small - scalar path
        weights[1] = new float[64];  // Large - vectorized path
        weights[2] = new float[8];   // Small - scalar path
        weights[3] = new float[128]; // Large - vectorized path
        
        int fanIn = 30;
        int fanOut = 60;
        float expectedLimit = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        
        WeightInitXavier.compute(weights, fanIn, fanOut);
        
        // Verify all rows are initialized correctly
        for (float[] row : weights) {
            boolean hasNonZero = false;
            for (float value : row) {
                if (value != 0.0f) {
                    hasNonZero = true;
                    // Also check bounds
                    assertTrue(Math.abs(value) <= expectedLimit + DELTA,
                              "Value should be within Xavier bounds");
                }
            }
            assertTrue(hasNonZero, "Row should have non-zero values");
        }
    }
    
    private double calculateMean2D(float[][] matrix) {
        double sum = 0.0;
        int count = 0;
        for (float[] row : matrix) {
            for (float value : row) {
                sum += value;
                count++;
            }
        }
        return sum / count;
    }
    
    private double calculateStd2D(float[][] matrix, double mean) {
        double variance = 0.0;
        int count = 0;
        for (float[] row : matrix) {
            for (float value : row) {
                double diff = value - mean;
                variance += diff * diff;
                count++;
            }
        }
        return Math.sqrt(variance / (count - 1));
    }
    
    private double calculateAbsoluteMean(float[][] matrix) {
        double sum = 0.0;
        int count = 0;
        for (float[] row : matrix) {
            for (float value : row) {
                sum += Math.abs(value);
                count++;
            }
        }
        return sum / count;
    }
}