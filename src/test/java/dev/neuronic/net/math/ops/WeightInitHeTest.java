package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.FastRandom;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class WeightInitHeTest {
    
    private static final float DELTA = 1e-6f;
    
    @Test
    void testComputeBasic() {
        float[][] weights = new float[3][4];
        int fanIn = 3;
        FastRandom random = new FastRandom(12345);
        
        WeightInitHe.compute(weights, fanIn, random);
        
        // Verify all weights are initialized (not zero)
        boolean hasNonZero = false;
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                if (weights[i][j] != 0.0f) {
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
        
        FastRandom random = new FastRandom(12345);
        WeightInitHe.compute(weights, fanIn, random);
        
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
        
        FastRandom random = new FastRandom(12345);
        WeightInitHe.compute(weights, fanIn, random);
        
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
        float expectedScale = (float) Math.sqrt(2.0 / fanIn);
        
        // Small array - will use scalar path
        float[][] smallWeights = new float[100][4];
        FastRandom random1 = new FastRandom(12345);
        WeightInitHe.compute(smallWeights, fanIn, random1);
        
        // Large array - will use vectorized path
        float[][] largeWeights = new float[100][64];
        FastRandom random2 = new FastRandom(12345);
        WeightInitHe.compute(largeWeights, fanIn, random2);
        
        // Calculate statistics for both
        double smallMean = calculateMean2D(smallWeights);
        double largeMean = calculateMean2D(largeWeights);
        double smallStd = calculateStd2D(smallWeights, smallMean);
        double largeStd = calculateStd2D(largeWeights, largeMean);
        
        // Both should have mean close to 0 (Gaussian distribution)
        assertTrue(Math.abs(smallMean) < 0.1, "Small array mean should be close to 0");
        assertTrue(Math.abs(largeMean) < 0.1, "Large array mean should be close to 0");
        
        // Both should have similar standard deviation close to expectedScale
        assertEquals(expectedScale, smallStd, expectedScale * 0.3, 
                    "Small array std should be close to He scale");
        assertEquals(expectedScale, largeStd, expectedScale * 0.3, 
                    "Large array std should be close to He scale");
    }
    
    @Test
    void testHeScaling() {
        int fanIn = 100;
        float expectedScale = (float) Math.sqrt(2.0 / fanIn);
        float[][] weights = new float[10][20];
        
        FastRandom random = new FastRandom(12345);
        WeightInitHe.compute(weights, fanIn, random);
        
        // Calculate sample standard deviation
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                sum += weights[i][j];
                count++;
            }
        }
        double mean = sum / count;
        
        double variance = 0.0;
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                double diff = weights[i][j] - mean;
                variance += diff * diff;
            }
        }
        variance /= (count - 1);
        double sampleStd = Math.sqrt(variance);
        
        // He initialization should produce std close to expectedScale
        assertTrue(Math.abs(sampleStd - expectedScale) < expectedScale * 0.5, 
                  "Sample std should be close to He scale factor");
    }
    
    @Test
    void testSingleElement() {
        float[][] weights = new float[1][1];
        int fanIn = 10;
        
        FastRandom random = new FastRandom(12345);
        WeightInitHe.compute(weights, fanIn, random);
        assertNotEquals(0.0f, weights[0][0], "Single element should be initialized");
    }
    
    @Test
    void testLargeArrayWithRemainder() {
        // Test array size that's not a multiple of vector length
        float[][] weights = new float[1][1001];  // Odd size to ensure remainder
        int fanIn = 50;
        
        FastRandom random = new FastRandom(12345);
        WeightInitHe.compute(weights, fanIn, random);
        
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
    void testInvalidFanIn() {
        float[][] weights = new float[2][2];
        
        // Test zero fanIn
        assertThrows(IllegalArgumentException.class, 
                    () -> WeightInitHe.compute(weights, 0, new FastRandom(12345)),
                    "Should throw exception for zero fanIn");
        
        // Test negative fanIn
        assertThrows(IllegalArgumentException.class, 
                    () -> WeightInitHe.compute(weights, -10, new FastRandom(12345)),
                    "Should throw exception for negative fanIn");
    }
    
    @Test
    void testDifferentFanIn() {
        float[][] weights1 = new float[10][10];
        float[][] weights2 = new float[10][10];
        
        FastRandom random1 = new FastRandom(12345);
        WeightInitHe.compute(weights1, 10, random1);   // Small fan-in
        FastRandom random2 = new FastRandom(12345);
        WeightInitHe.compute(weights2, 1000, random2); // Large fan-in
        
        // Larger fan-in should produce smaller weights on average
        double avg1 = calculateAbsoluteMean(weights1);
        double avg2 = calculateAbsoluteMean(weights2);
        
        assertTrue(avg1 > avg2, "Smaller fan-in should produce larger weights on average");
    }
    
    @Test
    void testMixedRowSizes() {
        // Test with different row sizes to ensure both paths work correctly
        float[][] weights = new float[4][];
        weights[0] = new float[4];   // Small - scalar path
        weights[1] = new float[64];  // Large - vectorized path
        weights[2] = new float[8];   // Small - scalar path
        weights[3] = new float[128]; // Large - vectorized path
        
        int fanIn = 50;
        float expectedScale = (float) Math.sqrt(2.0 / fanIn);
        
        FastRandom random = new FastRandom(12345);
        WeightInitHe.compute(weights, fanIn, random);
        
        // Verify all rows are initialized correctly
        for (int i = 0; i < weights.length; i++) {
            float[] row = weights[i];
            boolean hasNonZero = false;
            double rowSum = 0.0;
            for (float value : row) {
                if (value != 0.0f) {
                    hasNonZero = true;
                }
                rowSum += value * value;
            }
            assertTrue(hasNonZero, "Row should have non-zero values");
            
            // Check that RMS is reasonable (should be close to expectedScale)
            // Use more lenient bounds for small arrays due to higher variance
            double rms = Math.sqrt(rowSum / row.length);
            double lowerBound = row.length <= 8 ? expectedScale * 0.2 : expectedScale * 0.3;
            double upperBound = row.length <= 8 ? expectedScale * 4.0 : expectedScale * 3.0;
            assertTrue(rms > lowerBound && rms < upperBound,
                      String.format("RMS should be reasonable for He initialization. " +
                                    "Row length: %d, RMS: %.4f, Expected scale: %.4f, " +
                                    "Acceptable range: [%.4f, %.4f]", 
                                    row.length, rms, expectedScale, lowerBound, upperBound));
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
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                sum += Math.abs(matrix[i][j]);
                count++;
            }
        }
        return sum / count;
    }
}