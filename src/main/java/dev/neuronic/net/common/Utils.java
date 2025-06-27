package dev.neuronic.net.common;

import dev.neuronic.net.math.Vectorization;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class Utils {

    // ========== DATA PREPROCESSING UTILITIES ==========
    
    /**
     * Flatten a 2D byte matrix to a 1D float array.
     * 
     * @param matrix the 2D matrix to flatten
     * @param normalize if true, normalizes values from [0,255] to [0.0,1.0]
     * @return flattened 1D array
     */
    public static float[] flatten(byte[][] matrix, boolean normalize) {
        if (matrix == null || matrix.length == 0)
            throw new IllegalArgumentException("Matrix cannot be null or empty");
        
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[] result = new float[rows * cols];
        
        int index = 0;
        for (int row = 0; row < rows; row++) {
            if (matrix[row].length != cols)
                throw new IllegalArgumentException("Matrix must be rectangular");
            
            for (int col = 0; col < cols; col++) {
                byte b = matrix[row][col];
                int value = (b < 0) ? (b & 0xFF) : b; // Convert to unsigned only if negative
                if (normalize)
                    result[index++] = value / 255.0f;
                else
                    result[index++] = value;
            }
        }
        
        return result;
    }
    
    /**
     * Flatten a 2D int matrix to a 1D float array.
     * 
     * @param matrix the 2D matrix to flatten
     * @param normalize if true, normalizes values from [0,255] to [0.0,1.0]
     * @return flattened 1D array
     */
    public static float[] flatten(int[][] matrix, boolean normalize) {
        if (matrix == null || matrix.length == 0)
            throw new IllegalArgumentException("Matrix cannot be null or empty");
        
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[] result = new float[rows * cols];
        
        int index = 0;
        for (int row = 0; row < rows; row++) {
            if (matrix[row].length != cols)
                throw new IllegalArgumentException("Matrix must be rectangular");
            
            for (int col = 0; col < cols; col++) {
                if (normalize)
                    result[index++] = matrix[row][col] / 255.0f;
                else
                    result[index++] = matrix[row][col];
            }
        }
        
        return result;
    }
    
    /**
     * Flatten a 2D float matrix to a 1D float array.
     * 
     * @param matrix the 2D matrix to flatten
     * @return flattened 1D array
     */
    public static float[] flatten(float[][] matrix) {
        if (matrix == null || matrix.length == 0)
            throw new IllegalArgumentException("Matrix cannot be null or empty");
        
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[] result = new float[rows * cols];
        
        int index = 0;
        for (int row = 0; row < rows; row++) {
            if (matrix[row].length != cols)
                throw new IllegalArgumentException("Matrix must be rectangular");
            
            for (int col = 0; col < cols; col++) {
                result[index++] = matrix[row][col];
            }
        }
        
        return result;
    }
    
    /**
     * Create a one-hot encoded vector for classification.
     * 
     * @param label the class label (0-based index)
     * @param numClasses total number of classes
     * @return one-hot encoded array where only the label index is 1.0f
     */
    public static float[] oneHot(int label, int numClasses) {
        if (label < 0 || label >= numClasses)
            throw new IllegalArgumentException("Label must be between 0 and " + (numClasses - 1) + ", got: " + label);
        if (numClasses <= 0)
            throw new IllegalArgumentException("Number of classes must be positive, got: " + numClasses);
        
        float[] result = new float[numClasses];
        result[label] = 1.0f;
        return result;
    }
    
    // ========== ARRAY SEARCH UTILITIES ==========
    
    /**
     * Find the index of the maximum value in an array (argmax).
     * Essential for classification - converts probabilities to predicted class.
     * 
     * <p>Example: [0.1, 0.8, 0.1] → returns 1 (index of 0.8)
     * 
     * @param array the array to search
     * @return index of the maximum value
     * @throws IllegalArgumentException if array is null or empty
     */
    public static int argmax(float[] array) {
        if (array == null || array.length == 0)
            throw new IllegalArgumentException("Array cannot be null or empty");
        
        if (Vectorization.shouldVectorize(array.length))
            return argmaxVectorized(array);
        else
            return argmaxScalar(array);
    }
    
    /**
     * Scalar implementation of argmax for small arrays or fallback.
     */
    static int argmaxScalar(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];
        
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }
    
    /**
     * Vectorized implementation of argmax for large arrays.
     * Uses SIMD to find maximum values in chunks, then finds global maximum.
     */
    static int argmaxVectorized(float[] array) {
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int length = array.length;
        int upperBound = Vectorization.loopBound(length);
        
        if (upperBound == 0)
            return argmaxScalar(array);
        
        // Track maximum value and its index for each vector chunk
        float globalMax = Float.NEGATIVE_INFINITY;
        int globalMaxIndex = 0;
        
        // Process vector chunks
        for (int i = 0; i < upperBound; i += species.length()) {
            FloatVector v = FloatVector.fromArray(species, array, i);
            float chunkMax = v.reduceLanes(VectorOperators.MAX);
            
            if (chunkMax > globalMax) {
                globalMax = chunkMax;
                // Find the exact index within this chunk
                for (int j = 0; j < species.length() && (i + j) < length; j++) {
                    if (array[i + j] == chunkMax) {
                        globalMaxIndex = i + j;
                        break;
                    }
                }
            }
        }
        
        // Check remainder elements
        for (int i = upperBound; i < length; i++) {
            if (array[i] > globalMax) {
                globalMax = array[i];
                globalMaxIndex = i;
            }
        }
        
        return globalMaxIndex;
    }
    
    /**
     * Sort array indices by their values in descending order (highest values first).
     * Useful for getting top-k predictions in classification.
     * 
     * <p>Example: [0.1, 0.8, 0.1] → returns [1, 0, 2] (indices sorted by value)
     * 
     * @param array the array to sort indices for
     * @param descending if true, sort highest to lowest; if false, lowest to highest
     * @return array of indices sorted by their corresponding values
     */
    public static int[] argsort(float[] array, boolean descending) {
        if (array == null || array.length == 0)
            throw new IllegalArgumentException("Array cannot be null or empty");
        
        // Create array of indices
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }
        
        // Sort indices by their corresponding values
        java.util.Arrays.sort(indices, (i, j) -> {
            float diff = array[i] - array[j];
            if (descending) {
                return Float.compare(array[j], array[i]); // Reverse for descending
            } else {
                return Float.compare(array[i], array[j]); // Normal for ascending
            }
        });
        
        // Convert to primitive int array
        int[] result = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = indices[i];
        }
        
        return result;
    }
    
    /**
     * Get the indices of the top K values in an array.
     * Returns indices sorted by their corresponding values (highest first).
     * 
     * <p>Example: topKIndices([0.1, 0.8, 0.1, 0.5], 2) → [1, 3] (indices of 0.8 and 0.5)
     * 
     * @param array the array to search
     * @param k number of top indices to return
     * @return array of K indices with highest values
     * @throws IllegalArgumentException if k is invalid
     */
    public static int[] topKIndices(float[] array, int k) {
        if (array == null || array.length == 0)
            throw new IllegalArgumentException("Array cannot be null or empty");
        if (k <= 0 || k > array.length)
            throw new IllegalArgumentException("k must be between 1 and array length");
        
        // Get all indices sorted by value (descending)
        int[] sorted = argsort(array, true);
        
        // Return only the top k
        int[] topK = new int[k];
        System.arraycopy(sorted, 0, topK, 0, k);
        return topK;
    }

}
