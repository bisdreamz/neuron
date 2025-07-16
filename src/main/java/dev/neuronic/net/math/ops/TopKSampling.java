package dev.neuronic.net.math.ops;

import java.util.Arrays;

/**
 * Top-K sampling for probability distributions.
 * 
 * Keeps only the K highest probability values, zeros out the rest,
 * and renormalizes the distribution.
 */
public final class TopKSampling {
    
    /**
     * Apply Top-K sampling to a probability distribution.
     * 
     * @param probabilities input probability distribution
     * @param k number of top indices to keep
     * @param output output distribution with only top K values
     */
    public static void apply(float[] probabilities, int k, float[] output) {
        if (k <= 0 || k > probabilities.length) {
            throw new IllegalArgumentException("K must be between 1 and vocab size, got: " + k);
        }
        
        // Find top K indices
        Integer[] indices = new Integer[probabilities.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        
        // Partial sort to get top K
        Arrays.sort(indices, (a, b) -> Float.compare(probabilities[b], probabilities[a]));
        
        // Create distribution with only top K
        Arrays.fill(output, 0.0f);
        float sum = 0.0f;
        
        for (int i = 0; i < k && i < indices.length; i++) {
            int idx = indices[i];
            output[idx] = probabilities[idx];
            sum += probabilities[idx];
        }
        
        // Renormalize
        if (sum > 0) {
            for (int i = 0; i < output.length; i++) {
                output[i] /= sum;
            }
        }
    }
    
    /**
     * Apply Top-K sampling with excluded indices.
     * 
     * @param probabilities input probability distribution
     * @param k number of top indices to keep
     * @param excludeIndices indices to exclude from selection
     * @param output output distribution with only top K non-excluded values
     */
    public static void applyWithExclusions(float[] probabilities, int k, 
                                          int[] excludeIndices, float[] output) {
        if (k <= 0) {
            throw new IllegalArgumentException("K must be positive, got: " + k);
        }
        
        // Mark excluded indices
        boolean[] excluded = new boolean[probabilities.length];
        if (excludeIndices != null) {
            for (int idx : excludeIndices) {
                if (idx >= 0 && idx < excluded.length) {
                    excluded[idx] = true;
                }
            }
        }
        
        // Count valid indices
        int validCount = 0;
        for (int i = 0; i < excluded.length; i++) {
            if (!excluded[i]) validCount++;
        }
        
        if (validCount == 0) {
            throw new IllegalArgumentException("All indices are excluded");
        }
        
        // Adjust k if necessary
        k = Math.min(k, validCount);
        
        // Find top K non-excluded indices
        Integer[] indices = new Integer[probabilities.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        
        // Sort with exclusion handling
        Arrays.sort(indices, (a, b) -> {
            // Put excluded indices at the end
            if (excluded[a] && !excluded[b]) return 1;
            if (!excluded[a] && excluded[b]) return -1;
            // Otherwise sort by probability
            return Float.compare(probabilities[b], probabilities[a]);
        });
        
        // Create distribution with only top K non-excluded values
        Arrays.fill(output, 0.0f);
        float sum = 0.0f;
        
        int added = 0;
        for (int i = 0; i < indices.length && added < k; i++) {
            int idx = indices[i];
            if (!excluded[idx]) {
                output[idx] = probabilities[idx];
                sum += probabilities[idx];
                added++;
            }
        }
        
        // Renormalize
        if (sum > 0) {
            for (int i = 0; i < output.length; i++) {
                output[i] /= sum;
            }
        }
    }
    
    private TopKSampling() {} // Prevent instantiation
}