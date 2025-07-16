package dev.neuronic.net.math.ops;

import java.util.Arrays;

/**
 * Top-P (nucleus) sampling for probability distributions.
 * 
 * Keeps the smallest set of indices whose cumulative probability
 * exceeds the threshold p, zeros out the rest, and renormalizes.
 */
public final class TopPSampling {
    
    /**
     * Apply Top-P (nucleus) sampling to a probability distribution.
     * 
     * @param probabilities input probability distribution
     * @param p cumulative probability threshold (0.0 to 1.0)
     * @param output output distribution with nucleus sampling applied
     */
    public static void apply(float[] probabilities, float p, float[] output) {
        if (p <= 0 || p > 1) {
            throw new IllegalArgumentException("P must be between 0 and 1, got: " + p);
        }
        
        // Sort indices by probability (descending)
        Integer[] indices = new Integer[probabilities.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (a, b) -> Float.compare(probabilities[b], probabilities[a]));
        
        // Find nucleus
        Arrays.fill(output, 0.0f);
        float cumSum = 0.0f;
        float totalSum = 0.0f;
        
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            cumSum += probabilities[idx];
            output[idx] = probabilities[idx];
            totalSum += probabilities[idx];
            
            if (cumSum >= p) {
                break;
            }
        }
        
        // Renormalize
        if (totalSum > 0) {
            for (int i = 0; i < output.length; i++) {
                output[i] /= totalSum;
            }
        }
    }
    
    /**
     * Apply Top-P (nucleus) sampling with excluded indices.
     * 
     * @param probabilities input probability distribution
     * @param p cumulative probability threshold (0.0 to 1.0)
     * @param excludeIndices indices to exclude from selection
     * @param output output distribution with nucleus sampling applied
     */
    public static void applyWithExclusions(float[] probabilities, float p,
                                          int[] excludeIndices, float[] output) {
        if (p <= 0 || p > 1) {
            throw new IllegalArgumentException("P must be between 0 and 1, got: " + p);
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
        
        // Check if all indices are excluded
        int validCount = 0;
        for (int i = 0; i < excluded.length; i++) {
            if (!excluded[i]) validCount++;
        }
        
        if (validCount == 0) {
            throw new IllegalArgumentException("All indices are excluded");
        }
        
        // Sort indices by probability (descending), with excluded at end
        Integer[] indices = new Integer[probabilities.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        
        Arrays.sort(indices, (a, b) -> {
            // Put excluded indices at the end
            if (excluded[a] && !excluded[b]) return 1;
            if (!excluded[a] && excluded[b]) return -1;
            // Otherwise sort by probability
            return Float.compare(probabilities[b], probabilities[a]);
        });
        
        // Find nucleus among non-excluded indices
        Arrays.fill(output, 0.0f);
        float cumSum = 0.0f;
        float totalSum = 0.0f;
        
        // First, calculate total probability of non-excluded indices
        float nonExcludedSum = 0.0f;
        for (int i = 0; i < probabilities.length; i++) {
            if (!excluded[i]) {
                nonExcludedSum += probabilities[i];
            }
        }
        
        // Build nucleus based on proportion of non-excluded probability
        float adjustedThreshold = p * nonExcludedSum;
        
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            if (!excluded[idx]) {
                cumSum += probabilities[idx];
                output[idx] = probabilities[idx];
                totalSum += probabilities[idx];
                
                if (cumSum >= adjustedThreshold) {
                    break;
                }
            }
        }
        
        // Renormalize
        if (totalSum > 0) {
            for (int i = 0; i < output.length; i++) {
                output[i] /= totalSum;
            }
        }
    }
    
    private TopPSampling() {} // Prevent instantiation
}