package dev.neuronic.net.math.ops;

/**
 * Cumulative sampling from a probability distribution.
 * 
 * Given a random value in [0, 1), samples an index from the distribution
 * using the cumulative probability method.
 */
public final class CumulativeSampling {
    
    /**
     * Sample an index from a probability distribution.
     * 
     * @param probabilities probability distribution (must sum to 1)
     * @param randomValue random value in [0, 1)
     * @return sampled index
     */
    public static int sample(float[] probabilities, float randomValue) {
        if (randomValue < 0 || randomValue >= 1) {
            throw new IllegalArgumentException("Random value must be in [0, 1), got: " + randomValue);
        }
        
        float cumSum = 0.0f;
        
        for (int i = 0; i < probabilities.length; i++) {
            cumSum += probabilities[i];
            
            if (randomValue < cumSum) {
                return i;
            }
        }
        
        // Fallback to last index (in case of floating point precision issues)
        return probabilities.length - 1;
    }
    
    private CumulativeSampling() {} // Prevent instantiation
}