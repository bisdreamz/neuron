package dev.neuronic.net;

/**
 * Configuration for sampling strategies in text generation and classification.
 * 
 * <p>Controls how models select outputs from probability distributions.
 * Different strategies provide different trade-offs between quality, 
 * diversity, and coherence.
 * 
 * <p><b>Usage Examples:</b>
 * <pre>{@code
 * // For deterministic outputs (default)
 * SamplingConfig.argmax()
 * 
 * // For creative text generation
 * SamplingConfig.temperature(1.2f)
 * 
 * // For balanced generation with vocabulary restriction
 * SamplingConfig.topK(40, 0.8f)
 * 
 * // For high-quality generation with dynamic vocabulary
 * SamplingConfig.topP(0.9f, 0.8f)
 * }</pre>
 */
public class SamplingConfig {
    
    public enum Strategy {
        ARGMAX,      // Always pick highest probability
        TEMPERATURE, // Scale distribution before sampling
        TOP_K,       // Sample from top K tokens
        TOP_P        // Sample from nucleus (cumulative prob)
    }
    
    private final Strategy strategy;
    private final float temperature;
    private final int k;
    private final float p;
    
    private SamplingConfig(Strategy strategy, float temperature, int k, float p) {
        this.strategy = strategy;
        this.temperature = temperature;
        this.k = k;
        this.p = p;
    }
    
    /**
     * Deterministic argmax sampling (default).
     * Always picks the highest probability token.
     */
    public static SamplingConfig argmax() {
        return new SamplingConfig(Strategy.ARGMAX, 1.0f, 0, 0);
    }
    
    /**
     * Temperature-based sampling.
     * 
     * @param temperature controls randomness (0.1=focused, 1.0=normal, 2.0=creative)
     */
    public static SamplingConfig temperature(float temperature) {
        if (temperature <= 0) {
            throw new IllegalArgumentException("Temperature must be positive");
        }
        return new SamplingConfig(Strategy.TEMPERATURE, temperature, 0, 0);
    }
    
    /**
     * Top-K sampling with optional temperature.
     * 
     * @param k number of top tokens to consider
     * @param temperature optional temperature scaling (1.0 = no scaling)
     */
    public static SamplingConfig topK(int k, float temperature) {
        if (k <= 0) {
            throw new IllegalArgumentException("K must be positive");
        }
        if (temperature <= 0) {
            throw new IllegalArgumentException("Temperature must be positive");
        }
        return new SamplingConfig(Strategy.TOP_K, temperature, k, 0);
    }
    
    /**
     * Top-K sampling without temperature scaling.
     */
    public static SamplingConfig topK(int k) {
        return topK(k, 1.0f);
    }
    
    /**
     * Top-P (nucleus) sampling with optional temperature.
     * 
     * @param p cumulative probability threshold (e.g., 0.9)
     * @param temperature optional temperature scaling
     */
    public static SamplingConfig topP(float p, float temperature) {
        if (p <= 0 || p > 1) {
            throw new IllegalArgumentException("P must be between 0 and 1");
        }
        if (temperature <= 0) {
            throw new IllegalArgumentException("Temperature must be positive");
        }
        return new SamplingConfig(Strategy.TOP_P, temperature, 0, p);
    }
    
    /**
     * Top-P sampling without temperature scaling.
     */
    public static SamplingConfig topP(float p) {
        return topP(p, 1.0f);
    }
    
    // Getters
    public Strategy getStrategy() { return strategy; }
    public float getTemperature() { return temperature; }
    public int getK() { return k; }
    public float getP() { return p; }
    
    @Override
    public String toString() {
        switch (strategy) {
            case ARGMAX:
                return "SamplingConfig[argmax]";
            case TEMPERATURE:
                return String.format("SamplingConfig[temperature=%.2f]", temperature);
            case TOP_K:
                return String.format("SamplingConfig[topK=%d, temp=%.2f]", k, temperature);
            case TOP_P:
                return String.format("SamplingConfig[topP=%.2f, temp=%.2f]", p, temperature);
            default:
                return "SamplingConfig[unknown]";
        }
    }
}