package dev.neuronic.net;

import dev.neuronic.net.common.Utils;
import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.math.NetMath;

/**
 * Sampling strategies for language model generation.
 *
 * <p>These strategies operate on softmax probability distributions to select
 * tokens during text generation. Different strategies provide different
 * trade-offs between quality, diversity, and coherence.
 *
 * <p><b>Common Strategies:</b>
 * <ul>
 *   <li><b>Argmax:</b> Always pick highest probability (deterministic)</li>
 *   <li><b>Temperature:</b> Adjust distribution sharpness before sampling</li>
 *   <li><b>Top-K:</b> Sample only from K most likely tokens</li>
 *   <li><b>Top-P (Nucleus):</b> Sample from smallest set with cumulative prob > P</li>
 * </ul>
 */
public final class SamplingStrategies {

    private SamplingStrategies() {} // Utility class

    /**
     * Sample using argmax (always pick highest probability).
     * Deterministic and focused but can be repetitive.
     *
     * @param probabilities softmax output probabilities
     * @return index of highest probability token
     */
    public static int argmax(float[] probabilities) {
        return Utils.argmax(probabilities);
    }

    /**
     * Sample with temperature scaling.
     *
     * @param probabilities softmax output probabilities
     * @param temperature   controls randomness (0.1=focused, 1.0=normal, 2.0=creative)
     * @param random       FastRandom instance for sampling
     * @return sampled token index
     */
    public static int sampleWithTemperature(float[] probabilities, float temperature, FastRandom random) {
        return sampleWithTemperature(probabilities, temperature, null, random);
    }
    
    /**
     * Sample with temperature scaling, excluding specified indices.
     *
     * @param probabilities softmax output probabilities
     * @param temperature   controls randomness (0.1=focused, 1.0=normal, 2.0=creative)
     * @param excludeIndices indices to exclude from sampling (can be null)
     * @param random       FastRandom instance for sampling
     * @return sampled token index
     */
    public static int sampleWithTemperature(float[] probabilities, float temperature, 
                                          int[] excludeIndices, FastRandom random) {
        if (temperature <= 0)
            throw new IllegalArgumentException("Temperature must be positive, got: " + temperature);

        // For very low temperature, just use argmax (with exclusions)
        if (temperature < 0.01f) {
            if (excludeIndices == null || excludeIndices.length == 0) {
                return argmax(probabilities);
            } else {
                // Find argmax excluding specified indices
                int bestIdx = -1;
                float bestProb = -1;
                boolean[] excluded = new boolean[probabilities.length];
                for (int idx : excludeIndices) {
                    if (idx >= 0 && idx < excluded.length) {
                        excluded[idx] = true;
                    }
                }
                for (int i = 0; i < probabilities.length; i++) {
                    if (!excluded[i] && probabilities[i] > bestProb) {
                        bestProb = probabilities[i];
                        bestIdx = i;
                    }
                }
                if (bestIdx == -1) {
                    throw new IllegalArgumentException("All indices are excluded");
                }
                return bestIdx;
            }
        }

        // Apply temperature scaling
        float[] scaledProbs = new float[probabilities.length];
        if (excludeIndices == null || excludeIndices.length == 0) {
            NetMath.samplingTemperature(probabilities, temperature, scaledProbs);
        } else {
            NetMath.samplingTemperatureWithExclusions(probabilities, temperature, excludeIndices, scaledProbs);
        }

        return sampleFromDistribution(scaledProbs, random);
    }

    /**
     * Sample from top-K tokens.
     *
     * @param probabilities softmax output probabilities
     * @param k             number of top tokens to consider
     * @param temperature   optional temperature scaling (1.0 = no scaling)
     * @param random       FastRandom instance for sampling
     * @return sampled token index
     */
    public static int sampleTopK(float[] probabilities, int k, float temperature, FastRandom random) {
        return sampleTopK(probabilities, k, temperature, null, random);
    }
    
    /**
     * Sample from top-K tokens, excluding specified indices.
     *
     * @param probabilities softmax output probabilities
     * @param k             number of top tokens to consider
     * @param temperature   optional temperature scaling (1.0 = no scaling)
     * @param excludeIndices indices to exclude from sampling (can be null)
     * @param random       FastRandom instance for sampling
     * @return sampled token index
     */
    public static int sampleTopK(float[] probabilities, int k, float temperature, 
                                int[] excludeIndices, FastRandom random) {
        if (k <= 0 || k > probabilities.length) {
            throw new IllegalArgumentException("K must be between 1 and vocab size, got: " + k);
        }

        // Apply Top-K filtering
        float[] topKProbs = new float[probabilities.length];
        if (excludeIndices == null || excludeIndices.length == 0) {
            NetMath.samplingTopK(probabilities, k, topKProbs);
        } else {
            NetMath.samplingTopKWithExclusions(probabilities, k, excludeIndices, topKProbs);
        }

        // Apply temperature if needed
        if (Math.abs(temperature - 1.0f) > 1e-6f) {
            NetMath.samplingTemperatureInPlace(topKProbs, temperature);
        }

        return sampleFromDistribution(topKProbs, random);
    }

    /**
     * Sample from nucleus (top-P).
     *
     * @param probabilities softmax output probabilities
     * @param p             cumulative probability threshold (e.g., 0.9)
     * @param temperature   optional temperature scaling
     * @param random       FastRandom instance for sampling
     * @return sampled token index
     */
    public static int sampleTopP(float[] probabilities, float p, float temperature, FastRandom random) {
        return sampleTopP(probabilities, p, temperature, null, random);
    }
    
    /**
     * Sample from nucleus (top-P), excluding specified indices.
     *
     * @param probabilities softmax output probabilities
     * @param p             cumulative probability threshold (e.g., 0.9)
     * @param temperature   optional temperature scaling
     * @param excludeIndices indices to exclude from sampling (can be null)
     * @param random       FastRandom instance for sampling
     * @return sampled token index
     */
    public static int sampleTopP(float[] probabilities, float p, float temperature,
                                int[] excludeIndices, FastRandom random) {
        if (p <= 0 || p > 1) {
            throw new IllegalArgumentException("P must be between 0 and 1, got: " + p);
        }

        // Apply Top-P filtering
        float[] topPProbs = new float[probabilities.length];
        if (excludeIndices == null || excludeIndices.length == 0) {
            NetMath.samplingTopP(probabilities, p, topPProbs);
        } else {
            NetMath.samplingTopPWithExclusions(probabilities, p, excludeIndices, topPProbs);
        }

        // Apply temperature if needed
        if (Math.abs(temperature - 1.0f) > 1e-6f) {
            NetMath.samplingTemperatureInPlace(topPProbs, temperature);
        }

        return sampleFromDistribution(topPProbs, random);
    }

    /**
     * Sample an index from a probability distribution.
     */
    private static int sampleFromDistribution(float[] probabilities, FastRandom random) {
        float randomValue = random.nextFloat();
        return NetMath.samplingDrawFromDistribution(probabilities, randomValue);
    }
}