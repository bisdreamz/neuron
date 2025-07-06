package dev.neuronic.net;

import dev.neuronic.net.common.Utils;
import dev.neuronic.net.math.FastRandom;

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

    private SamplingStrategies() {
    } // Utility class

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
        if (temperature <= 0)
            throw new IllegalArgumentException("Temperature must be positive, got: " + temperature);

        // For very low temperature, just use argmax
        if (temperature < 0.01f)
            return argmax(probabilities);

        // Convert probabilities to logits, apply temperature, then resample
        float[] logits = new float[probabilities.length];
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > 0) {
                logits[i] = (float) Math.log(probabilities[i]) / temperature;
            } else {
                logits[i] = -1e10f / temperature;
            }
        }

        // Apply softmax to get new probabilities
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {
            maxLogit = Math.max(maxLogit, logit);
        }

        float sumExp = 0;
        for (int i = 0; i < logits.length; i++) {
            logits[i] = (float) Math.exp(logits[i] - maxLogit);
            sumExp += logits[i];
        }

        for (int i = 0; i < logits.length; i++) {
            logits[i] /= sumExp;
        }

        return sampleFromDistribution(logits, random);
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
        if (k <= 0 || k > probabilities.length) {
            throw new IllegalArgumentException("K must be between 1 and vocab size, got: " + k);
        }

        // Find top K indices
        Integer[] indices = new Integer[probabilities.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }

        // Partial sort to get top K
        java.util.Arrays.sort(indices, 0, Math.min(k + 1, indices.length),
                (a, b) -> Float.compare(probabilities[b], probabilities[a]));

        // Create distribution with only top K
        float[] topKProbs = new float[probabilities.length];
        float sum = 0;
        for (int i = 0; i < k && i < indices.length; i++) {
            topKProbs[indices[i]] = probabilities[indices[i]];
            sum += probabilities[indices[i]];
        }

        // Renormalize
        if (sum > 0) {
            for (int i = 0; i < topKProbs.length; i++) {
                topKProbs[i] /= sum;
            }
        }

        // Apply temperature if needed
        if (Math.abs(temperature - 1.0f) > 1e-6f) {
            return sampleWithTemperature(topKProbs, temperature, random);
        } else {
            return sampleFromDistribution(topKProbs, random);
        }
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
        if (p <= 0 || p > 1) {
            throw new IllegalArgumentException("P must be between 0 and 1, got: " + p);
        }

        // Sort indices by probability
        Integer[] indices = new Integer[probabilities.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        java.util.Arrays.sort(indices, (a, b) -> Float.compare(probabilities[b], probabilities[a]));

        // Find nucleus
        float cumSum = 0;
        int nucleusSize = 0;
        for (int i = 0; i < indices.length; i++) {
            cumSum += probabilities[indices[i]];
            nucleusSize++;
            if (cumSum >= p) {
                break;
            }
        }

        // Create distribution with only nucleus tokens
        float[] nucleusProbs = new float[probabilities.length];
        float sum = 0;
        for (int i = 0; i < nucleusSize; i++) {
            nucleusProbs[indices[i]] = probabilities[indices[i]];
            sum += probabilities[indices[i]];
        }

        // Renormalize
        if (sum > 0) {
            for (int i = 0; i < nucleusProbs.length; i++) {
                nucleusProbs[i] /= sum;
            }
        }

        // Apply temperature if needed
        if (Math.abs(temperature - 1.0f) > 1e-6f) {
            return sampleWithTemperature(nucleusProbs, temperature, random);
        } else {
            return sampleFromDistribution(nucleusProbs, random);
        }
    }

    /**
     * Sample an index from a probability distribution.
     */
    private static int sampleFromDistribution(float[] probabilities, FastRandom random) {
        float randomValue = random.nextFloat();
        float cumSum = 0;

        for (int i = 0; i < probabilities.length; i++) {
            cumSum += probabilities[i];

            if (randomValue < cumSum) {
                return i;
            }
        }

        // Fallback to last index
        return probabilities.length - 1;
    }
}