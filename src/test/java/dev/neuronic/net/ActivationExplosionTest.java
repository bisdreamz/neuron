package dev.neuronic.net;

import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to prove the theory of activation explosion causing model collapse.
 * This test bypasses the MixedFeatureInputLayer to isolate the effect of
 * unnormalized inputs on a deep ReLU network.
 */
public class ActivationExplosionTest {

    private static final int INPUT_SIZE = 128;
    private static final int NUM_SAMPLES = 2000;
    private static final int NUM_EPOCHS = 3;

    /**
     * Test Case 1: Unnormalized Inputs (Should Collapse)
     * Simulates the output of MixedFeatureInputLayer where some features (embeddings)
     * have a much larger magnitude than others (one-hot, scaled).
     */
    @Test
    void testWithUnnormalizedInput_ShouldCollapse() {
        System.out.println("=== Testing with Unnormalized Input (Expecting Collapse) ===");
        NeuralNet net = createDeepReluNetwork();
        float[][] inputs = generateMixedMagnitudeData(false); // false = don't normalize
        float[] targets = generateTargets(inputs);

        boolean didCollapse = trainAndCheckForCollapse(net, inputs, targets);

        assertTrue(didCollapse, "The network should collapse when trained on unnormalized, mixed-magnitude inputs.");
    }

    /**
     * Test Case 2: Normalized Inputs (Should Learn)
     * Uses the exact same data as Test 1, but applies manual normalization
     * before feeding it into the network, simulating a LayerNorm layer.
     */
    @Test
    void testWithNormalizedInput_ShouldLearn() {
        System.out.println("\n=== Testing with Normalized Input (Expecting Success) ===");
        NeuralNet net = createDeepReluNetwork();
        float[][] inputs = generateMixedMagnitudeData(true); // true = normalize
        float[] targets = generateTargets(inputs); // Generate targets AFTER normalization

        boolean didCollapse = trainAndCheckForCollapse(net, inputs, targets);

        assertFalse(didCollapse, "The network should learn successfully when inputs are normalized.");
    }

    private NeuralNet createDeepReluNetwork() {
        return NeuralNet.newBuilder()
                .input(INPUT_SIZE)
                .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
                .layer(Layers.hiddenDenseRelu(256))
                .layer(Layers.hiddenDenseRelu(128))
                .layer(Layers.hiddenDenseRelu(64))
                .output(Layers.outputLinearRegression(1));
    }

    private float[][] generateMixedMagnitudeData(boolean normalize) {
        Random rand = new Random(42);
        float[][] data = new float[NUM_SAMPLES][INPUT_SIZE];
        int splitPoint = INPUT_SIZE / 2;

        for (int i = 0; i < NUM_SAMPLES; i++) {
            // Part 1: Simulates one-hot/scaled features (range [0, 1])
            for (int j = 0; j < splitPoint; j++) {
                data[i][j] = rand.nextFloat();
            }
            // Part 2: Simulates embedding features (large magnitude, range [-5, 5])
            for (int j = splitPoint; j < INPUT_SIZE; j++) {
                data[i][j] = (rand.nextFloat() - 0.5f) * 10.0f;
            }

            if (normalize) {
                normalizeVector(data[i]);
            }
        }
        return data;
    }

    private float[] generateTargets(float[][] inputs) {
        float[] targets = new float[NUM_SAMPLES];
        // Simple target: sum of a few features, so there's a clear signal to learn.
        for (int i = 0; i < NUM_SAMPLES; i++) {
            targets[i] = inputs[i][0] + inputs[i][INPUT_SIZE / 2] * 0.5f;
        }
        return targets;
    }

    private void normalizeVector(float[] vector) {
        float sum = 0.0f;
        for (float v : vector) {
            sum += v;
        }
        float mean = sum / vector.length;

        float sumSq = 0.0f;
        for (float v : vector) {
            sumSq += (v - mean) * (v - mean);
        }
        float stdDev = (float) Math.sqrt(sumSq / vector.length);

        if (stdDev < 1e-6) {
            return; // Avoid division by zero for constant vectors
        }

        for (int i = 0; i < vector.length; i++) {
            vector[i] = (vector[i] - mean) / stdDev;
        }
    }

    private boolean trainAndCheckForCollapse(NeuralNet net, float[][] inputs, float[] targets) {
        List<Float> predictions = new ArrayList<>();
        final int checkInterval = NUM_SAMPLES / 4;

        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            for (int i = 0; i < NUM_SAMPLES; i++) {
                net.train(inputs[i], new float[]{targets[i]});

                if (i > 0 && i % checkInterval == 0) {
                    // Check predictions on a fixed subset of data
                    predictions.clear();
                    for (int j = 0; j < 10; j++) {
                        predictions.add(net.predict(inputs[j])[0]);
                    }
                    double stdDev = calculateStdDev(predictions);
                    System.out.printf("Epoch %d, Sample %d: Prediction StdDev=%.6f, First Pred=%.4f\n",
                            epoch, i, stdDev, predictions.get(0));

                    // If StdDev is near zero, it has collapsed.
                    if (stdDev < 1e-4 && i > NUM_SAMPLES / 2) {
                        System.out.println("Collapse detected!");
                        return true;
                    }
                }
            }
        }
        System.out.println("Training finished without collapsing.");
        return false;
    }

    private double calculateStdDev(List<Float> data) {
        if (data.size() < 2) return 0.0;
        double sum = 0.0;
        for (float v : data) {
            sum += v;
        }
        double mean = sum / data.size();
        double stdDev = 0.0;
        for (float v : data) {
            stdDev += Math.pow(v - mean, 2);
        }
        return Math.sqrt(stdDev / data.size());
    }
}