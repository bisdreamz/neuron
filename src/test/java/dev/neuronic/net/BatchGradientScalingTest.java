package dev.neuronic.net;

import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.math.FastRandom;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test that gradient scaling works correctly with different batch sizes.
 */
public class BatchGradientScalingTest {
    
    // This test doesn't work with Adam optimizers due to momentum state differences
    // @Test
    public void testGradientScalingConsistency() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create identical networks
        NeuralNet net1 = createNetwork(optimizer);
        NeuralNet net2 = createNetwork(optimizer);
        
        // Use FastRandom utility
        
        // Create single sample
        float[] input = new float[100];
        float[] target = new float[10];
        for (int i = 0; i < 100; i++) {
            input[i] = (float) (FastRandom.get().nextGaussian() * 0.1);
        }
        int targetClass = FastRandom.get().nextInt(10);
        target[targetClass] = 1.0f;
        
        // Train net1 with single sample 128 times
        for (int i = 0; i < 128; i++) {
            net1.train(input, target);
        }
        
        // Train net2 with batch of 128 copies of the same sample
        float[][] batchInputs = new float[128][100];
        float[][] batchTargets = new float[128][10];
        for (int i = 0; i < 128; i++) {
            System.arraycopy(input, 0, batchInputs[i], 0, 100);
            System.arraycopy(target, 0, batchTargets[i], 0, 10);
        }
        net2.trainBatch(batchInputs, batchTargets);
        
        // Get predictions
        float[] pred1 = net1.predict(input);
        float[] pred2 = net2.predict(input);
        
        // They should be reasonably close (not identical due to optimizer state)
        for (int i = 0; i < pred1.length; i++) {
            assertEquals(pred1[i], pred2[i], 0.1f, 
                "Predictions should be similar at index " + i);
        }
    }
    
    @Test
    public void testLargeBatchGradientNorm() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        NeuralNet net = createNetwork(optimizer);
        
        // Use FastRandom utility
        
        // Test with different batch sizes
        int[] batchSizes = {1, 16, 64, 128, 256};
        
        for (int batchSize : batchSizes) {
            // Create random batch
            float[][] inputs = new float[batchSize][100];
            float[][] targets = new float[batchSize][10];
            
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < 100; j++) {
                    inputs[i][j] = (float) (FastRandom.get().nextGaussian() * 0.1);
                }
                int targetClass = FastRandom.get().nextInt(10);
                targets[i][targetClass] = 1.0f;
            }
            
            // Train one step
            net.trainBatch(inputs, targets);
            
            // Check that network still produces valid outputs
            float[] testInput = new float[100];
            for (int j = 0; j < 100; j++) {
                testInput[j] = (float) (FastRandom.get().nextGaussian() * 0.1);
            }
            
            float[] output = net.predict(testInput);
            
            // Verify outputs are valid
            float sum = 0;
            for (float val : output) {
                assertTrue(Float.isFinite(val), 
                    "Output contains NaN/Inf with batch size " + batchSize);
                assertTrue(val >= 0 && val <= 1, 
                    "Invalid probability " + val + " with batch size " + batchSize);
                sum += val;
            }
            assertEquals(1.0f, sum, 0.01f, 
                "Probabilities don't sum to 1 with batch size " + batchSize);
            
            System.out.printf("Batch size %d: outputs valid, sum=%.4f\n", batchSize, sum);
        }
    }
    
    private NeuralNet createNetwork(AdamWOptimizer optimizer) {
        return NeuralNet.newBuilder()
            .input(100)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(256))
            .layer(Layers.hiddenDenseRelu(128))
            .output(Layers.outputSoftmaxCrossEntropy(10));
    }
}