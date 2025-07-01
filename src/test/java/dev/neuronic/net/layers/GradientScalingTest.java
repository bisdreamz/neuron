package dev.neuronic.net.layers;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test to verify gradient scaling is working correctly.
 * 
 * This test ensures that:
 * 1. Gradients are properly scaled by 1/batchSize before optimization
 * 2. The effective learning rate doesn't scale with batch size
 * 3. Training with different batch sizes produces similar results
 */
public class GradientScalingTest {
    
    @Test
    void testGradientScalingWithDifferentBatchSizes() {
        // Use SGD optimizer for predictable updates (no momentum)
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        
        // Create first network
        NeuralNet net1 = createNetwork(optimizer);
        MixedFeatureInputLayer layer1 = (MixedFeatureInputLayer) net1.getInputLayer();
        
        // Store initial embedding
        float[] initialEmbed = layer1.getEmbedding(0, 5).clone();
        
        // Create second network
        NeuralNet net2 = createNetwork(optimizer);
        MixedFeatureInputLayer layer2 = (MixedFeatureInputLayer) net2.getInputLayer();
        
        // Copy initial embedding to second network to ensure identical starting point
        System.arraycopy(initialEmbed, 0, layer2.getEmbedding(0, 5), 0, initialEmbed.length);
        
        // Create identical training data
        Random rand = new Random(42);
        float[][] allInputs = new float[32][];
        float[][] allTargets = new float[32][];
        
        for (int i = 0; i < 32; i++) {
            allInputs[i] = new float[]{5.0f, rand.nextFloat()}; // Always use embedding 5
            allTargets[i] = new float[]{1.0f}; // Same target
        }
        
        // Train network 1 with batch size 1 (32 separate updates)
        for (int i = 0; i < 32; i++) {
            net1.train(allInputs[i], allTargets[i]);
        }
        
        // Train network 2 with batch size 32 (1 batch update)
        net2.trainBatch(allInputs, allTargets);
        
        // Get final embedding states
        float[] final1 = layer1.getEmbedding(0, 5);
        float[] final2 = layer2.getEmbedding(0, 5);
        
        // Calculate total change magnitude
        float change1 = calculateChange(initialEmbed, final1);
        float change2 = calculateChange(initialEmbed, final2);
        
        System.out.printf("Change with batch size 1: %.6f\n", change1);
        System.out.printf("Change with batch size 32: %.6f\n", change2);
        
        // The changes will be different due to update patterns:
        // - Sequential: each gradient applied immediately (compounds)
        // - Batch: all gradients averaged and applied once
        float ratio = change2 / change1;
        System.out.printf("Ratio (batch32/batch1): %.3f\n", ratio);
        
        // With proper gradient scaling, batch update should be smaller but not tiny
        // Sequential updates compound, so batch ratio typically 0.3-0.7
        assertTrue(ratio > 0.3f && ratio < 0.7f, 
            String.format("Gradient scaling issue: ratio %.3f is outside expected range [0.3, 0.7]", ratio));
        
        // Also verify both made meaningful updates
        assertTrue(change1 > 0.01f, "Sequential training should update embeddings");
        assertTrue(change2 > 0.01f, "Batch training should update embeddings");
    }
    
    @Test
    void testBatchGradientAveraging() {
        // Test that gradients are properly averaged, not summed
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0f);
        
        NeuralNet net = createNetwork(optimizer);
        MixedFeatureInputLayer inputLayer = (MixedFeatureInputLayer) net.getInputLayer();
        
        // Get initial state
        float[] initialEmbed = inputLayer.getEmbedding(0, 10).clone();
        
        // Create batch with same input repeated
        int batchSize = 16;
        float[][] inputs = new float[batchSize][];
        float[][] targets = new float[batchSize][];
        
        for (int i = 0; i < batchSize; i++) {
            inputs[i] = new float[]{10.0f, 0.5f}; // Same input
            targets[i] = new float[]{2.0f}; // Same target
        }
        
        // Train with batch
        net.trainBatch(inputs, targets);
        
        // Get updated embedding
        float[] updatedEmbed = inputLayer.getEmbedding(0, 10);
        
        // Calculate change
        float change = calculateChange(initialEmbed, updatedEmbed);
        
        // Now train a single sample with same data
        NeuralNet netSingle = createNetwork(optimizer);
        MixedFeatureInputLayer layerSingle = (MixedFeatureInputLayer) netSingle.getInputLayer();
        
        // Set same initial embedding
        System.arraycopy(initialEmbed, 0, layerSingle.getEmbedding(0, 10), 0, initialEmbed.length);
        
        // Train single sample
        netSingle.train(inputs[0], targets[0]);
        
        float[] singleUpdate = layerSingle.getEmbedding(0, 10);
        float singleChange = calculateChange(initialEmbed, singleUpdate);
        
        System.out.printf("Batch update magnitude: %.6f\n", change);
        System.out.printf("Single update magnitude: %.6f\n", singleChange);
        System.out.printf("Ratio: %.3f\n", change / singleChange);
        
        // With proper averaging, batch update should be similar to single update
        // AdamW's momentum can cause some variation, so allow wider range
        float ratio = change / singleChange;
        assertTrue(ratio > 0.5f && ratio < 1.5f,
            String.format("Batch averaging issue: ratio %.3f suggests gradients might not be properly scaled", ratio));
        
        // More importantly, verify the batch update isn't massive (which would indicate no averaging)
        assertTrue(ratio < 2.0f, 
            String.format("Batch update is too large (%.3fx single), suggesting gradients are summed not averaged", ratio));
    }
    
    @Test
    void testGradientScalingConsistency() {
        // Test that gradient scaling is consistent across different features
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.0f);
        
        Feature[] features = {
            Feature.embedding(100, 8, "feature1"),
            Feature.embedding(100, 16, "feature2"),
            Feature.passthrough("value")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
        
        MixedFeatureInputLayer inputLayer = (MixedFeatureInputLayer) net.getInputLayer();
        
        // Store initial embeddings
        float[] init1 = inputLayer.getEmbedding(0, 20).clone();
        float[] init2 = inputLayer.getEmbedding(1, 30).clone();
        
        // Create batch using both embeddings
        float[][] inputs = new float[10][];
        float[][] targets = new float[10][];
        
        Random rand = new Random(123);
        for (int i = 0; i < 10; i++) {
            inputs[i] = new float[]{20.0f, 30.0f, rand.nextFloat()};
            targets[i] = new float[]{rand.nextFloat() * 2};
        }
        
        // Train batch
        net.trainBatch(inputs, targets);
        
        // Check updates
        float[] final1 = inputLayer.getEmbedding(0, 20);
        float[] final2 = inputLayer.getEmbedding(1, 30);
        
        float change1 = calculateChange(init1, final1);
        float change2 = calculateChange(init2, final2);
        
        System.out.printf("Feature 1 change: %.6f\n", change1);
        System.out.printf("Feature 2 change: %.6f\n", change2);
        
        // Both should have been updated
        assertTrue(change1 > 0.0001f, "Feature 1 embedding should be updated");
        assertTrue(change2 > 0.0001f, "Feature 2 embedding should be updated");
        
        // The ratio should be reasonable (not orders of magnitude different)
        float ratio = Math.max(change1, change2) / Math.min(change1, change2);
        assertTrue(ratio < 10.0f, 
            String.format("Feature updates are too different: ratio %.2f", ratio));
    }
    
    private NeuralNet createNetwork(dev.neuronic.net.optimizers.Optimizer optimizer) {
        Feature[] features = {
            Feature.embedding(50, 16, "token"),
            Feature.passthrough("value")
        };
        
        return NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
    }
    
    private float calculateChange(float[] before, float[] after) {
        float sum = 0;
        for (int i = 0; i < before.length; i++) {
            float diff = after[i] - before[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }
}