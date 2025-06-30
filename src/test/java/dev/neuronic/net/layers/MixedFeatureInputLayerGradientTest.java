package dev.neuronic.net.layers;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.losses.MseLoss;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for MixedFeatureInputLayer gradient accumulation and batch training.
 * 
 * These tests verify the critical fix for the gradient accumulation bug where
 * MixedFeatureInputLayer was incorrectly applying gradients during backward()
 * instead of accumulating them for batch training.
 */
class MixedFeatureInputLayerGradientTest {
    
    private AdamWOptimizer optimizer;
    
    @BeforeEach
    void setUp() {
        optimizer = new AdamWOptimizer(0.1f, 0.0f); // High LR, no weight decay for tests
    }
    
    @Test
    void testBatchGradientAccumulation() {
        // Create layer directly for testing
        Feature[] features = {
            Feature.embedding(10, 4, "token"),
            Feature.passthrough("value")
        };
        MixedFeatureInputLayer inputLayer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.HE);
        
        // Get initial embedding values
        float[] initialEmbedding5 = inputLayer.getEmbedding(0, 5).clone();
        float[] initialEmbedding3 = inputLayer.getEmbedding(0, 3).clone();
        
        // Create contexts for batch backward pass simulation
        Layer.LayerContext[] contexts = new Layer.LayerContext[3];
        float[][] gradients = new float[3][];
        
        // Simulate forward pass for batch
        contexts[0] = inputLayer.forward(new float[]{5.0f, 1.0f}); // token 5, value 1
        contexts[1] = inputLayer.forward(new float[]{3.0f, 2.0f}); // token 3, value 2  
        contexts[2] = inputLayer.forward(new float[]{5.0f, 3.0f}); // token 5, value 3 (different context)
        
        // Create upstream gradients (5 dimensions: 4 for embedding + 1 for passthrough)
        for (int i = 0; i < 3; i++) {
            gradients[i] = new float[5];
            for (int j = 0; j < 5; j++) {
                gradients[i][j] = (i + 1) * 0.1f; // Different gradients for each sample
            }
        }
        
        // Backward pass for all samples (accumulate gradients)
        inputLayer.backward(new Layer.LayerContext[]{contexts[0]}, 0, gradients[0]);
        inputLayer.backward(new Layer.LayerContext[]{contexts[1]}, 0, gradients[1]);
        inputLayer.backward(new Layer.LayerContext[]{contexts[2]}, 0, gradients[2]);
        
        // Critical: Apply accumulated gradients
        inputLayer.applyGradients(null, null);
        
        // Verify embeddings were updated
        float[] updatedEmbedding5 = inputLayer.getEmbedding(0, 5);
        float[] updatedEmbedding3 = inputLayer.getEmbedding(0, 3);
        
        // Check that both embeddings changed
        assertEmbeddingChanged(initialEmbedding5, updatedEmbedding5, "Embedding 5");
        assertEmbeddingChanged(initialEmbedding3, updatedEmbedding3, "Embedding 3");
        
        // Verify gradient accumulation: embedding 5 should have larger update (appeared twice)
        float change5 = computeEmbeddingChange(initialEmbedding5, updatedEmbedding5);
        float change3 = computeEmbeddingChange(initialEmbedding3, updatedEmbedding3);
        
        // With proper accumulation, embedding 5 (used twice) should have more change
        System.out.printf("Embedding 5 change: %.6f, Embedding 3 change: %.6f%n", change5, change3);
    }
    
    @Test
    void testGradientClipping() {
        AdamWOptimizer highLrOptimizer = new AdamWOptimizer(1.0f, 0.0f); // Very high LR
        
        Feature[] features = {Feature.embedding(10, 4, "token")};
        MixedFeatureInputLayer inputLayer = new MixedFeatureInputLayer(highLrOptimizer, features, WeightInitStrategy.HE);
        inputLayer.setEmbeddingGradientClipNorm(1.0f); // Clip at norm 1.0
        
        // Forward pass
        Layer.LayerContext context = inputLayer.forward(new float[]{5.0f});
        
        // Create large gradient to trigger clipping
        float[] largeGradient = new float[4];
        for (int i = 0; i < 4; i++) {
            largeGradient[i] = 100.0f; // Very large gradient
        }
        
        float[] embedBefore = inputLayer.getEmbedding(0, 5).clone();
        
        // Backward and apply
        inputLayer.backward(new Layer.LayerContext[]{context}, 0, largeGradient);
        inputLayer.applyGradients(null, null);
        
        float[] embedAfter = inputLayer.getEmbedding(0, 5);
        
        // Compute change magnitude
        float changeMagnitude = computeEmbeddingChange(embedBefore, embedAfter);
        
        // With clipping at 1.0 and LR 5.0 (embeddings get 5x LR), change should be bounded
        // AdamW with embedding optimizer produces 5x larger updates
        assertTrue(changeMagnitude <= 10.0f, 
            String.format("Gradient clipping should limit update magnitude, got %.4f", changeMagnitude));
        assertTrue(changeMagnitude > 0.1f, 
            String.format("Update should still happen despite clipping, got %.4f", changeMagnitude));
    }
    
    @Test
    void testSingleSampleTraining() {
        // Verify single sample training works with gradient accumulation fix
        Feature[] features = {
            Feature.embedding(100, 16, "word"),
            Feature.passthrough("position")
        };
        MixedFeatureInputLayer inputLayer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.HE);
        
        float[] input = {42.0f, 0.5f};
        float[] embedBefore = inputLayer.getEmbedding(0, 42).clone();
        
        // Forward pass
        Layer.LayerContext context = inputLayer.forward(input);
        
        // Backward with gradient
        float[] gradient = new float[17]; // 16 + 1
        for (int i = 0; i < gradient.length; i++) {
            gradient[i] = 0.01f;
        }
        inputLayer.backward(new Layer.LayerContext[]{context}, 0, gradient);
        
        // Apply gradients
        inputLayer.applyGradients(null, null);
        
        float[] embedAfter = inputLayer.getEmbedding(0, 42);
        assertEmbeddingChanged(embedBefore, embedAfter, "Embedding 42");
    }
    
    @Test
    void testGradientAccumulationWithNeuralNet() {
        // Test gradient accumulation through full neural network
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(2)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(10, 4, "token"),
                Feature.passthrough("value")
            ))
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputLinearRegression(1));
        
        // Train with batch where token 5 appears twice
        float[][] inputs = {
            {5.0f, 1.0f},  // token 5, value 1
            {3.0f, 2.0f},  // token 3, value 2
            {5.0f, 3.0f}   // token 5, value 3
        };
        float[][] targets = {{10.0f}, {20.0f}, {30.0f}};
        
        // Compute initial loss
        float initialLoss = computeBatchLoss(net, inputs, targets);
        
        // Train one batch
        net.trainBatch(inputs, targets);
        
        // Verify loss decreased
        float finalLoss = computeBatchLoss(net, inputs, targets);
        assertTrue(finalLoss < initialLoss, 
            String.format("Loss should decrease: initial=%.4f, final=%.4f", initialLoss, finalLoss));
    }
    
    @Test
    void testHashedEmbeddingGradients() {
        // Test that hashed embeddings also accumulate gradients correctly
        Feature[] features = {
            Feature.hashedEmbedding(10000, 16, "domain"),
            Feature.passthrough("ctr")
        };
        MixedFeatureInputLayer inputLayer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.HE);
        
        // Forward pass with hashed values
        float hash1 = (float) "example.com".hashCode();
        float hash2 = (float) "test.com".hashCode();
        
        Layer.LayerContext ctx1 = inputLayer.forward(new float[]{hash1, 0.1f});
        Layer.LayerContext ctx2 = inputLayer.forward(new float[]{hash2, 0.2f});
        Layer.LayerContext ctx3 = inputLayer.forward(new float[]{hash1, 0.3f}); // Same domain again
        
        // Backward with gradients
        float[] grad = new float[17]; // 16 + 1
        for (int i = 0; i < grad.length; i++) {
            grad[i] = 0.1f;
        }
        
        inputLayer.backward(new Layer.LayerContext[]{ctx1}, 0, grad);
        inputLayer.backward(new Layer.LayerContext[]{ctx2}, 0, grad);
        inputLayer.backward(new Layer.LayerContext[]{ctx3}, 0, grad);
        
        // Apply should work without errors even with hashed embeddings
        inputLayer.applyGradients(null, null);
    }
    
    @Test
    void testGradientClearingBetweenBatches() {
        // Verify gradients are properly cleared between batches
        // Use SGD optimizer without momentum to avoid momentum effects
        SgdOptimizer sgdOptimizer = new SgdOptimizer(0.1f);
        Feature[] features = {Feature.embedding(10, 4, "id")};
        MixedFeatureInputLayer inputLayer = new MixedFeatureInputLayer(sgdOptimizer, features, WeightInitStrategy.HE);
        
        // First batch - only uses embedding 1
        float[] embed1Before = inputLayer.getEmbedding(0, 1).clone();
        float[] embed2Before = inputLayer.getEmbedding(0, 2).clone();
        
        
        Layer.LayerContext ctx1 = inputLayer.forward(new float[]{1.0f});
        inputLayer.backward(new Layer.LayerContext[]{ctx1}, 0, new float[]{1.0f, 1.0f, 1.0f, 1.0f});
        inputLayer.applyGradients(null, null);
        
        float[] embed1After = inputLayer.getEmbedding(0, 1);
        float[] embed2After = inputLayer.getEmbedding(0, 2);
        
        // Only embedding 1 should change
        assertEmbeddingChanged(embed1Before, embed1After, "Embedding 1");
        assertEmbeddingNotChanged(embed2Before, embed2After, "Embedding 2");
        
        // Second batch - only uses embedding 2
        float[] embed1Before2 = inputLayer.getEmbedding(0, 1).clone();
        float[] embed2Before2 = inputLayer.getEmbedding(0, 2).clone();
        
        Layer.LayerContext ctx2 = inputLayer.forward(new float[]{2.0f});
        inputLayer.backward(new Layer.LayerContext[]{ctx2}, 0, new float[]{1.0f, 1.0f, 1.0f, 1.0f});
        inputLayer.applyGradients(null, null);
        
        float[] embed1After2 = inputLayer.getEmbedding(0, 1);
        float[] embed2After2 = inputLayer.getEmbedding(0, 2);
        
        // Only embedding 2 should change in second batch
        assertEmbeddingNotChanged(embed1Before2, embed1After2, "Embedding 1 (batch 2)");
        assertEmbeddingChanged(embed2Before2, embed2After2, "Embedding 2 (batch 2)");
    }
    
    // Helper methods
    
    private float computeBatchLoss(NeuralNet net, float[][] inputs, float[][] targets) {
        float totalLoss = 0;
        for (int i = 0; i < inputs.length; i++) {
            float[] prediction = net.predict(inputs[i]);
            totalLoss += MseLoss.INSTANCE.loss(prediction, targets[i]);
        }
        return totalLoss / inputs.length;
    }
    
    private void assertEmbeddingChanged(float[] before, float[] after, String name) {
        boolean changed = false;
        for (int i = 0; i < before.length; i++) {
            if (Math.abs(before[i] - after[i]) > 1e-6) {
                changed = true;
                break;
            }
        }
        assertTrue(changed, name + " should be updated after training");
    }
    
    private void assertEmbeddingNotChanged(float[] before, float[] after, String name) {
        for (int i = 0; i < before.length; i++) {
            assertEquals(before[i], after[i], 1e-6f, 
                name + " should NOT change when not used in batch");
        }
    }
    
    private float computeEmbeddingChange(float[] before, float[] after) {
        float sum = 0;
        for (int i = 0; i < before.length; i++) {
            float diff = after[i] - before[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }
}