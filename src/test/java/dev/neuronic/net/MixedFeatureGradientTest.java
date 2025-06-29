package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.HashUtils;
import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression test for MixedFeatureInputLayer gradient issues.
 * This test should catch the gradient accumulation bug and gradient explosion issues.
 */
public class MixedFeatureGradientTest {
    
    @Test
    public void testGradientClearingBetweenBatches() {
        // This test verifies that gradients are properly cleared between training steps
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        Feature[] features = {
            Feature.hashedEmbedding(10000, 8, "item"),
            Feature.passthrough("value")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train on same example multiple times
        Map<String, Object> input = new HashMap<>();
        input.put("item", HashUtils.hashString("test"));
        input.put("value", 1.0f);
        
        // Get initial prediction
        float initialPred = model.predictFloat(input);
        
        // First training step
        model.train(input, 5.0f);
        float afterFirstPred = model.predictFloat(input);
        
        // Calculate first step change in prediction
        float firstChange = Math.abs(afterFirstPred - initialPred);
        
        // Second training step (same input/target)
        model.train(input, 5.0f);
        float afterSecondPred = model.predictFloat(input);
        
        // Calculate second step change in prediction
        float secondChange = Math.abs(afterSecondPred - afterFirstPred);
        
        System.out.println("Initial pred: " + initialPred + ", after first: " + afterFirstPred + ", after second: " + afterSecondPred);
        System.out.println("First step change: " + firstChange);
        System.out.println("Second step change: " + secondChange);
        
        // Both steps should show similar magnitude of learning
        // If gradients accumulated incorrectly, second step would diverge dramatically
        assertTrue(secondChange <= firstChange * 2.0f, 
                  "Second step change should be reasonable, not accumulated");
    }
    
    @Test
    public void testGradientMagnitudeStability() {
        // Test that gradients don't explode with reasonable learning rates
        SgdOptimizer optimizer = new SgdOptimizer(0.01f); // Lower learning rate for stability
        
        Feature[] features = {
            Feature.hashedEmbedding(10000, 4, "item")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Track weight norms over training
        float[] weightNorms = new float[10];
        
        Map<String, Object> input = new HashMap<>();
        input.put("item", HashUtils.hashString("test_item"));
        
        for (int i = 0; i < 10; i++) {
            model.train(input, 3.0f);
            // Track prediction stability instead of weight norms
            float pred = model.predictFloat(input);
            weightNorms[i] = pred;
            System.out.println("Step " + i + " prediction: " + weightNorms[i]);
        }
        
        // Check predictions remain stable
        for (int i = 1; i < 10; i++) {
            assertFalse(Float.isNaN(weightNorms[i]), "Predictions should not become NaN");
            assertFalse(Float.isInfinite(weightNorms[i]), "Predictions should not become infinite");
        }
        // Predictions should converge toward target
        assertTrue(Math.abs(weightNorms[9] - 3.0f) < Math.abs(weightNorms[0] - 3.0f), 
                  "Predictions should move toward target");
    }
    
    @Test
    public void testHashedVsRegularEmbeddingGradients() {
        // Compare gradient behavior between hashed and regular embeddings
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        // Model with regular embedding
        Feature[] regularFeatures = {
            Feature.embedding(100, 4, "item")
        };
        
        NeuralNet regularNet = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(regularFeatures))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat regularModel = SimpleNet.ofFloatRegression(regularNet);
        
        // Model with hashed embedding
        Feature[] hashedFeatures = {
            Feature.hashedEmbedding(10000, 4, "item")
        };
        
        NeuralNet hashedNet = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(hashedFeatures))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat hashedModel = SimpleNet.ofFloatRegression(hashedNet);
        
        // Train both models
        Map<String, Object> regularInput = new HashMap<>();
        regularInput.put("item", 42); // Regular uses integer ID
        
        Map<String, Object> hashedInput = new HashMap<>();
        hashedInput.put("item", HashUtils.hashString("item_42")); // Hashed uses hash
        
        float target = 2.5f;
        
        // Get initial predictions
        float regularInit = regularModel.predictFloat(regularInput);
        float hashedInit = hashedModel.predictFloat(hashedInput);
        
        // Track initial predictions instead of weights
        
        // Single training step
        regularModel.train(regularInput, target);
        hashedModel.train(hashedInput, target);
        
        // Get final predictions
        float regularFinal = regularModel.predictFloat(regularInput);
        float hashedFinal = hashedModel.predictFloat(hashedInput);
        
        // Calculate prediction changes
        float regularChange = Math.abs(regularFinal - regularInit);
        float hashedChange = Math.abs(hashedFinal - hashedInit);
        
        System.out.println("Regular embedding prediction change: " + regularInit + " -> " + regularFinal + " (change: " + regularChange + ")");
        System.out.println("Hashed embedding prediction change: " + hashedInit + " -> " + hashedFinal + " (change: " + hashedChange + ")");
        
        // Both should learn (predictions should move toward target)
        assertTrue(regularChange > 0.001f, "Regular embedding should learn");
        assertTrue(hashedChange > 0.001f, "Hashed embedding should learn");
        
        // Learning rates should be comparable
        assertTrue(hashedChange < regularChange * 10, 
                  "Hashed embedding learning should not be orders of magnitude different");
        assertTrue(hashedChange > regularChange * 0.1, 
                  "Hashed embedding learning should not be orders of magnitude different");
    }
    
    @Test
    public void testMultipleHashedFeatures() {
        // Test with multiple hashed embedding features
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        Feature[] features = {
            Feature.hashedEmbedding(10000, 8, "domain"),
            Feature.hashedEmbedding(5000, 4, "app"),
            Feature.oneHot(3, "device"),
            Feature.passthrough("bid")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Create diverse training data
        Map<String, Object>[] inputs = new Map[5];
        float[] targets = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        
        for (int i = 0; i < 5; i++) {
            inputs[i] = new HashMap<>();
            inputs[i].put("domain", HashUtils.hashString("domain_" + i));
            inputs[i].put("app", HashUtils.hashString("app_" + i));
            inputs[i].put("device", i % 3);
            inputs[i].put("bid", 0.5f + i * 0.1f);
        }
        
        // Train and check predictions become diverse
        float[] initialPreds = new float[5];
        for (int i = 0; i < 5; i++) {
            initialPreds[i] = model.predictFloat(inputs[i]);
        }
        
        // Train multiple epochs
        for (int epoch = 0; epoch < 20; epoch++) {
            for (int i = 0; i < 5; i++) {
                model.train(inputs[i], targets[i]);
            }
        }
        
        // Check final predictions
        float[] finalPreds = new float[5];
        for (int i = 0; i < 5; i++) {
            finalPreds[i] = model.predictFloat(inputs[i]);
            System.out.println("Sample " + i + ": target=" + targets[i] + 
                             ", pred=" + finalPreds[i]);
        }
        
        // Verify predictions are diverse (not all the same)
        float minPred = Float.MAX_VALUE, maxPred = Float.MIN_VALUE;
        for (float pred : finalPreds) {
            minPred = Math.min(minPred, pred);
            maxPred = Math.max(maxPred, pred);
        }
        
        assertTrue(maxPred - minPred > 0.5f, 
                  "Predictions should be diverse, not all converged to mean");
    }
}