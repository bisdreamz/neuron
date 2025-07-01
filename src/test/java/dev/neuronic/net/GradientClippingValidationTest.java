package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to verify gradient clipping is working properly with embeddings.
 * This test creates scenarios where gradient explosion would occur without clipping.
 */
public class GradientClippingValidationTest {
    
    @Test
    public void testGradientClippingPreventsExplosion() {
        System.out.println("\n=== GRADIENT CLIPPING VALIDATION TEST ===\n");
        
        // Create a model with embeddings that would be prone to gradient explosion
        Feature[] features = {
            Feature.embedding(1000, 64, "item"),      // High-dim embeddings
            Feature.embedding(500, 32, "category"),   
            Feature.passthrough("value")
        };
        
        // Use higher learning rate to trigger gradient issues
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.01f); // Intentionally high LR
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Track predictions to detect explosion
        float[] predictions = new float[20];
        boolean exploded = false;
        
        System.out.println("Training with high learning rate (0.1) - should explode without clipping:");
        
        for (int i = 0; i < 20; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", rand.nextInt(1000));
            input.put("category", rand.nextInt(500));
            input.put("value", rand.nextFloat() * 10);
            
            // Large target values to encourage gradient explosion
            float target = 100.0f + rand.nextFloat() * 50.0f;
            
            // Train with standard method (no clipping)
            model.train(input, target);
            
            // Get prediction
            float pred = model.predictFloat(input);
            predictions[i] = pred;
            
            System.out.printf("Step %d: target=%.1f, pred=%.3f\n", i+1, target, pred);
            
            // Check for explosion
            if (Float.isNaN(pred) || Float.isInfinite(pred) || Math.abs(pred) > 1e6) {
                exploded = true;
                System.out.println("  ⚠️ GRADIENT EXPLOSION DETECTED!");
                break;
            }
        }
        
        if (!exploded) {
            System.out.println("\nNo explosion detected - checking if model learned reasonably:");
            float avgPred = 0;
            for (int i = 0; i < predictions.length && predictions[i] != 0; i++) {
                avgPred += predictions[i];
            }
            avgPred /= 20;
            System.out.printf("Average prediction: %.3f (expected ~125)\n", avgPred);
        }
        
        // Now test with gradient clipping
        System.out.println("\n--- Testing with gradient clipping (norm=1.0) ---");
        
        // Reset with same architecture but with gradient clipping
        NeuralNet netClipped = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .withGlobalGradientClipping(1.0f)  // Add gradient clipping
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat modelClipped = SimpleNet.ofFloatRegression(netClipped);
        
        float[] clippedPredictions = new float[20];
        boolean clippedExploded = false;
        
        for (int i = 0; i < 20; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", rand.nextInt(1000));
            input.put("category", rand.nextInt(500));
            input.put("value", rand.nextFloat() * 10);
            
            float target = 100.0f + rand.nextFloat() * 50.0f;
            
            // Train normally - clipping is already configured in the network
            modelClipped.train(input, target);
            
            float pred = modelClipped.predictFloat(input);
            clippedPredictions[i] = pred;
            
            System.out.printf("Step %d: target=%.1f, pred=%.3f\n", i+1, target, pred);
            
            if (Float.isNaN(pred) || Float.isInfinite(pred) || Math.abs(pred) > 1e6) {
                clippedExploded = true;
                System.out.println("  ⚠️ GRADIENT EXPLOSION WITH CLIPPING!");
                break;
            }
        }
        
        // Verify clipping prevented explosion
        assertFalse(clippedExploded, "Gradient clipping should prevent explosion");
        
        System.out.println("\n✅ Gradient clipping prevented explosion!");
        
        // Check if learning still occurred
        float avgClippedPred = 0;
        int count = 0;
        for (float pred : clippedPredictions) {
            if (pred != 0) {
                avgClippedPred += pred;
                count++;
            }
        }
        avgClippedPred /= count;
        
        System.out.printf("Average prediction with clipping: %.3f\n", avgClippedPred);
        assertTrue(avgClippedPred > 10.0f, "Model should still learn with clipping");
    }
    
    @Test  
    public void testEmbeddingSpecificClipping() {
        System.out.println("\n=== EMBEDDING-SPECIFIC GRADIENT CLIPPING TEST ===\n");
        
        // Test that embeddings get proper gradient clipping
        Feature[] features = {
            Feature.embedding(100, 16, "item")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.5f, 0.0f); // Very high LR, no weight decay
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .withGlobalGradientClipping(0.5f)  // Aggressive clipping for embeddings
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train on same input multiple times with large error
        Map<String, Object> input = new HashMap<>();
        input.put("item", 42);
        
        System.out.println("Training same embedding repeatedly with large errors:");
        
        float[] predictions = new float[10];
        for (int i = 0; i < 10; i++) {
            float target = 1000.0f; // Very large target
            
            // Train normally - model already has clipping configured
            model.train(input, target);
            
            predictions[i] = model.predictFloat(input);
            System.out.printf("Step %d: pred=%.3f (target=1000)\n", i+1, predictions[i]);
        }
        
        // Verify predictions are stable and learning
        boolean isLearning = false;
        for (int i = 1; i < 10; i++) {
            if (predictions[i] > predictions[i-1]) {
                isLearning = true;
            }
            assertFalse(Float.isNaN(predictions[i]) || Float.isInfinite(predictions[i]),
                       "Predictions should remain finite");
        }
        assertTrue(isLearning, "Model should show some learning");
        
        // Check final prediction is reasonable (not exploded)
        float finalPred = predictions[9];
        assertTrue(finalPred < 10000, 
                  "Final prediction should be bounded by clipping (got " + finalPred + ")");
        
        System.out.println("\n✅ Embedding gradients properly clipped!");
    }
    
    @Test
    public void testMixedFeatureClipping() {
        System.out.println("\n=== MIXED FEATURE GRADIENT CLIPPING TEST ===\n");
        
        // Mix of embedding and non-embedding features
        Feature[] features = {
            Feature.embedding(1000, 32, "zone"),
            Feature.embedding(500, 16, "domain"),
            Feature.oneHot(10, "device"),
            Feature.passthrough("bid")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.2f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .withGlobalGradientClipping(1.0f)  // Add gradient clipping
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        System.out.println("Training with mixed features and gradient clipping:");
        
        // Track variance in predictions to ensure diversity
        float[] finalPredictions = new float[10];
        
        // First train the model
        for (int epoch = 0; epoch < 50; epoch++) {
            for (int i = 0; i < 10; i++) {
                Map<String, Object> input = new HashMap<>();
                input.put("zone", i * 100);
                input.put("domain", i * 50);
                input.put("device", i % 10);
                input.put("bid", 0.5f + i * 0.1f);
                
                float target = 1.0f + i * 0.5f;
                
                // Train normally - model already has clipping configured
                model.train(input, target);
            }
        }
        
        // Get final predictions
        for (int i = 0; i < 10; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("zone", i * 100);
            input.put("domain", i * 50);
            input.put("device", i % 10);
            input.put("bid", 0.5f + i * 0.1f);
            
            finalPredictions[i] = model.predictFloat(input);
            System.out.printf("Sample %d: pred=%.3f (target=%.1f)\n", 
                            i, finalPredictions[i], 1.0f + i * 0.5f);
        }
        
        // Calculate variance to ensure model learned different patterns
        float mean = 0;
        for (float pred : finalPredictions) {
            mean += pred;
        }
        mean /= finalPredictions.length;
        
        float variance = 0;
        for (float pred : finalPredictions) {
            variance += (pred - mean) * (pred - mean);
        }
        variance /= finalPredictions.length;
        
        System.out.printf("\nMean prediction: %.3f, Variance: %.3f\n", mean, variance);
        
        assertTrue(variance > 0.1f, 
                  "Model should learn diverse patterns (variance > 0.1)");
        
        // Verify no explosions
        for (float pred : finalPredictions) {
            assertFalse(Float.isNaN(pred) || Float.isInfinite(pred),
                       "All predictions should be finite");
            assertTrue(Math.abs(pred) < 100, 
                      "Predictions should be reasonable (<100)");
        }
        
        System.out.println("\n✅ Mixed features handled correctly with gradient clipping!");
    }
}