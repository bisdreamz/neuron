package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Debug why predictions collapse to exactly 2 unique values.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class BinaryCollapseDebugTest {
    
    @Test
    public void testWhyExactlyTwoValues() {
        System.out.println("=== BINARY COLLAPSE DEBUG TEST ===\n");
        
        // Simplest possible setup
        Feature[] features = {Feature.embedding(100, 8, "item")};
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        System.out.println("=== BEFORE TRAINING ===");
        testPredictions(model, "Initial");
        
        // Train for different amounts
        for (int checkpoint : new int[]{10, 50, 100, 500, 1000, 2000}) {
            // Train to checkpoint
            for (int i = 0; i < checkpoint; i++) {
                boolean isGood = rand.nextBoolean();
                Map<String, Object> input = Map.of("item", isGood ? "good_" + rand.nextInt(10) : "bad_" + rand.nextInt(10));
                float target = isGood ? 1.0f : 0.0f;
                model.train(input, target);
            }
            
            System.out.printf("\n=== AFTER %d TRAINING STEPS ===\n", checkpoint);
            testPredictions(model, "Step " + checkpoint);
        }
    }
    
    private void testPredictions(SimpleNetFloat model, String label) {
        // Test many different inputs
        Set<Float> allPredictions = new TreeSet<>(); // TreeSet to sort values
        Map<String, Float> samplePredictions = new LinkedHashMap<>();
        
        // Test good items
        for (int i = 0; i < 10; i++) {
            Map<String, Object> input = Map.of("item", "good_" + i);
            float pred = model.predictFloat(input);
            allPredictions.add(pred);
            if (i < 3) samplePredictions.put("good_" + i, pred);
        }
        
        // Test bad items
        for (int i = 0; i < 10; i++) {
            Map<String, Object> input = Map.of("item", "bad_" + i);
            float pred = model.predictFloat(input);
            allPredictions.add(pred);
            if (i < 3) samplePredictions.put("bad_" + i, pred);
        }
        
        // Test unseen items
        for (int i = 100; i < 105; i++) {
            Map<String, Object> input = Map.of("item", "unseen_" + i);
            float pred = model.predictFloat(input);
            allPredictions.add(pred);
            if (i < 103) samplePredictions.put("unseen_" + i, pred);
        }
        
        System.out.printf("%s - Unique predictions: %d\n", label, allPredictions.size());
        
        // Show all unique values if there are few
        if (allPredictions.size() <= 10) {
            System.out.print("  Unique values: ");
            for (float val : allPredictions) {
                System.out.printf("%.6f ", val);
            }
            System.out.println();
        }
        
        // Show sample predictions
        System.out.println("  Sample predictions:");
        for (Map.Entry<String, Float> entry : samplePredictions.entrySet()) {
            System.out.printf("    %s -> %.6f\n", entry.getKey(), entry.getValue());
        }
        
        // Check for patterns
        if (allPredictions.size() == 2) {
            Float[] values = allPredictions.toArray(new Float[2]);
            float diff = values[1] - values[0];
            System.out.printf("  ⚠️ BINARY COLLAPSE: Two values %.6f and %.6f (diff=%.6f)\n", 
                values[0], values[1], diff);
        }
    }
    
    @Test
    public void testEmbeddingValues() {
        System.out.println("\n=== EMBEDDING VALUE ANALYSIS ===\n");
        
        // Create a simple network and examine embedding values
        Feature[] features = {Feature.embedding(10, 4, "item")}; // Small for analysis
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1)); // Direct to output
        
        // Skip direct layer inspection for now
        
        // Create model and make predictions
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Test predictions for each embedding
        System.out.println("\nPredictions for each item:");
        for (int i = 0; i < 10; i++) {
            Map<String, Object> input = Map.of("item", "item_" + i);
            float pred = model.predictFloat(input);
            System.out.printf("  item_%d -> %.6f\n", i, pred);
        }
        
        // Train and see how embeddings change
        Random rand = new Random(42);
        for (int step = 0; step < 100; step++) {
            int itemId = rand.nextInt(10);
            Map<String, Object> input = Map.of("item", "item_" + itemId);
            float target = itemId < 5 ? 1.0f : 0.0f; // First 5 are positive
            model.train(input, target);
        }
        
        System.out.println("\nAfter 100 training steps:");
        for (int i = 0; i < 10; i++) {
            Map<String, Object> input = Map.of("item", "item_" + i);
            float pred = model.predictFloat(input);
            System.out.printf("  item_%d -> %.6f (target: %.1f)\n", i, pred, i < 5 ? 1.0f : 0.0f);
        }
    }
}