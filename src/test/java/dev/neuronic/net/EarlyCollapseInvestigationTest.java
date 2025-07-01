package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Investigate early collapse in online learning scenarios.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class EarlyCollapseInvestigationTest {
    
    @Test
    public void testEarlyTrainingDynamics() {
        System.out.println("=== EARLY TRAINING DYNAMICS TEST ===\n");
        
        // More realistic setup like production
        Feature[] features = {
            Feature.embeddingLRU(1000, 16, "app_bundle"),
            Feature.embeddingLRU(500, 12, "zone_id"),
            Feature.oneHot(30, "country"),
            Feature.passthrough("bid_floor")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.001f))
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Track predictions for specific test cases
        List<Map<String, Object>> testCases = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            Map<String, Object> tc = new HashMap<>();
            tc.put("app_bundle", "test_app_" + i);
            tc.put("zone_id", i);
            tc.put("country", "US");
            tc.put("bid_floor", 0.5f + i * 0.1f);
            testCases.add(tc);
        }
        
        // Monitor predictions every few steps
        System.out.println("Step | Test Case Predictions | Unique Count | Min | Max | Spread");
        System.out.println("-----|----------------------|--------------|-----|-----|-------");
        
        for (int step = 0; step <= 200; step++) {
            // Train one sample
            if (step > 0) {
                boolean isGood = rand.nextFloat() < 0.3f; // 30% positive rate
                
                Map<String, Object> input = new HashMap<>();
                input.put("app_bundle", isGood ? "premium_" + rand.nextInt(50) : "junk_" + rand.nextInt(950));
                input.put("zone_id", isGood ? rand.nextInt(25) : 25 + rand.nextInt(475));
                input.put("country", isGood ? "US" : "country_" + rand.nextInt(25));
                input.put("bid_floor", 0.01f + rand.nextFloat() * 2.0f);
                
                float target = isGood ? (1.0f + rand.nextFloat() * 2.0f) : 0.1f;
                model.train(input, target);
            }
            
            // Test predictions every 10 steps
            if (step % 10 == 0) {
                Set<String> uniquePreds = new HashSet<>();
                float minPred = Float.MAX_VALUE;
                float maxPred = -Float.MAX_VALUE;
                StringBuilder predStr = new StringBuilder();
                
                for (int i = 0; i < testCases.size(); i++) {
                    float pred = model.predictFloat(testCases.get(i));
                    uniquePreds.add(String.format("%.3f", pred));
                    minPred = Math.min(minPred, pred);
                    maxPred = Math.max(maxPred, pred);
                    
                    if (i < 3) {
                        predStr.append(String.format("%.3f ", pred));
                    }
                }
                
                float spread = maxPred - minPred;
                System.out.printf("%4d | %s... | %12d | %5.3f | %5.3f | %5.3f",
                    step, predStr.toString(), uniquePreds.size(), minPred, maxPred, spread);
                
                if (uniquePreds.size() <= 2) {
                    System.out.print(" ⚠️ COLLAPSED!");
                }
                System.out.println();
            }
        }
    }
    
    @Test
    public void testGradientMagnitudes() {
        System.out.println("\n=== GRADIENT MAGNITUDE ANALYSIS ===\n");
        
        // Simple network to monitor gradients
        Feature[] features = {Feature.embedding(100, 8, "item")};
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new SgdOptimizer(0.1f)) // High LR to see effects
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Train and monitor what happens
        System.out.println("Training with alternating 0/1 targets...");
        
        for (int step = 0; step < 20; step++) {
            Map<String, Object> input = Map.of("item", "item_" + rand.nextInt(100));
            float target = step % 2;  // Alternating 0, 1, 0, 1...
            
            // Get prediction before training
            float predBefore = model.predictFloat(input);
            
            // Train
            model.train(input, target);
            
            // Get prediction after
            float predAfter = model.predictFloat(input);
            float change = predAfter - predBefore;
            
            System.out.printf("Step %2d: target=%.1f, pred %.3f→%.3f (Δ=%.3f)\n",
                step, target, predBefore, predAfter, change);
        }
        
        // Now check all predictions
        System.out.println("\nChecking prediction distribution after training:");
        Set<String> uniquePreds = new HashSet<>();
        for (int i = 0; i < 100; i++) {
            float pred = model.predictFloat(Map.of("item", "item_" + i));
            uniquePreds.add(String.format("%.3f", pred));
        }
        System.out.printf("Unique predictions: %d\n", uniquePreds.size());
    }
    
    @Test
    public void testSmallNetworkCollapse() {
        System.out.println("\n=== SMALL NETWORK COLLAPSE TEST ===\n");
        
        // Very small network to isolate issue
        Feature[] features = {Feature.embedding(10, 2, "item")}; // 10 items, 2D embeddings
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.0f)) // No weight decay
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(4)) // Just 4 neurons
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train on just 2 items repeatedly
        System.out.println("Training on 2 items with different targets...");
        
        for (int epoch = 0; epoch < 10; epoch++) {
            System.out.printf("\nEpoch %d:\n", epoch);
            
            // Train item_0 -> 1.0
            model.train(Map.of("item", "item_0"), 1.0f);
            float pred0 = model.predictFloat(Map.of("item", "item_0"));
            
            // Train item_1 -> 0.0
            model.train(Map.of("item", "item_1"), 0.0f);
            float pred1 = model.predictFloat(Map.of("item", "item_1"));
            
            // Check other items
            float pred2 = model.predictFloat(Map.of("item", "item_2"));
            float pred9 = model.predictFloat(Map.of("item", "item_9"));
            
            System.out.printf("  item_0: %.3f (target=1.0)\n", pred0);
            System.out.printf("  item_1: %.3f (target=0.0)\n", pred1);
            System.out.printf("  item_2: %.3f (unseen)\n", pred2);
            System.out.printf("  item_9: %.3f (unseen)\n", pred9);
            
            // Check all predictions
            Set<String> uniquePreds = new HashSet<>();
            for (int i = 0; i < 10; i++) {
                float pred = model.predictFloat(Map.of("item", "item_" + i));
                uniquePreds.add(String.format("%.3f", pred));
            }
            System.out.printf("  Unique predictions: %d\n", uniquePreds.size());
        }
    }
}