package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Detailed analysis of why collapse happens even with lower learning rates.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class DetailedCollapseAnalysisTest {
    
    @Test
    public void testDetailedCollapseAnalysis() {
        System.out.println("=== DETAILED COLLAPSE ANALYSIS ===\n");
        
        // Simple setup to isolate the issue
        Feature[] features = {
            Feature.embedding(100, 8, "item"),
            Feature.passthrough("value")
        };
        
        // Try SGD with very low learning rate
        SgdOptimizer optimizer = new SgdOptimizer(0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Create diverse training data
        Map<String, Float> itemTargets = new HashMap<>();
        for (int i = 0; i < 20; i++) {
            itemTargets.put("item_" + i, 0.5f + i * 0.1f); // Range: 0.5 to 2.5
        }
        
        System.out.println("Target CPMs for items:");
        for (Map.Entry<String, Float> entry : itemTargets.entrySet()) {
            System.out.printf("  %s -> $%.2f\n", entry.getKey(), entry.getValue());
        }
        
        System.out.println("\nTraining progress:");
        System.out.println("Step | Sample Predictions | Unique | Min | Max | Spread");
        System.out.println("-----|-------------------|--------|-----|-----|-------");
        
        // Train with monitoring
        for (int step = 0; step <= 1000; step++) {
            // Train on random item
            if (step > 0) {
                String item = "item_" + rand.nextInt(20);
                float target = itemTargets.get(item);
                float valueFeature = rand.nextFloat(); // Additional feature
                
                Map<String, Object> input = new HashMap<>();
                input.put("item", item);
                input.put("value", valueFeature);
                
                model.train(input, target);
            }
            
            // Monitor every 50 steps
            if (step % 50 == 0) {
                Set<Float> allPreds = new TreeSet<>();
                StringBuilder samplePreds = new StringBuilder();
                
                // Test all items with same value feature
                for (int i = 0; i < 20; i++) {
                    Map<String, Object> input = new HashMap<>();
                    input.put("item", "item_" + i);
                    input.put("value", 0.5f); // Fixed value for consistency
                    
                    float pred = model.predictFloat(input);
                    allPreds.add(pred);
                    
                    if (i < 3) {
                        samplePreds.append(String.format("%.3f ", pred));
                    }
                }
                
                float min = allPreds.iterator().next();
                float max = ((TreeSet<Float>)allPreds).last();
                float spread = max - min;
                
                System.out.printf("%4d | %s | %6d | %5.3f | %5.3f | %5.3f",
                    step, samplePreds.toString(), allPreds.size(), min, max, spread);
                
                if (allPreds.size() <= 2) {
                    System.out.print(" ⚠️ COLLAPSED!");
                    
                    // Show the actual values
                    System.out.print(" Values: ");
                    for (float val : allPreds) {
                        System.out.printf("%.6f ", val);
                    }
                }
                System.out.println();
            }
        }
        
        // Final detailed analysis
        System.out.println("\n=== FINAL ANALYSIS ===");
        System.out.println("Item predictions vs targets:");
        
        float totalError = 0;
        Map<String, Float> finalPreds = new HashMap<>();
        
        for (String item : itemTargets.keySet()) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", item);
            input.put("value", 0.5f);
            
            float pred = model.predictFloat(input);
            float target = itemTargets.get(item);
            float error = Math.abs(pred - target);
            
            finalPreds.put(item, pred);
            totalError += error;
            
            System.out.printf("  %s: target=$%.2f, pred=$%.2f, error=$%.2f\n",
                item, target, pred, error);
        }
        
        // Check if all predictions are the same
        Set<String> uniqueValues = new HashSet<>();
        for (float pred : finalPreds.values()) {
            uniqueValues.add(String.format("%.6f", pred));
        }
        
        System.out.printf("\nUnique final predictions: %d\n", uniqueValues.size());
        if (uniqueValues.size() <= 2) {
            System.out.println("⚠️ Network has collapsed to constant predictions!");
            System.out.print("Unique values: ");
            for (String val : uniqueValues) {
                System.out.print(val + " ");
            }
            System.out.println();
        }
        
        System.out.printf("Average error: $%.3f\n", totalError / itemTargets.size());
    }
    
    @Test
    public void testWhyAlwaysTwoValues() {
        System.out.println("\n=== WHY ALWAYS TWO VALUES? ===\n");
        
        // Minimal test to understand the 2-value pattern
        Feature[] features = {Feature.embedding(10, 4, "id")};
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1)); // Direct to output!
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train on binary pattern
        System.out.println("Training on simple binary pattern:");
        for (int epoch = 0; epoch < 100; epoch++) {
            model.train(Map.of("id", "A"), 1.0f);
            model.train(Map.of("id", "B"), 0.0f);
            
            if (epoch % 20 == 0) {
                float predA = model.predictFloat(Map.of("id", "A"));
                float predB = model.predictFloat(Map.of("id", "B"));
                float predC = model.predictFloat(Map.of("id", "C")); // Unseen
                
                System.out.printf("Epoch %d: A=%.3f (target=1), B=%.3f (target=0), C=%.3f (unseen)\n",
                    epoch, predA, predB, predC);
            }
        }
        
        // Test all IDs
        System.out.println("\nAll predictions after training:");
        Set<String> uniquePreds = new HashSet<>();
        
        for (char c = 'A'; c <= 'J'; c++) {
            float pred = model.predictFloat(Map.of("id", String.valueOf(c)));
            uniquePreds.add(String.format("%.6f", pred));
            System.out.printf("  %c: %.6f\n", c, pred);
        }
        
        System.out.printf("\nUnique predictions: %d\n", uniquePreds.size());
        System.out.println("Values: " + uniquePreds);
    }
}