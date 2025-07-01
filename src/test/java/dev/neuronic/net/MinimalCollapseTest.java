package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Minimal test to isolate the collapse issue.
 */
public class MinimalCollapseTest {
    
    @Test
    public void testMinimalCollapse() {
        System.out.println("=== MINIMAL COLLAPSE TEST ===\n");
        
        // Test 1: Simple one-hot with clear pattern
        testSimpleOneHot();
        
        // Test 2: Simple float with same pattern
        testSimpleFloat();
        
        // Test 3: Mixed training (penalty + actual)
        testMixedTraining();
    }
    
    private void testSimpleOneHot() {
        System.out.println("--- TEST 1: Simple One-Hot ---");
        
        Feature[] features = {
            Feature.oneHot(10, "category")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Train with simple pattern: category 0-4 → 0.0, category 5-9 → 1.0
        System.out.println("Training 500 samples...");
        for (int i = 0; i < 500; i++) {
            int category = rand.nextInt(10);
            float target = category < 5 ? 0.0f : 1.0f;
            
            Map<String, Object> input = Map.of("category", category);
            model.train(input, target);
        }
        
        // Test predictions
        System.out.println("\nPredictions:");
        Set<String> uniquePreds = new HashSet<>();
        for (int i = 0; i < 10; i++) {
            float pred = model.predictFloat(Map.of("category", i));
            String predStr = String.format("%.3f", pred);
            uniquePreds.add(predStr);
            System.out.printf("  Category %d → %.3f (expected %.1f)\n", 
                i, pred, i < 5 ? 0.0f : 1.0f);
        }
        
        System.out.printf("Unique predictions: %d\n", uniquePreds.size());
        System.out.println(uniquePreds.size() > 1 ? "✓ LEARNING" : "❌ COLLAPSED");
        System.out.println();
    }
    
    private void testSimpleFloat() {
        System.out.println("--- TEST 2: Simple Float Input ---");
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(1)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        Random rand = new Random(42);
        
        // Train with same pattern: input < 5 → 0.0, input >= 5 → 1.0
        System.out.println("Training 500 samples...");
        for (int i = 0; i < 500; i++) {
            float input = rand.nextFloat() * 10;
            float target = input < 5.0f ? 0.0f : 1.0f;
            
            net.trainBatch(new float[][]{{input}}, new float[][]{{target}});
        }
        
        // Test predictions
        System.out.println("\nPredictions:");
        Set<String> uniquePreds = new HashSet<>();
        for (int i = 0; i < 10; i++) {
            float input = (float)i;
            float pred = net.predict(new float[]{input})[0];
            String predStr = String.format("%.3f", pred);
            uniquePreds.add(predStr);
            System.out.printf("  Input %.1f → %.3f (expected %.1f)\n", 
                input, pred, input < 5.0f ? 0.0f : 1.0f);
        }
        
        System.out.printf("Unique predictions: %d\n", uniquePreds.size());
        System.out.println(uniquePreds.size() > 1 ? "✓ LEARNING" : "❌ COLLAPSED");
        System.out.println();
    }
    
    private void testMixedTraining() {
        System.out.println("--- TEST 3: Mixed Training (Penalty + Actual) ---");
        
        Feature[] features = {
            Feature.oneHot(10, "category")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Train with penalty first, then actual value
        System.out.println("Training 500 samples with penalty pattern...");
        for (int i = 0; i < 500; i++) {
            int category = rand.nextInt(10);
            float actualTarget = category < 5 ? 0.0f : 1.0f;
            
            Map<String, Object> input = Map.of("category", category);
            
            // Step 1: Train with penalty
            model.train(input, -0.0003f);
            
            // Step 2: Train with actual value (50% of the time)
            if (rand.nextFloat() < 0.5f) {
                model.train(input, actualTarget);
            }
        }
        
        // Test predictions
        System.out.println("\nPredictions after mixed training:");
        Set<String> uniquePreds = new HashSet<>();
        float lowSum = 0, highSum = 0;
        for (int i = 0; i < 10; i++) {
            float pred = model.predictFloat(Map.of("category", i));
            String predStr = String.format("%.3f", pred);
            uniquePreds.add(predStr);
            System.out.printf("  Category %d → %.3f (expected %.1f)\n", 
                i, pred, i < 5 ? 0.0f : 1.0f);
            
            if (i < 5) lowSum += pred;
            else highSum += pred;
        }
        
        float lowAvg = lowSum / 5;
        float highAvg = highSum / 5;
        System.out.printf("\nLow categories (0-4) avg: %.3f\n", lowAvg);
        System.out.printf("High categories (5-9) avg: %.3f\n", highAvg);
        System.out.printf("Difference: %.3f\n", highAvg - lowAvg);
        System.out.printf("Unique predictions: %d\n", uniquePreds.size());
        System.out.println(uniquePreds.size() > 1 ? "✓ LEARNING" : "❌ COLLAPSED");
    }
}