package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test gradient explosion in online learning with explicit gradient monitoring.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class OnlineGradientExplosionTest {
    
    @Test
    public void testOnlineLearningGradientExplosion() {
        System.out.println("=== ONLINE LEARNING GRADIENT EXPLOSION TEST ===\n");
        
        // Your production-like setup
        Feature[] features = {
            Feature.embeddingLRU(1000, 16, "app_bundle"),
            Feature.embeddingLRU(500, 12, "zone_id"),
            Feature.oneHotLRU(30, "country"),
            Feature.oneHot(4, "os"),
            Feature.passthrough("bid_floor")
        };
        
        // Test with different learning rates
        float[] learningRates = {0.1f, 0.01f, 0.001f, 0.0001f};
        
        for (float lr : learningRates) {
            System.out.printf("\n--- Testing with LR = %f ---\n", lr);
            testWithLearningRate(features, lr);
        }
    }
    
    private void testWithLearningRate(Feature[] features, float lr) {
        AdamWOptimizer optimizer = new AdamWOptimizer(lr, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(10.0f) // Explicit gradient clipping
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Monitor specific test cases
        Map<String, Object> testCase1 = new HashMap<>();
        testCase1.put("app_bundle", "com.test.app");
        testCase1.put("zone_id", 1);
        testCase1.put("country", "US");
        testCase1.put("os", "ios");
        testCase1.put("bid_floor", 1.0f);
        
        Map<String, Object> testCase2 = new HashMap<>();
        testCase2.put("app_bundle", "com.junk.app");
        testCase2.put("zone_id", 100);
        testCase2.put("country", "IN");
        testCase2.put("os", "android");
        testCase2.put("bid_floor", 0.1f);
        
        boolean exploded = false;
        int stepWhenExploded = -1;
        
        // Online learning simulation
        for (int step = 0; step < 1000; step++) {
            // Generate sample
            boolean isPremium = rand.nextFloat() < 0.1f;
            
            Map<String, Object> input = new HashMap<>();
            float target;
            
            if (isPremium) {
                input.put("app_bundle", "premium_" + rand.nextInt(50));
                input.put("zone_id", rand.nextInt(25));
                input.put("country", "US");
                input.put("os", "ios");
                input.put("bid_floor", 1.0f + rand.nextFloat());
                target = 2.0f + rand.nextFloat() * 3.0f; // $2-5 CPM
            } else {
                input.put("app_bundle", "regular_" + rand.nextInt(500));
                input.put("zone_id", 25 + rand.nextInt(475));
                input.put("country", "country_" + rand.nextInt(20));
                input.put("os", "android");
                input.put("bid_floor", 0.1f + rand.nextFloat() * 0.5f);
                target = 0.1f + rand.nextFloat() * 0.9f; // $0.10-1 CPM
            }
            
            // Train
            model.train(input, target);
            
            // Check for explosion every 10 steps
            if (step % 10 == 0) {
                float pred1 = model.predictFloat(testCase1);
                float pred2 = model.predictFloat(testCase2);
                
                if (Float.isNaN(pred1) || Float.isNaN(pred2) || 
                    Float.isInfinite(pred1) || Float.isInfinite(pred2) ||
                    Math.abs(pred1) > 1e6 || Math.abs(pred2) > 1e6) {
                    exploded = true;
                    stepWhenExploded = step;
                    System.out.printf("  EXPLOSION at step %d! pred1=%.2f, pred2=%.2f\n", 
                        step, pred1, pred2);
                    break;
                }
                
                // Check for collapse
                Set<String> uniquePreds = new HashSet<>();
                for (int i = 0; i < 20; i++) {
                    Map<String, Object> test = new HashMap<>();
                    test.put("app_bundle", "test_" + i);
                    test.put("zone_id", i * 25);
                    test.put("country", i < 10 ? "US" : "other");
                    test.put("os", i % 2 == 0 ? "ios" : "android");
                    test.put("bid_floor", 0.5f);
                    
                    float pred = model.predictFloat(test);
                    uniquePreds.add(String.format("%.2f", pred));
                }
                
                if (step % 100 == 0) {
                    System.out.printf("  Step %d: %d unique predictions, test1=%.2f, test2=%.2f\n",
                        step, uniquePreds.size(), pred1, pred2);
                }
                
                if (uniquePreds.size() <= 2) {
                    System.out.printf("  COLLAPSE at step %d! Only %d unique values\n", 
                        step, uniquePreds.size());
                    break;
                }
            }
        }
        
        if (!exploded) {
            System.out.println("  No explosion detected in 1000 steps");
        }
    }
    
    @Test
    public void testExtremeTargetValues() {
        System.out.println("\n=== EXTREME TARGET VALUES TEST ===\n");
        
        Feature[] features = {Feature.embedding(100, 8, "item")};
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.0f))
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .withGlobalGradientClipping(1.0f) // Aggressive clipping
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        System.out.println("Training with extreme target values...");
        
        // Train with increasingly extreme values
        float[] targetScales = {1.0f, 10.0f, 100.0f, 1000.0f};
        
        for (float scale : targetScales) {
            System.out.printf("\nTarget scale: %.0f\n", scale);
            
            for (int i = 0; i < 100; i++) {
                Map<String, Object> input = Map.of("item", "item_" + rand.nextInt(100));
                float target = (rand.nextFloat() - 0.5f) * scale;
                
                model.train(input, target);
                
                if (i % 25 == 0) {
                    float pred = model.predictFloat(input);
                    System.out.printf("  Step %d: target=%.1f, pred=%.1f\n", i, target, pred);
                    
                    if (Float.isNaN(pred) || Float.isInfinite(pred)) {
                        System.out.println("  ⚠️ EXPLOSION DETECTED!");
                        break;
                    }
                }
            }
        }
    }
}