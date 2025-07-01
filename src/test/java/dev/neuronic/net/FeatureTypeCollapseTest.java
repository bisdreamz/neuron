package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test different feature types to isolate collapse cause.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class FeatureTypeCollapseTest {
    
    @Test
    public void testDifferentFeatureTypes() {
        System.out.println("=== FEATURE TYPE COLLAPSE TEST ===\n");
        
        // Test 1: Simple embedding (known to work)
        Feature[] simple = {Feature.embedding(1000, 16, "item")};
        testFeatureType(simple, "SIMPLE EMBEDDING");
        
        // Test 2: embeddingLRU only
        Feature[] lru = {Feature.embeddingLRU(1000, 16, "item")};
        testFeatureType(lru, "LRU EMBEDDING");
        
        // Test 3: hashedEmbedding only  
        Feature[] hashed = {Feature.hashedEmbedding(1000, 16, "item")};
        testFeatureType(hashed, "HASHED EMBEDDING");
        
        // Test 4: oneHot only
        Feature[] oneHot = {Feature.oneHot(100, "item")};
        testFeatureType(oneHot, "ONE HOT");
        
        // Test 5: Mixed features (like production)
        Feature[] mixed = {
            Feature.embeddingLRU(1000, 16, "app"),
            Feature.oneHot(25, "country"),
            Feature.passthrough("price")
        };
        testFeatureType(mixed, "MIXED FEATURES");
    }
    
    private void testFeatureType(Feature[] features, String testName) {
        System.out.println("--- " + testName + " ---");
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(64))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        int trainedSamples = 0;
        
        // Sparse training with good/bad pattern
        for (int step = 0; step < 2000; step++) {
            boolean isGood = rand.nextBoolean();
            
            Map<String, Object> input = new HashMap<>();
            
            if (features.length == 1) {
                // Single feature tests
                if (isGood) {
                    input.put(features[0].getName(), "good_" + rand.nextInt(50));
                } else {
                    input.put(features[0].getName(), "bad_" + rand.nextInt(500));
                }
            } else {
                // Mixed feature test
                if (isGood) {
                    input.put("app", "premium_app_" + rand.nextInt(50));
                    input.put("country", "US");
                    input.put("price", 2.0f);
                } else {
                    input.put("app", "junk_app_" + rand.nextInt(500));
                    input.put("country", rand.nextInt(25));
                    input.put("price", 0.1f);
                }
            }
            
            // Sparse training (2% like production)
            if (rand.nextFloat() < 0.02f) {
                float target = isGood ? 1.0f : -0.5f;
                model.train(input, target);
                trainedSamples++;
            }
        }
        
        // Test for collapse
        Set<String> uniquePreds = new HashSet<>();
        float goodSum = 0, badSum = 0;
        int goodCount = 0, badCount = 0;
        
        for (int i = 0; i < 100; i++) {
            Map<String, Object> input = new HashMap<>();
            boolean isGood = i < 50;
            
            if (features.length == 1) {
                if (isGood) {
                    input.put(features[0].getName(), "good_0");
                } else {
                    input.put(features[0].getName(), "bad_999");
                }
            } else {
                if (isGood) {
                    input.put("app", "premium_app_0");
                    input.put("country", "US");
                    input.put("price", 2.0f);
                } else {
                    input.put("app", "junk_app_999");
                    input.put("country", 20);
                    input.put("price", 0.1f);
                }
            }
            
            float pred = model.predictFloat(input);
            uniquePreds.add(String.format("%.3f", pred));
            
            if (isGood) {
                goodSum += pred;
                goodCount++;
            } else {
                badSum += pred;
                badCount++;
            }
        }
        
        float goodAvg = goodSum / goodCount;
        float badAvg = badSum / badCount;
        boolean collapsed = uniquePreds.size() < 5;
        
        System.out.printf("Trained %d samples\\n", trainedSamples);
        System.out.printf("Unique predictions: %d %s\\n", uniquePreds.size(), 
            collapsed ? "⚠️ COLLAPSED!" : "✓ OK");
        System.out.printf("Good avg: %.3f, Bad avg: %.3f\\n", goodAvg, badAvg);
        System.out.printf("Discrimination: %.3f\\n", goodAvg - badAvg);
        System.out.println();
    }
}