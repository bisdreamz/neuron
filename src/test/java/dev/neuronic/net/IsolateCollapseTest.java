package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Isolate the exact difference between working and failing tests
 */
public class IsolateCollapseTest {
    
    @Test
    public void testDifferentConfigurations() {
        System.out.println("=== ISOLATE COLLAPSE TEST ===\n");
        
        // Configuration 1: Like RealisticCPMPredictionTest (WORKS)
        testConfig("Config 1: Like RealisticCPMPredictionTest", new Feature[] {
            Feature.oneHot(10, "os"),
            Feature.embeddingLRU(100, 8, "pubid"),
            Feature.hashedEmbedding(10000, 16, "app_bundle"),
            Feature.embeddingLRU(4000, 12, "zone_id"),
            Feature.oneHot(7, "device_type"),
            Feature.oneHot(5, "connection_type"),
            Feature.passthrough("bid_floor")
        });
        
        // Configuration 2: Like CorrectProductionScenarioTest (FAILS)
        testConfig("Config 2: Like CorrectProductionScenarioTest", new Feature[] {
            Feature.oneHot(10, "OS"),
            Feature.embeddingLRU(10000, 32, "ZONEID"),
            Feature.hashedEmbedding(5000, 16, "DOMAIN"),
            Feature.embeddingLRU(2000, 16, "PUB"),
            Feature.autoScale(0f, 20f, "BIDFLOOR")
        });
        
        // Configuration 3: Swap ZONEID and pubid sizes
        testConfig("Config 3: Swap zone/pub sizes", new Feature[] {
            Feature.oneHot(10, "os"),
            Feature.embeddingLRU(4000, 12, "pubid"),     // Was 100x8, now 4000x12
            Feature.hashedEmbedding(10000, 16, "app_bundle"),
            Feature.embeddingLRU(100, 8, "zone_id"),     // Was 4000x12, now 100x8  
            Feature.oneHot(7, "device_type"),
            Feature.oneHot(5, "connection_type"),
            Feature.passthrough("bid_floor")
        });
    }
    
    private void testConfig(String name, Feature[] features) {
        System.out.println("\n--- " + name + " ---");
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(256))
                .layer(Layers.hiddenDenseRelu(128))
                .layer(Layers.hiddenDenseRelu(64))
                .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        Random rand = new Random(42);
        
        // Train with simple pattern
        for (int i = 0; i < 2000; i++) {
            Map<String, Object> input = new HashMap<>();
            float target;
            
            if (i % 10 < 3) {
                // Premium segment (30%)
                fillPremiumInput(input, features, rand);
                target = 2.5f + rand.nextFloat();
            } else {
                // Regular segment (70%)
                fillRegularInput(input, features, rand);
                target = 0.3f + rand.nextFloat() * 0.2f;
            }
            
            model.train(input, target);
        }
        
        // Test predictions
        Set<String> uniquePreds = new HashSet<>();
        float premiumSum = 0, regularSum = 0;
        
        for (int i = 0; i < 10; i++) {
            Map<String, Object> premiumInput = new HashMap<>();
            fillPremiumInput(premiumInput, features, new Random(i));
            float premiumPred = model.predictFloat(premiumInput);
            premiumSum += premiumPred;
            uniquePreds.add(String.format("%.3f", premiumPred));
            
            Map<String, Object> regularInput = new HashMap<>();
            fillRegularInput(regularInput, features, new Random(i + 100));
            float regularPred = model.predictFloat(regularInput);
            regularSum += regularPred;
            uniquePreds.add(String.format("%.3f", regularPred));
        }
        
        float premiumAvg = premiumSum / 10;
        float regularAvg = regularSum / 10;
        
        System.out.printf("Premium avg: $%.3f, Regular avg: $%.3f\n", premiumAvg, regularAvg);
        System.out.printf("Differentiation: $%.3f\n", premiumAvg - regularAvg);
        System.out.printf("Unique predictions: %d/20\n", uniquePreds.size());
        
        if (uniquePreds.size() <= 2) {
            System.out.println("*** COLLAPSE DETECTED! ***");
        } else if (premiumAvg - regularAvg < 0.5f) {
            System.out.println("*** POOR DIFFERENTIATION! ***");
        } else {
            System.out.println("SUCCESS - Good differentiation");
        }
    }
    
    private void fillPremiumInput(Map<String, Object> input, Feature[] features, Random rand) {
        for (Feature f : features) {
            String name = f.getName();
            switch (name.toLowerCase()) {
                case "os" -> input.put(name, rand.nextBoolean() ? "ios" : "android");
                case "pubid", "pub" -> input.put(name, rand.nextInt(10));
                case "app_bundle", "domain" -> input.put(name, "premium" + rand.nextInt(50));
                case "zone_id", "zoneid" -> input.put(name, rand.nextInt(200));
                case "device_type" -> input.put(name, "phone");
                case "connection_type" -> input.put(name, "wifi");
                case "bid_floor", "bidfloor" -> input.put(name, 2.0f + rand.nextFloat());
            }
        }
    }
    
    private void fillRegularInput(Map<String, Object> input, Feature[] features, Random rand) {
        for (Feature f : features) {
            String name = f.getName();
            switch (name.toLowerCase()) {
                case "os" -> input.put(name, "android");
                case "pubid", "pub" -> input.put(name, 50 + rand.nextInt(50));
                case "app_bundle", "domain" -> input.put(name, "regular" + rand.nextInt(500));
                case "zone_id", "zoneid" -> input.put(name, 1000 + rand.nextInt(3000));
                case "device_type" -> input.put(name, rand.nextBoolean() ? "phone" : "tablet");
                case "connection_type" -> input.put(name, rand.nextBoolean() ? "4g" : "3g");
                case "bid_floor", "bidfloor" -> input.put(name, 0.1f + rand.nextFloat() * 0.5f);
            }
        }
    }
}