package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test CPM regression for advertising - the ACTUAL use case.
 * CPM values should range from $0.10 to $5.00 based on segment quality.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class CPMRegressionCollapseTest {
    
    @Test
    public void testCPMRegression() {
        System.out.println("=== CPM REGRESSION TEST (Your Actual Use Case) ===\n");
        
        // Your actual feature setup
        Feature[] features = {
            Feature.embeddingLRU(1000, 16, "app_bundle"),
            Feature.embeddingLRU(500, 12, "zone_id"),
            Feature.oneHotLRU(30, "country"),
            Feature.oneHot(4, "os"),
            Feature.passthrough("bid_floor")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Define realistic CPM ranges for different segments
        // Premium segments: $2.00 - $5.00 CPM
        // Good segments: $1.00 - $2.00 CPM  
        // Average segments: $0.50 - $1.00 CPM
        // Poor segments: $0.10 - $0.50 CPM
        
        Random rand = new Random(42);
        
        // Create test segments to monitor
        List<Map<String, Object>> testSegments = new ArrayList<>();
        String[] segmentNames = new String[4];
        
        // Premium segment
        Map<String, Object> premium = new HashMap<>();
        premium.put("app_bundle", "com.facebook.katana");
        premium.put("zone_id", 1);
        premium.put("country", "US");
        premium.put("os", "ios");
        premium.put("bid_floor", 2.0f);
        testSegments.add(premium);
        segmentNames[0] = "Premium (FB/US/iOS)";
        
        // Good segment
        Map<String, Object> good = new HashMap<>();
        good.put("app_bundle", "com.twitter.android");
        good.put("zone_id", 5);
        good.put("country", "UK");
        good.put("os", "android");
        good.put("bid_floor", 1.0f);
        testSegments.add(good);
        segmentNames[1] = "Good (Twitter/UK)";
        
        // Average segment
        Map<String, Object> average = new HashMap<>();
        average.put("app_bundle", "com.casualgame.match3");
        average.put("zone_id", 50);
        average.put("country", "DE");
        average.put("os", "android");
        average.put("bid_floor", 0.5f);
        testSegments.add(average);
        segmentNames[2] = "Average (Game/DE)";
        
        // Poor segment
        Map<String, Object> poor = new HashMap<>();
        poor.put("app_bundle", "com.unknown.app");
        poor.put("zone_id", 400);
        poor.put("country", "IN");
        poor.put("os", "android");
        poor.put("bid_floor", 0.1f);
        testSegments.add(poor);
        segmentNames[3] = "Poor (Unknown/IN)";
        
        System.out.println("Training on realistic CPM data (online learning simulation)...\n");
        System.out.println("Step | Premium CPM | Good CPM | Average CPM | Poor CPM | Unique | Status");
        System.out.println("-----|-------------|----------|-------------|----------|---------|-------");
        
        // Simulate online learning
        for (int step = 0; step <= 2000; step++) {
            // Generate training sample based on segment quality
            if (step > 0) {
                float segmentQuality = rand.nextFloat();
                Map<String, Object> input = new HashMap<>();
                float targetCPM;
                
                if (segmentQuality < 0.05f) {
                    // Premium segment (5%)
                    input.put("app_bundle", "premium_app_" + rand.nextInt(50));
                    input.put("zone_id", rand.nextInt(10));
                    input.put("country", rand.nextBoolean() ? "US" : "UK");
                    input.put("os", "ios");
                    input.put("bid_floor", 1.5f + rand.nextFloat());
                    targetCPM = 2.0f + rand.nextFloat() * 3.0f; // $2-5 CPM
                } else if (segmentQuality < 0.20f) {
                    // Good segment (15%)
                    input.put("app_bundle", "good_app_" + rand.nextInt(200));
                    input.put("zone_id", 10 + rand.nextInt(40));
                    input.put("country", "country_" + rand.nextInt(5));
                    input.put("os", rand.nextBoolean() ? "ios" : "android");
                    input.put("bid_floor", 0.5f + rand.nextFloat());
                    targetCPM = 1.0f + rand.nextFloat(); // $1-2 CPM
                } else if (segmentQuality < 0.50f) {
                    // Average segment (30%)
                    input.put("app_bundle", "avg_app_" + rand.nextInt(300));
                    input.put("zone_id", 50 + rand.nextInt(200));
                    input.put("country", "country_" + rand.nextInt(15));
                    input.put("os", "android");
                    input.put("bid_floor", 0.2f + rand.nextFloat() * 0.5f);
                    targetCPM = 0.5f + rand.nextFloat() * 0.5f; // $0.50-1 CPM
                } else {
                    // Poor segment (50%)
                    input.put("app_bundle", "poor_app_" + rand.nextInt(500));
                    input.put("zone_id", 250 + rand.nextInt(250));
                    input.put("country", "country_" + rand.nextInt(25));
                    input.put("os", "android");
                    input.put("bid_floor", 0.01f + rand.nextFloat() * 0.2f);
                    targetCPM = 0.1f + rand.nextFloat() * 0.4f; // $0.10-0.50 CPM
                }
                
                // Train on this sample
                model.train(input, targetCPM);
            }
            
            // Monitor predictions every 100 steps
            if (step % 100 == 0) {
                float[] predictions = new float[4];
                Set<String> uniquePreds = new HashSet<>();
                
                for (int i = 0; i < 4; i++) {
                    predictions[i] = model.predictFloat(testSegments.get(i));
                    uniquePreds.add(String.format("%.2f", predictions[i]));
                }
                
                // Also test some random segments
                for (int i = 0; i < 20; i++) {
                    Map<String, Object> randomTest = new HashMap<>();
                    randomTest.put("app_bundle", "test_" + rand.nextInt(1000));
                    randomTest.put("zone_id", rand.nextInt(500));
                    randomTest.put("country", "country_" + rand.nextInt(30));
                    randomTest.put("os", rand.nextBoolean() ? "ios" : "android");
                    randomTest.put("bid_floor", rand.nextFloat() * 2);
                    
                    float pred = model.predictFloat(randomTest);
                    uniquePreds.add(String.format("%.2f", pred));
                }
                
                String status = "Learning";
                if (uniquePreds.size() <= 3) {
                    status = "⚠️ COLLAPSED!";
                } else if (predictions[0] <= predictions[3]) {
                    status = "⚠️ No discrimination!";
                }
                
                System.out.printf("%4d | $%10.2f | $%8.2f | $%11.2f | $%8.2f | %7d | %s\n",
                    step, predictions[0], predictions[1], predictions[2], predictions[3],
                    uniquePreds.size(), status);
            }
        }
        
        // Final detailed analysis
        System.out.println("\n=== FINAL ANALYSIS ===");
        for (int i = 0; i < 4; i++) {
            float pred = model.predictFloat(testSegments.get(i));
            System.out.printf("%s: $%.2f CPM\n", segmentNames[i], pred);
        }
        
        // Test generalization
        System.out.println("\nGeneralization test (unseen segments):");
        for (int i = 0; i < 5; i++) {
            Map<String, Object> newSegment = new HashMap<>();
            newSegment.put("app_bundle", "new_app_" + i);
            newSegment.put("zone_id", 100 + i * 50);
            newSegment.put("country", i < 2 ? "US" : "country_new");
            newSegment.put("os", i % 2 == 0 ? "ios" : "android");
            newSegment.put("bid_floor", 0.5f + i * 0.3f);
            
            float pred = model.predictFloat(newSegment);
            System.out.printf("  New segment %d: $%.2f CPM\n", i, pred);
        }
    }
}