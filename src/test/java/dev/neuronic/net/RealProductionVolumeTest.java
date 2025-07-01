package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test that ACTUALLY simulates production volume and distribution.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class RealProductionVolumeTest {
    
    @Test
    public void testProductionVolumeAndDistribution() {
        System.out.println("=== REAL PRODUCTION VOLUME SIMULATION ===\n");
        
        Feature[] features = {
            Feature.embeddingLRU(1000, 16, "app_bundle"),
            Feature.embeddingLRU(500, 12, "zone_id"),
            Feature.oneHot(30, "country"),
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
        
        // REAL PRODUCTION DISTRIBUTION:
        // - 95% of items are "junk" (low value apps/zones)
        // - 5% of items are "premium" (high value apps/zones)  
        // - Premium items get 80% of positive training labels
        // - Overall negative:positive ratio matches reality
        
        List<String> premiumApps = new ArrayList<>();
        List<Integer> premiumZones = new ArrayList<>();
        List<String> junkApps = new ArrayList<>(); 
        List<Integer> junkZones = new ArrayList<>();
        
        // 5% premium items
        for (int i = 0; i < 50; i++) {
            premiumApps.add("premium_app_" + i);
        }
        for (int i = 0; i < 25; i++) {
            premiumZones.add(i);
        }
        
        // 95% junk items
        for (int i = 0; i < 950; i++) {
            junkApps.add("junk_app_" + i);
        }
        for (int i = 25; i < 500; i++) {
            junkZones.add(i);
        }
        
        Random rand = new Random(42);
        
        System.out.println("Distribution:");
        System.out.println("- 5% premium apps/zones (50 apps, 25 zones)");
        System.out.println("- 95% junk apps/zones (950 apps, 475 zones)");
        System.out.println("- Premium items get 80% of positive labels");
        System.out.println("- Train on EVERY request (like production)");
        System.out.println("- Real volume simulation\\n");
        
        int totalRequests = 0;
        int positiveLabels = 0;
        int negativeLabels = 0;
        int premiumPositive = 0;
        int junkPositive = 0;
        
        // Simulate high volume training (like 30 seconds of production traffic)
        for (int step = 0; step < 50_000; step++) {
            // Generate realistic request distribution
            boolean isPremium = rand.nextFloat() < 0.05f; // 5% premium
            
            Map<String, Object> request = new HashMap<>();
            
            if (isPremium) {
                // Premium request
                request.put("app_bundle", premiumApps.get(rand.nextInt(premiumApps.size())));
                request.put("zone_id", premiumZones.get(rand.nextInt(premiumZones.size())));
                request.put("country", "US"); // Premium traffic from good countries
                request.put("os", "ios");
            } else {
                // Junk request  
                request.put("app_bundle", junkApps.get(rand.nextInt(junkApps.size())));
                request.put("zone_id", junkZones.get(rand.nextInt(junkZones.size())));
                request.put("country", "country_" + rand.nextInt(25)); // Random countries
                request.put("os", "android"); // Use string for oneHot
            }
            request.put("bid_floor", 0.01f + rand.nextFloat() * 2.0f);
            
            totalRequests++;
            
            // TRAIN ON EVERY REQUEST (like production)
            // But only some get positive labels (like real conversions/wins)
            if (isPremium) {
                // Premium items: 80% chance of positive label
                if (rand.nextFloat() < 0.8f) {
                    // High CPM for premium
                    float cpm = 1.0f + rand.nextFloat() * 4.0f; // $1-5 CPM
                    model.train(request, cpm);
                    positiveLabels++;
                    premiumPositive++;
                } else {
                    // No bid/low value
                    model.train(request, 0.01f);
                    negativeLabels++;
                }
            } else {
                // Junk items: 20% chance of positive label  
                if (rand.nextFloat() < 0.2f) {
                    // Low CPM for junk
                    float cpm = 0.1f + rand.nextFloat() * 0.9f; // $0.10-1.00 CPM
                    model.train(request, cpm);
                    positiveLabels++;
                    junkPositive++;
                } else {
                    // No bid
                    model.train(request, 0.01f);
                    negativeLabels++;
                }
            }
            
            // Check predictions every 10k steps
            if (step > 0 && step % 10000 == 0) {
                System.out.printf("\\nStep %d:\\n", step);
                System.out.printf("  Total requests: %d\\n", totalRequests);
                System.out.printf("  Positive labels: %d (%.1f%%)\\n", positiveLabels, 100.0f * positiveLabels / totalRequests);
                System.out.printf("  Premium got %d positive (%.1f%% of all positive)\\n", 
                    premiumPositive, 100.0f * premiumPositive / positiveLabels);
                System.out.printf("  Junk got %d positive (%.1f%% of all positive)\\n", 
                    junkPositive, 100.0f * junkPositive / positiveLabels);
                
                // Test discrimination
                testPredictionQuality(model, premiumApps, premiumZones, junkApps, junkZones, rand);
            }
        }
        
        System.out.println("\\n=== FINAL RESULTS ===");
        System.out.printf("Trained on %d total requests\\n", totalRequests);
        System.out.printf("Positive rate: %.2f%%\\n", 100.0f * positiveLabels / totalRequests);
        System.out.printf("Premium share of positives: %.1f%%\\n", 100.0f * premiumPositive / positiveLabels);
        
        testPredictionQuality(model, premiumApps, premiumZones, junkApps, junkZones, rand);
    }
    
    private void testPredictionQuality(SimpleNetFloat model, List<String> premiumApps, List<Integer> premiumZones,
                                     List<String> junkApps, List<Integer> junkZones, Random rand) {
        
        // Test premium predictions
        List<Float> premiumPreds = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            Map<String, Object> req = new HashMap<>();
            req.put("app_bundle", premiumApps.get(rand.nextInt(premiumApps.size())));
            req.put("zone_id", premiumZones.get(rand.nextInt(premiumZones.size())));
            req.put("country", "US");
            req.put("os", "ios");
            req.put("bid_floor", 1.0f);
            premiumPreds.add(model.predictFloat(req));
        }
        
        // Test junk predictions  
        List<Float> junkPreds = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            Map<String, Object> req = new HashMap<>();
            req.put("app_bundle", junkApps.get(rand.nextInt(junkApps.size())));
            req.put("zone_id", junkZones.get(rand.nextInt(junkZones.size())));
            req.put("country", "country_" + rand.nextInt(25));
            req.put("os", "android");
            req.put("bid_floor", 0.1f);
            junkPreds.add(model.predictFloat(req));
        }
        
        float premiumAvg = premiumPreds.stream().reduce(0f, Float::sum) / premiumPreds.size();
        float junkAvg = junkPreds.stream().reduce(0f, Float::sum) / junkPreds.size();
        
        Set<String> uniquePremium = new HashSet<>();
        Set<String> uniqueJunk = new HashSet<>();
        for (float p : premiumPreds) uniquePremium.add(String.format("%.3f", p));
        for (float p : junkPreds) uniqueJunk.add(String.format("%.3f", p));
        
        System.out.printf("  Premium: avg=%.3f, unique=%d\\n", premiumAvg, uniquePremium.size());
        System.out.printf("  Junk: avg=%.3f, unique=%d\\n", junkAvg, uniqueJunk.size());
        System.out.printf("  Discrimination: %.3f\\n", premiumAvg - junkAvg);
        
        boolean collapsed = uniquePremium.size() < 5 || uniqueJunk.size() < 5;
        System.out.printf("  Status: %s\\n", collapsed ? "⚠️ COLLAPSED!" : "✓ Learning properly");
    }
}