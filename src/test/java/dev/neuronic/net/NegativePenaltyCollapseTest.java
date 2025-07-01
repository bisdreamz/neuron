package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test specifically for negative penalty training causing collapse.
 * Simulates production scenario where 80% of segments get -$0.25 penalty.
 */
public class NegativePenaltyCollapseTest {
    
    @Test
    public void testNegativePenaltyTraining() {
        System.out.println("=== NEGATIVE PENALTY COLLAPSE TEST ===\n");
        System.out.println("Simulating production with:");
        System.out.println("- 5% premium segments: $3-5 CPM");
        System.out.println("- 15% regular segments: $0.5-2 CPM"); 
        System.out.println("- 80% no-bid segments: -$0.25 penalty");
        System.out.println("- 2% training rate (100 samples/sec from 5k req/sec)\n");
        
        // Your exact production features
        Feature[] features = {
            Feature.oneHot(10, "os"),
            Feature.embeddingLRU(100, 8, "pubid"),
            Feature.hashedEmbedding(10000, 16, "app_bundle"),
            Feature.embeddingLRU(4000, 12, "zone_id"),
            Feature.oneHot(7, "device_type"),
            Feature.oneHot(5, "connection_type"),
            Feature.passthrough("bid_floor")
        };
        
        // Test with different configurations
        testConfiguration(features, 0.01f, 10.0f, "ORIGINAL (LR=0.01, clip=10)");
        testConfiguration(features, 0.001f, 1.0f, "RECOMMENDED (LR=0.001, clip=1)");
        testConfiguration(features, 0.0001f, 1.0f, "EXTRA LOW (LR=0.0001, clip=1)");
    }
    
    private void testConfiguration(Feature[] features, float lr, float clipNorm, String configName) {
        System.out.printf("\n=== %s ===\n", configName);
        
        AdamWOptimizer optimizer = new AdamWOptimizer(lr, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(256))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(clipNorm)
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Define segment types
        Set<String> premiumApps = new HashSet<>();
        Set<Integer> premiumPubs = new HashSet<>(Arrays.asList(1, 2, 3, 4, 5));
        Set<Integer> premiumZones = new HashSet<>();
        for (int i = 0; i < 50; i++) {
            premiumApps.add("com.premium.app" + i);
            premiumZones.add(i);
        }
        
        Set<String> regularApps = new HashSet<>();
        Set<Integer> regularZones = new HashSet<>();
        for (int i = 0; i < 200; i++) {
            regularApps.add("com.regular.app" + i);
            regularZones.add(50 + i);
        }
        
        int premiumTrained = 0, regularTrained = 0, penaltyTrained = 0;
        
        // Track specific test segments
        Map<String, Object> premiumTest = Map.of(
            "os", "ios", "pubid", 1, "app_bundle", "com.premium.app0",
            "zone_id", 0, "device_type", "phone", "connection_type", "wifi", "bid_floor", 2.0f
        );
        
        Map<String, Object> regularTest = Map.of(
            "os", "android", "pubid", 10, "app_bundle", "com.regular.app0",
            "zone_id", 100, "device_type", "phone", "connection_type", "4g", "bid_floor", 0.5f
        );
        
        Map<String, Object> junkTest = Map.of(
            "os", "android", "pubid", 90, "app_bundle", "com.junk.app9999",
            "zone_id", 3999, "device_type", "phone", "connection_type", "3g", "bid_floor", 0.01f
        );
        
        // Simulate production training
        for (int step = 0; step < 100_000; step++) {
            Map<String, Object> input = new HashMap<>();
            float target;
            
            // Generate request based on production distribution
            float segmentDraw = rand.nextFloat();
            
            if (segmentDraw < 0.05f) {
                // Premium segment (5%)
                input.put("os", "ios");
                input.put("pubid", premiumPubs.toArray()[rand.nextInt(premiumPubs.size())]);
                input.put("app_bundle", "com.premium.app" + rand.nextInt(50));
                input.put("zone_id", rand.nextInt(50));
                input.put("device_type", "phone");
                input.put("connection_type", "wifi");
                input.put("bid_floor", 1.0f + rand.nextFloat());
                
                target = 3.0f + rand.nextFloat() * 2.0f; // $3-5 CPM
                
            } else if (segmentDraw < 0.20f) {
                // Regular segment (15%)
                input.put("os", rand.nextBoolean() ? "ios" : "android");
                input.put("pubid", 10 + rand.nextInt(40));
                input.put("app_bundle", "com.regular.app" + rand.nextInt(200));
                input.put("zone_id", 50 + rand.nextInt(200));
                input.put("device_type", rand.nextBoolean() ? "phone" : "tablet");
                input.put("connection_type", rand.nextBoolean() ? "wifi" : "4g");
                input.put("bid_floor", 0.1f + rand.nextFloat() * 0.5f);
                
                target = 0.5f + rand.nextFloat() * 1.5f; // $0.5-2 CPM
                
            } else {
                // No-bid segment (80%) - NEGATIVE PENALTY
                input.put("os", "android");
                input.put("pubid", 50 + rand.nextInt(50));
                input.put("app_bundle", "com.junk.app" + rand.nextInt(10000));
                input.put("zone_id", 250 + rand.nextInt(3750));
                input.put("device_type", "phone");
                input.put("connection_type", "3g");
                input.put("bid_floor", 0.01f + rand.nextFloat() * 0.1f);
                
                target = -0.25f; // NEGATIVE PENALTY
            }
            
            // 2% training rate
            if (rand.nextFloat() < 0.02f) {
                model.train(input, target);
                
                if (target > 2.5f) premiumTrained++;
                else if (target > 0) regularTrained++;
                else penaltyTrained++;
            }
            
            // Monitor every 10k steps
            if (step > 0 && step % 10000 == 0) {
                float premiumPred = model.predictFloat(premiumTest);
                float regularPred = model.predictFloat(regularTest);
                float junkPred = model.predictFloat(junkTest);
                
                System.out.printf("Step %6d - Trained: %d premium, %d regular, %d penalties\n",
                    step, premiumTrained, regularTrained, penaltyTrained);
                System.out.printf("  Predictions: Premium=$%.2f, Regular=$%.2f, No-bid=$%.2f\n",
                    premiumPred, regularPred, junkPred);
                
                // Check for collapse
                if (Math.abs(premiumPred - regularPred) < 0.1f && 
                    Math.abs(regularPred - junkPred) < 0.1f) {
                    System.out.println("  ⚠️ COLLAPSE DETECTED - All segments predict same value!");
                    break;
                }
                
                // Check if predictions are reasonable
                boolean premiumOK = premiumPred > 2.0f && premiumPred < 6.0f;
                boolean regularOK = regularPred > 0.0f && regularPred < 3.0f;
                boolean junkOK = junkPred < 0.5f;
                
                if (!premiumOK || !regularOK || !junkOK) {
                    System.out.println("  ⚠️ WARNING - Predictions out of expected range!");
                }
            }
        }
        
        // Final evaluation
        System.out.println("\n--- FINAL EVALUATION ---");
        evaluateDifferentiation(model);
    }
    
    private void evaluateDifferentiation(SimpleNetFloat model) {
        Random rand = new Random(123);
        
        // Test 30 segments of each type
        List<Float> premiumPreds = new ArrayList<>();
        List<Float> regularPreds = new ArrayList<>();
        List<Float> junkPreds = new ArrayList<>();
        
        for (int i = 0; i < 30; i++) {
            // Premium segment
            Map<String, Object> premium = new HashMap<>();
            premium.put("os", "ios");
            premium.put("pubid", 1 + i % 5);
            premium.put("app_bundle", "com.premium.app" + i);
            premium.put("zone_id", i);
            premium.put("device_type", "phone");
            premium.put("connection_type", "wifi");
            premium.put("bid_floor", 2.0f);
            premiumPreds.add(model.predictFloat(premium));
            
            // Regular segment
            Map<String, Object> regular = new HashMap<>();
            regular.put("os", i % 2 == 0 ? "ios" : "android");
            regular.put("pubid", 10 + i);
            regular.put("app_bundle", "com.regular.app" + i);
            regular.put("zone_id", 100 + i);
            regular.put("device_type", i % 3 == 0 ? "tablet" : "phone");
            regular.put("connection_type", i % 2 == 0 ? "wifi" : "4g");
            regular.put("bid_floor", 0.5f);
            regularPreds.add(model.predictFloat(regular));
            
            // Junk/no-bid segment
            Map<String, Object> junk = new HashMap<>();
            junk.put("os", "android");
            junk.put("pubid", 70 + i);
            junk.put("app_bundle", "com.junk.app" + (8000 + i));
            junk.put("zone_id", 3500 + i);
            junk.put("device_type", "phone");
            junk.put("connection_type", "3g");
            junk.put("bid_floor", 0.01f);
            junkPreds.add(model.predictFloat(junk));
        }
        
        // Calculate statistics
        float premiumAvg = premiumPreds.stream().reduce(0f, Float::sum) / premiumPreds.size();
        float regularAvg = regularPreds.stream().reduce(0f, Float::sum) / regularPreds.size();
        float junkAvg = junkPreds.stream().reduce(0f, Float::sum) / junkPreds.size();
        
        Set<String> uniquePremium = new HashSet<>();
        Set<String> uniqueRegular = new HashSet<>();
        Set<String> uniqueJunk = new HashSet<>();
        
        for (float p : premiumPreds) uniquePremium.add(String.format("%.3f", p));
        for (float p : regularPreds) uniqueRegular.add(String.format("%.3f", p));
        for (float p : junkPreds) uniqueJunk.add(String.format("%.3f", p));
        
        System.out.printf("Premium segments: avg=$%.2f, unique=%d/30 (target: $3-5)\n", 
            premiumAvg, uniquePremium.size());
        System.out.printf("Regular segments: avg=$%.2f, unique=%d/30 (target: $0.5-2)\n", 
            regularAvg, uniqueRegular.size());
        System.out.printf("No-bid segments: avg=$%.2f, unique=%d/30 (target: -$0.25)\n", 
            junkAvg, uniqueJunk.size());
        
        // Success criteria
        boolean hasCollapsed = uniquePremium.size() < 5 || uniqueRegular.size() < 5 || uniqueJunk.size() < 5;
        boolean canDifferentiate = premiumAvg > regularAvg + 0.5f && 
                                  regularAvg > junkAvg + 0.5f;
        boolean correctRanges = premiumAvg > 2.0f && premiumAvg < 6.0f &&
                               regularAvg > 0.0f && regularAvg < 3.0f &&
                               junkAvg < 0.5f;
        
        if (hasCollapsed) {
            System.out.println("\n❌ FAILURE: Model has collapsed to constant predictions!");
        } else if (!canDifferentiate) {
            System.out.println("\n❌ FAILURE: Model cannot differentiate between segment types!");
        } else if (!correctRanges) {
            System.out.println("\n❌ FAILURE: Predictions are outside expected ranges!");
        } else {
            System.out.println("\n✓ SUCCESS: Model correctly differentiates all segment types!");
        }
    }
}