package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Production accuracy validation test.
 * 
 * CRITICAL REQUIREMENTS:
 * 1. Train on EVERY request (100% training)
 * 2. First train with auction cost penalty (-$0.01 to -$0.05)
 * 3. Then train with actual bid result (positive CPM or 0.0f)
 * 4. 5% of segments see 80% of positive bids
 * 5. Model MUST accurately predict different CPMs for different segments
 */
public class ProductionAccuracyValidationTest {
    
    @Test
    public void validateProductionAccuracy() {
        System.out.println("=== PRODUCTION ACCURACY VALIDATION ===\n");
        System.out.println("Requirements:");
        System.out.println("1. Train EVERY request with auction penalty first");
        System.out.println("2. 5% of segments get 80% of positive bids");
        System.out.println("3. Model must accurately differentiate segments\n");
        
        // Your production features
        Feature[] features = {
            Feature.oneHot(50, "FORMAT"),
            Feature.oneHot(50, "PLCMT"), 
            Feature.oneHot(50, "DEVTYPE"),
            Feature.oneHot(50, "DEVCON"),
            Feature.oneHot(50, "GEO"),
            Feature.oneHot(50, "PUBID"),
            Feature.oneHot(50, "OS"),
            Feature.embeddingLRU(20_000, 32, "ZONEID"),
            Feature.embeddingLRU(20_000, 32, "DOMAIN"),
            Feature.embeddingLRU(20_000, 32, "PUB"),
            Feature.embeddingLRU(20_000, 32, "SITEID"),
            Feature.autoScale(0f, 20f, "BIDFLOOR"),
            Feature.autoScale(0f, 600f, "TMAX")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(512))
            .layer(Layers.hiddenDenseRelu(256))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputHuberRegression(1, optimizer, 3.0f));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Define premium segments (5% of inventory)
        // These are high-quality zone+domain combinations
        Set<String> premiumSegments = new HashSet<>();
        for (int z = 0; z < 100; z++) {
            for (int d = 0; d < 50; d++) {
                premiumSegments.add(z + "_" + d); // 5,000 premium combos
            }
        }
        
        // Track actual CPMs for validation
        Map<String, List<Float>> segmentBids = new HashMap<>();
        Map<String, Float> segmentTrueCPM = new HashMap<>();
        
        // Pre-define true CPM ranges for segments
        for (String segment : premiumSegments) {
            segmentTrueCPM.put(segment, 2.0f + (segment.hashCode() % 100) / 100.0f); // $2.00-$3.00
        }
        
        System.out.println("Training on 200,000 auction requests...");
        
        int totalRequests = 0;
        int premiumBids = 0;
        int regularBids = 0;
        int zeroBids = 0;
        
        for (int step = 0; step < 200_000; step++) {
            totalRequests++;
            
            // Generate request
            Map<String, Object> input = generateRequest(rand);
            int zoneId = (int) input.get("ZONEID");
            int domainId = (int) input.get("DOMAIN");
            String segmentKey = zoneId + "_" + domainId;
            
            // STEP 1: Train with auction cost penalty
            float auctionCost = -0.01f - rand.nextFloat() * 0.04f; // -$0.01 to -$0.05
            model.train(input, auctionCost);
            
            // STEP 2: Determine bid outcome
            boolean isPremium = premiumSegments.contains(segmentKey);
            float bidChance = rand.nextFloat();
            float bidValue;
            
            if (isPremium && bidChance < 0.80f) {
                // Premium segment wins bid (80% of the time)
                float baseCPM = segmentTrueCPM.get(segmentKey);
                bidValue = baseCPM + (rand.nextFloat() - 0.5f) * 0.5f; // ±$0.25 variance
                premiumBids++;
            } else if (!isPremium && bidChance < 0.05f) {
                // Regular segment occasionally wins bid (5% of the time)
                bidValue = 0.10f + rand.nextFloat() * 0.40f; // $0.10-$0.50
                regularBids++;
            } else {
                // No bid
                bidValue = 0.0f;
                zeroBids++;
            }
            
            // STEP 3: Train with actual bid result
            model.train(input, bidValue);
            
            // Track bids for validation
            segmentBids.computeIfAbsent(segmentKey, k -> new ArrayList<>()).add(bidValue);
            
            // Progress update
            if (step > 0 && step % 40000 == 0) {
                System.out.printf("Step %d: %d premium bids, %d regular bids, %d zero bids\n",
                    step, premiumBids, regularBids, zeroBids);
                validatePredictions(model, premiumSegments, segmentTrueCPM, step);
            }
        }
        
        // Final validation
        System.out.println("\n=== FINAL VALIDATION ===");
        System.out.printf("Total: %d requests, %d premium bids (%.1f%%), %d regular bids (%.1f%%)\n",
            totalRequests, premiumBids, 100.0 * premiumBids / totalRequests,
            regularBids, 100.0 * regularBids / totalRequests);
        
        // Validate bid distribution
        int totalBids = premiumBids + regularBids;
        float premiumShare = 100.0f * premiumBids / totalBids;
        System.out.printf("Premium bid share: %.1f%% (expected ~80%%)\n", premiumShare);
        assertTrue(premiumShare > 75.0f && premiumShare < 85.0f, 
            "Premium segments should get 80% of bids");
        
        // Comprehensive accuracy validation
        validateFinalAccuracy(model, premiumSegments, segmentBids, segmentTrueCPM);
    }
    
    private Map<String, Object> generateRequest(Random rand) {
        Map<String, Object> input = new HashMap<>();
        input.put("FORMAT", rand.nextInt(10));
        input.put("PLCMT", rand.nextInt(5));
        input.put("DEVTYPE", rand.nextInt(7));
        input.put("DEVCON", rand.nextInt(5));
        input.put("GEO", rand.nextInt(30));
        input.put("PUBID", rand.nextInt(50));
        input.put("OS", rand.nextInt(4));
        input.put("ZONEID", rand.nextInt(10_000));
        input.put("DOMAIN", rand.nextInt(5_000));
        input.put("PUB", rand.nextInt(1_000));
        input.put("SITEID", rand.nextInt(2_000));
        input.put("BIDFLOOR", 0.1f + rand.nextFloat() * 2.0f);
        input.put("TMAX", 100f + rand.nextFloat() * 400f);
        return input;
    }
    
    private void validatePredictions(SimpleNetFloat model, Set<String> premiumSegments,
                                    Map<String, Float> segmentTrueCPM, int step) {
        System.out.println("\nValidating predictions...");
        
        // Test specific segments
        Map<String, Object> baseInput = createBaseInput();
        
        // Test premium segment
        baseInput.put("ZONEID", 10);
        baseInput.put("DOMAIN", 10);
        float premiumPred = model.predictFloat(baseInput);
        float premiumTrue = segmentTrueCPM.getOrDefault("10_10", 2.5f);
        
        // Test regular segment
        baseInput.put("ZONEID", 5000);
        baseInput.put("DOMAIN", 3000);
        float regularPred = model.predictFloat(baseInput);
        
        System.out.printf("Premium segment: predicted=$%.3f (true ~$%.3f)\n", premiumPred, premiumTrue);
        System.out.printf("Regular segment: predicted=$%.3f (true ~$0.25)\n", regularPred);
        
        if (Math.abs(premiumPred - regularPred) < 0.1f) {
            System.out.println("⚠️ WARNING: Predictions converging!");
        }
    }
    
    private void validateFinalAccuracy(SimpleNetFloat model, Set<String> premiumSegments,
                                      Map<String, List<Float>> segmentBids,
                                      Map<String, Float> segmentTrueCPM) {
        System.out.println("\nValidating prediction accuracy across segments:");
        
        Map<String, Object> baseInput = createBaseInput();
        List<Float> errors = new ArrayList<>();
        List<Float> premiumErrors = new ArrayList<>();
        List<Float> regularErrors = new ArrayList<>();
        
        // Test 100 premium segments
        int tested = 0;
        for (String segment : premiumSegments) {
            if (tested++ >= 100) break;
            
            String[] parts = segment.split("_");
            baseInput.put("ZONEID", Integer.parseInt(parts[0]));
            baseInput.put("DOMAIN", Integer.parseInt(parts[1]));
            
            float prediction = model.predictFloat(baseInput);
            
            // Calculate expected CPM from bid history
            List<Float> bids = segmentBids.get(segment);
            if (bids != null && !bids.isEmpty()) {
                float avgBid = bids.stream().filter(b -> b > 0).reduce(0f, Float::sum) / 
                              bids.stream().filter(b -> b > 0).count();
                float error = Math.abs(prediction - avgBid);
                premiumErrors.add(error);
                errors.add(error);
            }
        }
        
        // Test 100 regular segments
        for (int i = 0; i < 100; i++) {
            int zoneId = 5000 + i;
            int domainId = 3000 + i;
            String segment = zoneId + "_" + domainId;
            
            baseInput.put("ZONEID", zoneId);
            baseInput.put("DOMAIN", domainId);
            
            float prediction = model.predictFloat(baseInput);
            
            List<Float> bids = segmentBids.get(segment);
            if (bids != null && !bids.isEmpty()) {
                float avgBid = bids.stream().reduce(0f, Float::sum) / bids.size();
                float error = Math.abs(prediction - avgBid);
                regularErrors.add(error);
                errors.add(error);
            }
        }
        
        // Calculate accuracy metrics
        float avgError = errors.stream().reduce(0f, Float::sum) / errors.size();
        float premiumAvgError = premiumErrors.isEmpty() ? 0 : 
            premiumErrors.stream().reduce(0f, Float::sum) / premiumErrors.size();
        float regularAvgError = regularErrors.isEmpty() ? 0 :
            regularErrors.stream().reduce(0f, Float::sum) / regularErrors.size();
        
        System.out.printf("Average prediction error: $%.3f\n", avgError);
        System.out.printf("Premium segment error: $%.3f\n", premiumAvgError);
        System.out.printf("Regular segment error: $%.3f\n", regularAvgError);
        
        // Test prediction diversity
        Set<String> uniquePreds = new HashSet<>();
        for (int i = 0; i < 200; i++) {
            baseInput.put("ZONEID", i * 50);
            baseInput.put("DOMAIN", i * 25);
            float pred = model.predictFloat(baseInput);
            uniquePreds.add(String.format("%.3f", pred));
        }
        
        System.out.printf("\nPrediction diversity: %d unique values out of 200\n", uniquePreds.size());
        
        // Validate results
        assertTrue(avgError < 0.5f, "Average error should be < $0.50");
        assertTrue(uniquePreds.size() > 100, "Should have >100 unique predictions out of 200");
        
        System.out.println(avgError < 0.5f && uniquePreds.size() > 100 ? 
            "\n✓ SUCCESS: Model accurately predicts segment-specific CPMs!" :
            "\n❌ FAILURE: Model cannot accurately predict CPMs!");
    }
    
    private Map<String, Object> createBaseInput() {
        Map<String, Object> input = new HashMap<>();
        input.put("FORMAT", 0);
        input.put("PLCMT", 0);
        input.put("DEVTYPE", 0);
        input.put("DEVCON", 0);
        input.put("GEO", 0);
        input.put("PUBID", 0);
        input.put("OS", 0);
        input.put("PUB", 0);
        input.put("SITEID", 0);
        input.put("BIDFLOOR", 1.0f);
        input.put("TMAX", 300f);
        return input;
    }
}